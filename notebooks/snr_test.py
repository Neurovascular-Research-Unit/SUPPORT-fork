"""Build clean references, then train/test SNR sweep jobs across multiple GPUs.

The sweep has two stages separated by a dependency barrier. First, each raw
input TIFF is denoised once with SUPPORT to create a persistent ``original``
reference. Every configured power level is then simulated from that common
reference and denoised by its own SUPPORT model. Multi-GPU execution
happens at the job level and preserves the single-GPU model hyperparameters.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Union

import numpy as np
import tifffile

if TYPE_CHECKING:
    if __package__:
        from ..utils.salt import PhotonTransferEstimate
    else:
        from utils.salt import PhotonTransferEstimate


# Direct script runs default to the repository containing this file. The
# notebook can still supply its own repository path through SweepConfiguration.
path_support = Path(__file__).resolve().parents[1]


# Basic configuration. Model parameters are read from parameters.json.
dir_noisy = Path("/Volumes/Amreen_Mughal_lab/20260921_snrTest/01_data")
dimensionality = "3+1D"
power_percent = [100, 90, 80, 70, 60, 50, 40]

# ``None`` estimates one value per raw/SUPPORT pair after the original-image
# stage. Set a positive float to bypass estimation with an explicit override.
photons_per_unit = None
# A fixed additive detector/digitizer baseline stays constant while
# fluorescence, including out-of-focus background, scales with power. Low
# electronic-noise variance does not imply that this offset is zero. Prefer a
# dark-frame measurement; the fitted effective pedestal is an explicit,
# model-dependent fallback.
photon_zero_level = 0.0
use_estimated_photon_pedestal = False
photon_estimator_frame_start = 60
photon_estimator_frame_end_margin = 60
photon_estimator_max_pairs = 50
photon_estimator_sample_count = 40_000
photon_estimator_random_seed = 7
allow_provisional_photon_estimate = False

# Node-dependent settings. ``None`` uses every GPU visible to this process.
# To keep the sweep on one GPU, use ``gpu_ids = [0]``. IDs are logical positions
# within CUDA_VISIBLE_DEVICES (as assigned by Slurm, when applicable).
gpu_ids = None
n_cpu = 8  # Per GPU; total DataLoader workers scale with selected GPU count.

rm_tmp = True  # removes simulated power TIFFs; denoised originals are retained
rm_dir = False
checkpoints = False
resume = True
rm_intermediaries = True


@dataclass(frozen=True)
class SweepConfiguration:
    repository: Path
    input_root: Path
    parameters_path: Path
    output_root: Path
    dimensionality: str
    n_cpu: int
    rm_tmp: bool
    rm_dir: bool
    checkpoints: bool
    resume: bool
    rm_intermediaries: bool
    photons_per_unit: Optional[float] = None
    photon_estimator_frame_start: int = 60
    photon_estimator_frame_end_margin: int = 60
    photon_estimator_max_pairs: Optional[int] = 50
    photon_estimator_sample_count: int = 40_000
    photon_estimator_random_seed: int = 7
    allow_provisional_photon_estimate: bool = False
    photon_zero_level: float = 0.0
    use_estimated_photon_pedestal: bool = False


@dataclass(frozen=True)
class OriginalTask:
    index: int
    source: Path
    exp_name: str
    out_dir: Path
    output_path: Path
    power: Optional[int] = None


@dataclass(frozen=True)
class SweepTask:
    index: int
    source: Path
    reference_path: Path
    power: int
    exp_name: str
    out_dir: Path
    simulated_path: Path
    photons_per_unit: Optional[float] = None
    zero_photon_level: Optional[float] = None


@dataclass(frozen=True)
class TaskResult:
    task_index: int
    gpu_slot: int
    source: Path
    power: Optional[int]
    success: bool
    output_path: Optional[Path] = None
    error: str = ""
    stage: str = "power"


class ResourceCleanupError(RuntimeError):
    """Raised when a worker cannot safely continue after releasing a job."""


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("GPU IDs must be zero or greater")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return number


def finite_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite")
    return number


def power_percentage(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("power must be between 1 and 100")
    return number


def _ensure_repository_importable(repository: Path) -> None:
    """Put the repo root on ``sys.path`` for direct-script execution."""
    repository_token = str(Path(repository).expanduser().resolve())
    if repository_token not in sys.path:
        sys.path.insert(0, repository_token)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Denoise each source once, then run independent SUPPORT SNR sweep "
            "jobs concurrently with one single-GPU worker per selected GPU."
        )
    )
    parser.add_argument(
        "--repository",
        type=Path,
        help="SUPPORT repository root (default: repository containing this script)",
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        help="directory recursively containing source TIFFs",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="persistent sweep output root (default: REPOSITORY/notebooks/outputs)",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        help="SUPPORT parameter JSON (default: REPOSITORY/notebooks/parameters.json)",
    )
    parser.add_argument(
        "--dimensionality",
        choices=("2+1D", "3D", "3+1D"),
        help="SUPPORT mode (default: dimensionality configured in this file)",
    )
    parser.add_argument(
        "--power-percent",
        "--powers",
        dest="power_percent",
        type=power_percentage,
        nargs="+",
        metavar="PERCENT",
        help="power percentages to simulate (default: power_percent in this file)",
    )
    parser.add_argument(
        "--n-cpu",
        type=positive_int,
        help="DataLoader workers per selected GPU (default: n_cpu in this file)",
    )
    parser.add_argument(
        "--gpus",
        type=nonnegative_int,
        nargs="+",
        metavar="ID",
        help=(
            "logical CUDA device IDs to use (default: gpu_ids in this file, or "
            "every visible GPU when gpu_ids is None)"
        ),
    )
    parser.add_argument(
        "--photons-per-unit",
        type=positive_float,
        help=(
            "fixed SUPPORT-unit photon conversion; default estimates one value "
            "per raw/SUPPORT source pair"
        ),
    )
    parser.add_argument(
        "--photon-frame-start",
        type=nonnegative_int,
        help="first raw frame used for photon calibration",
    )
    parser.add_argument(
        "--photon-frame-end-margin",
        type=nonnegative_int,
        help=(
            "number of final frames excluded from photon calibration "
            "(default: configured value, normally 60)"
        ),
    )
    parser.add_argument(
        "--photon-max-pairs",
        type=positive_int,
        help=(
            "maximum adjacent, non-overlapping frame pairs distributed across "
            "the calibration window"
        ),
    )
    parser.add_argument(
        "--photon-samples",
        type=positive_int,
        help="maximum matched spatial coordinates sampled by the estimator",
    )
    parser.add_argument(
        "--allow-provisional-photon-estimate",
        action="store_true",
        help="continue when photon-transfer linearity diagnostics emit warnings",
    )
    parser.add_argument(
        "--photon-zero-level",
        type=finite_float,
        help=(
            "fixed additive detector baseline in SUPPORT units (default: "
            "photon_zero_level in this file, normally zero)"
        ),
    )
    parser.add_argument(
        "--use-estimated-photon-pedestal",
        action="store_true",
        help=(
            "use the photon-transfer fit's model-dependent effective pedestal; "
            "prefer a measured dark-frame value when available"
        ),
    )
    return parser


def visible_device_tokens(
    cuda_visible_devices: Optional[str], device_count: Optional[int] = None
) -> tuple[str, ...]:
    """Return the inherited CUDA device tokens without changing their meaning."""
    if cuda_visible_devices is not None:
        value = cuda_visible_devices.strip()
        if not value or value == "-1":
            return ()
        tokens = tuple(token.strip() for token in value.split(","))
        if any(not token for token in tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES contains an empty device token")
        if len(set(tokens)) != len(tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate device tokens")
        return tokens

    if device_count is None:
        raise ValueError("device_count is required when CUDA_VISIBLE_DEVICES is unset")
    if device_count < 0:
        raise ValueError("CUDA device count cannot be negative")
    return tuple(str(index) for index in range(device_count))


def select_gpu_slots(
    requested: Optional[Sequence[int]], visible_tokens: Sequence[str]
) -> tuple[tuple[int, str], ...]:
    """Map requested logical GPU slots to inherited CUDA device tokens."""
    tokens = tuple(visible_tokens)
    if not tokens:
        raise ValueError("no CUDA GPUs are visible")

    slots = tuple(range(len(tokens))) if requested is None else tuple(requested)
    if not slots:
        raise ValueError("at least one GPU must be selected")
    if any(isinstance(slot, bool) or not isinstance(slot, int) for slot in slots):
        raise TypeError("GPU IDs must be integers")
    if any(slot < 0 for slot in slots):
        raise ValueError("GPU IDs must be zero or greater")
    if len(set(slots)) != len(slots):
        raise ValueError("GPU IDs must be unique")

    unavailable = [slot for slot in slots if slot >= len(tokens)]
    if unavailable:
        raise ValueError(
            f"GPU ID {unavailable[0]} is unavailable; visible logical IDs are "
            f"0 through {len(tokens) - 1}"
        )
    return tuple((slot, tokens[slot]) for slot in slots)


def build_original_tasks(
    paths: Sequence[Path], output_root: Path
) -> tuple[OriginalTask, ...]:
    """Build one persistent SUPPORT-denoised reference job per raw source."""
    tasks = []
    destinations = {}
    output_root = Path(output_root).resolve()
    for path in paths:
        source = Path(path).resolve()
        exp_name = f"{source.stem}_original"
        out_dir = output_root / source.stem
        output_path = out_dir / exp_name / f"{exp_name}_SPRT.tif"
        previous = destinations.get(output_path)
        if previous is not None:
            raise ValueError(
                "SNR original-image output collision: "
                f"{previous} and {source} both map to {output_path}. "
                "Use unique input stems before running the parallel sweep."
            )
        destinations[output_path] = source
        tasks.append(
            OriginalTask(
                index=len(tasks),
                source=source,
                exp_name=exp_name,
                out_dir=out_dir,
                output_path=output_path,
            )
        )
    return tuple(tasks)


def build_sweep_tasks(
    paths: Sequence[Path],
    powers: Sequence[int],
    output_root: Path,
    reference_paths: Optional[Mapping[Path, Path]] = None,
) -> tuple[SweepTask, ...]:
    """Build deterministic power jobs derived from denoised references."""
    normalized_powers = tuple(powers)
    if not normalized_powers:
        raise ValueError("power_percent must contain at least one value")
    if any(
        isinstance(power, bool) or not isinstance(power, int)
        for power in normalized_powers
    ):
        raise TypeError("power percentages must be integers")
    if any(power < 1 or power > 100 for power in normalized_powers):
        raise ValueError("power percentages must be between 1 and 100")
    if len(set(normalized_powers)) != len(normalized_powers):
        raise ValueError("power percentages must be unique")

    output_root = Path(output_root).resolve()
    normalized_references = (
        {
            Path(source).resolve(): Path(reference).resolve()
            for source, reference in reference_paths.items()
        }
        if reference_paths is not None
        else {
            task.source: task.output_path
            for task in build_original_tasks(paths, output_root)
        }
    )

    tasks = []
    destinations = {}
    for source in paths:
        source = Path(source).resolve()
        try:
            reference_path = normalized_references[source]
        except KeyError as exc:
            raise ValueError(
                f"No denoised original-image path was provided for {source}"
            ) from exc
        for power in normalized_powers:
            # This namespace must remain distinct from the older raw-derived
            # sweep so resume=True cannot load checkpoints with stale provenance.
            exp_name = f"{source.stem}_original_{power:03d}"
            out_dir = output_root / source.stem
            simulated_path = out_dir / exp_name / f"{exp_name}.tif"
            previous = destinations.get(simulated_path)
            if previous is not None:
                raise ValueError(
                    "SNR sweep output collision: "
                    f"{previous} and {source} both map to {simulated_path}. "
                    "Use unique input stems before running the parallel sweep."
                )
            destinations[simulated_path] = source
            tasks.append(
                SweepTask(
                    index=len(tasks),
                    source=source,
                    reference_path=reference_path,
                    power=power,
                    exp_name=exp_name,
                    out_dir=out_dir,
                    simulated_path=simulated_path,
                )
            )
    return tuple(tasks)


def partition_tasks(
    tasks: Sequence[Union[OriginalTask, SweepTask]], gpu_slots: Sequence[int]
) -> dict[int, tuple[Union[OriginalTask, SweepTask], ...]]:
    """Assign jobs round-robin while keeping one serial lane per GPU."""
    slots = tuple(gpu_slots)
    if not slots:
        raise ValueError("at least one GPU slot is required")
    assignments = {slot: [] for slot in slots}
    for index, task in enumerate(tasks):
        assignments[slots[index % len(slots)]].append(task)
    return {slot: tuple(assigned) for slot, assigned in assignments.items()}


def _write_simulated_tiff(path: Path, stack: np.ndarray, axes: str) -> None:
    """Atomically publish a simulated stack so failed writes cannot be resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.tif", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        tifffile.imwrite(
            temporary_path,
            stack.astype(np.float32, copy=False),
            imagej=True,
            metadata={"axes": axes},
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_time_first_tiff(path: Path) -> np.ndarray:
    """Open a TIFF for calibration and move its metadata T axis first."""
    path = Path(path).resolve()
    with tifffile.TiffFile(path) as tif:
        axes = tif.series[0].axes
    try:
        array = tifffile.memmap(path, series=0, mode="r")
    except (OSError, ValueError):
        # Compressed/tiled TIFFs cannot be mapped in place. Decode them into an
        # unlinked temporary memmap instead of materializing a large movie in
        # parent-process RAM.
        array = tifffile.imread(path, series=0, out="memmap")

    if axes and "T" in axes:
        time_axis = axes.index("T")
        if time_axis != 0:
            array = np.moveaxis(array, time_axis, 0)
    elif array.ndim not in (3, 4):
        raise ValueError(
            f"Cannot identify a time axis for {path}: axes={axes!r}, "
            f"shape={array.shape}"
        )
    else:
        print(
            f"WARNING: {path.name} has axes={axes!r}; assuming axis 0 is time "
            "for photon calibration.",
            flush=True,
        )
    return array


def _estimate_source_photons_per_unit(
    source: Path,
    reference_path: Path,
    configuration: SweepConfiguration,
) -> PhotonTransferEstimate:
    """Calibrate one raw/original-SUPPORT pair in the parent process."""
    if configuration.dimensionality == "3D":
        raise ValueError(
            "Automatic photons_per_unit estimation requires repeated time "
            "frames; set an explicit photons_per_unit for static 3D data."
        )

    _ensure_repository_importable(configuration.repository)
    if __package__:
        from ..utils.salt import estimate_photons_per_unit
    else:
        from utils.salt import estimate_photons_per_unit

    print(
        f"Estimating photons_per_unit for {Path(source).name} from "
        f"{Path(reference_path).name}...",
        flush=True,
    )
    raw_stack = _read_time_first_tiff(source)
    support_stack = _read_time_first_tiff(reference_path)
    try:
        configured_end_margin = configuration.photon_estimator_frame_end_margin
        if isinstance(configured_end_margin, bool) or not isinstance(
            configured_end_margin, (int, np.integer)
        ):
            raise ValueError(
                "photon_estimator_frame_end_margin must be a nonnegative integer"
            )
        frame_end_margin = int(configured_end_margin)
        if frame_end_margin < 0:
            raise ValueError(
                "photon_estimator_frame_end_margin must be a nonnegative integer"
            )
        frame_stop = raw_stack.shape[0] - frame_end_margin
        pair_limit = configuration.photon_estimator_max_pairs
        pair_description = (
            "all available adjacent pairs"
            if pair_limit is None
            else f"up to {pair_limit} adjacent pairs"
        )
        print(
            "  calibration frame window: "
            f"[{configuration.photon_estimator_frame_start}, {frame_stop}) "
            f"of {raw_stack.shape[0]} frames; {pair_description} are distributed "
            "across the window",
            flush=True,
        )
        estimate = estimate_photons_per_unit(
            raw_stack,
            support_stack,
            frame_start=configuration.photon_estimator_frame_start,
            frame_stop=frame_stop,
            max_pairs=configuration.photon_estimator_max_pairs,
            sample_count=configuration.photon_estimator_sample_count,
            random_seed=configuration.photon_estimator_random_seed,
        )
    finally:
        del raw_stack
        del support_stack

    print(
        f"Photon calibration for {Path(source).name}: "
        f"photons_per_unit={estimate.photons_per_unit:.9g} "
        f"(p_raw={estimate.photons_per_raw_unit:.9g}, "
        f"SUPPORT gain={estimate.support_gain:.7g}, "
        f"effective pedestal={estimate.effective_support_pedestal:.7g}, "
        f"variance R^2={estimate.linear_r_squared:.6f}, "
        f"variance delta AICc={estimate.delta_aicc:+.3f}, "
        f"scale R^2={estimate.scale_r_squared:.6f}, "
        f"scale delta AICc={estimate.scale_delta_aicc:+.3f})",
        flush=True,
    )
    if estimate.quiet_photons_per_unit is None:
        print("  quiet-pair sensitivity estimate: unavailable", flush=True)
    else:
        relative_change = (
            estimate.quiet_photons_per_unit / estimate.photons_per_unit - 1.0
        )
        print(
            "  quiet-pair sensitivity estimate: "
            f"{estimate.quiet_photons_per_unit:.9g} "
            f"({relative_change:+.1%} versus primary, "
            f"R^2={estimate.quiet_linear_r_squared:.6f})",
            flush=True,
        )
    if estimate.warnings:
        for warning in estimate.warnings:
            print(f"  photon-estimator warning: {warning}", flush=True)
        if not configuration.allow_provisional_photon_estimate:
            raise RuntimeError(
                "Photon calibration failed quality checks: "
                + "; ".join(estimate.warnings)
            )
    return estimate


def _close_resource_log(resource) -> None:
    """Close LogManager handlers and break its reference cycle, if present."""
    manager = getattr(resource, "log", None)
    if manager is None:
        return
    logger = getattr(manager, "logger", None)
    if logger is not None:
        for handler in tuple(logger.handlers):
            try:
                handler.flush()
            finally:
                handler.close()
                logger.removeHandler(handler)
    if hasattr(manager, "obj"):
        manager.obj = None


def _release_job_resource(resource, torch) -> None:
    """Attempt every cleanup step and make cleanup failures explicit."""
    primary_error = sys.exc_info()[1]
    cleanup_errors = []
    try:
        resource.release_resources()
    except BaseException as exc:
        cleanup_errors.append(("release_resources", exc))
    try:
        _close_resource_log(resource)
    except BaseException as exc:
        cleanup_errors.append(("log cleanup", exc))
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except BaseException as exc:
        cleanup_errors.append(("CUDA cache cleanup", exc))

    if cleanup_errors:
        details = "; ".join(
            f"{stage}: {type(error).__name__}: {error}"
            for stage, error in cleanup_errors
        )
        if primary_error is not None:
            details = (
                f"job also failed with {type(primary_error).__name__}: "
                f"{primary_error}; {details}"
            )
        raise ResourceCleanupError(details) from cleanup_errors[0][1]


def _run_support(
    *,
    input_path: Path,
    exp_name: str,
    out_dir: Path,
    output_path: Path,
    configuration: SweepConfiguration,
    torch,
) -> Path:
    """Train and infer one SUPPORT model, releasing each GPU owner in turn."""
    _ensure_repository_importable(configuration.repository)
    if __package__:
        from ..utils.test_collection import TestingClass
        from ..utils.train_collection import TrainingClass
    else:
        from utils.test_collection import TestingClass
        from utils.train_collection import TrainingClass

    with configuration.parameters_path.open(encoding="utf-8") as file:
        params = json.load(file)
    train_params = dict(params[configuration.dimensionality]["train"])
    test_params = dict(params[configuration.dimensionality]["test"])
    train_params.update(
        {
            "dimensionality": configuration.dimensionality,
            "rm_dir": configuration.rm_dir,
            "out_dir": str(out_dir),
            "n_cpu": configuration.n_cpu,
            # Each worker sees only its assigned physical GPU as logical GPU 0.
            "cuda_device": "0",
            "in_path": str(input_path),
            "train_gen": False,
            "exp_name": exp_name,
            "checkpoints": configuration.checkpoints,
            "resume": configuration.resume,
            "rm_intermediaries": configuration.rm_intermediaries,
        }
    )

    trainer = None
    try:
        trainer = TrainingClass(train_params)
        trainer.vars_dict()
        trainer.log.log_params()
        # TrainingClass currently seeds torch after model construction. Seed
        # here too so GPU count/task assignment cannot change initial weights.
        torch.manual_seed(int(train_params.get("rnd_seed", 0)))
        trainer.run()
        model_path = str(trainer.model_path).replace("/vf/users", "/data")
        test_out_dir = str(trainer.out_dir).replace("/vf/users", "/data")
        test_exp_name = trainer.exp_name
    finally:
        if trainer is not None:
            try:
                _release_job_resource(trainer, torch)
            finally:
                del trainer

    test_dict = {
        "in_path": str(input_path),
        "model_path": model_path,
        "exp_name": test_exp_name,
        "out_dir": test_out_dir,
        "output_path": str(output_path).replace("/vf/users", "/data"),
        "dimensionality": train_params["dimensionality"],
        "cuda_device": "0",
        "patch_size": train_params["patch_size"],
        "patch_interval": test_params["patch_interval"],
        "batch_size": test_params["batch_size"],
        "pin_memory": test_params.get("pin_memory", True),
        "depth": train_params["depth"],
        "bc_channels": train_params["bc_channels"],
        "obo_channels": train_params["obo_channels"],
        "ll_channels": train_params["ll_channels"],
        "unet_channels": train_params["unet_channels"],
        "bs_size": train_params["bs_size"],
        "bp": train_params.get("bp", False),
        "n_cpu": train_params["n_cpu"],
        "prefetch_factor": test_params["prefetch_factor"],
        "pad": test_params["pad"],
    }
    tester = None
    try:
        tester = TestingClass(test_dict)
        tester.run()
        output_path = Path(tester.out_path)
    finally:
        if tester is not None:
            try:
                _release_job_resource(tester, torch)
            finally:
                del tester

    return output_path


def _run_original_task(
    task: OriginalTask,
    configuration: SweepConfiguration,
    torch,
    gpu_slot: int,
) -> Path:
    """Denoise one raw source to create the persistent simulation reference."""
    print(
        f"[GPU {gpu_slot}] Original job {task.index + 1}: {task.source.name}",
        flush=True,
    )
    output_path = _run_support(
        input_path=task.source,
        exp_name=task.exp_name,
        out_dir=task.out_dir,
        output_path=task.output_path,
        configuration=configuration,
        torch=torch,
    )
    print(f"Created denoised original image at {output_path}", flush=True)
    return output_path


def _run_task(
    task: SweepTask,
    configuration: SweepConfiguration,
    torch,
    gpu_slot: int,
) -> Path:
    """Simulate one power from its denoised reference, then run SUPPORT."""
    _ensure_repository_importable(configuration.repository)
    if __package__:
        from ..utils.salt import simulate_lower_power
    else:
        from utils.salt import simulate_lower_power

    task_photons_per_unit = (
        task.photons_per_unit
        if task.photons_per_unit is not None
        else configuration.photons_per_unit
    )
    try:
        task_photons_per_unit = float(task_photons_per_unit)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Power jobs require a finite positive photons_per_unit"
        ) from exc
    if not np.isfinite(task_photons_per_unit) or task_photons_per_unit <= 0:
        raise ValueError(
            "Power jobs require a finite positive photons_per_unit. Run them "
            "through run_staged_sweep for automatic calibration or provide an "
            "explicit override."
        )
    task_zero_photon_level = (
        task.zero_photon_level
        if task.zero_photon_level is not None
        else configuration.photon_zero_level
    )
    try:
        task_zero_photon_level = float(task_zero_photon_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("zero_photon_level must be a finite scalar") from exc
    if not np.isfinite(task_zero_photon_level):
        raise ValueError("zero_photon_level must be a finite scalar")

    print(
        f"[GPU {gpu_slot}] Power job {task.index + 1}: "
        f"{task.source.name} at {task.power}% power "
        f"(photons_per_unit={task_photons_per_unit:.9g}, "
        f"zero_photon_level={task_zero_photon_level:.9g})",
        flush=True,
    )
    with tifffile.TiffFile(task.reference_path) as tif:
        stack = tif.asarray()
    stack = simulate_lower_power(
        stack,
        task.power / 100,
        # ``power`` is measured excitation power at the specimen. Unsaturated
        # two-photon fluorescence, including optical background, scales as P^2.
        gamma=2.0,
        photons_per_unit=task_photons_per_unit,
        zero_photon_level=task_zero_photon_level,
    )
    if stack.ndim == 4:
        axes = "TZYX"
    elif configuration.dimensionality == "3D":
        axes = "ZYX"
    else:
        axes = "TYX"
    print(f"Saving simulated TIFF to {task.simulated_path}...", flush=True)
    _write_simulated_tiff(task.simulated_path, stack, axes)
    del stack

    output_path = _run_support(
        input_path=task.simulated_path,
        exp_name=task.exp_name,
        out_dir=task.out_dir,
        output_path=(
            task.out_dir / task.exp_name / f"{task.exp_name}_SPRT.tif"
        ),
        configuration=configuration,
        torch=torch,
    )

    if configuration.rm_tmp:
        task.simulated_path.unlink()
    return output_path


def _gpu_worker(
    gpu_slot: int,
    device_token: str,
    tasks: Sequence[Union[OriginalTask, SweepTask]],
    configuration: SweepConfiguration,
    result_queue,
    stage: str = "power",
) -> None:
    """Pin one spawned process to one GPU, then run its jobs serially."""
    if stage not in {"original", "power"}:
        raise ValueError(f"Unknown SNR worker stage: {stage}")
    runner = _run_original_task if stage == "original" else _run_task

    # This must happen before torch or either model class is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = device_token
    repository = str(configuration.repository)
    if repository not in sys.path:
        sys.path.insert(0, repository)

    failures = 0
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"GPU worker {gpu_slot} could not isolate CUDA device {device_token!r}"
            )
        torch.cuda.set_device(0)
        print(
            f"GPU worker {gpu_slot}: {torch.cuda.get_device_name(0)}; "
            f"{len(tasks)} independent {stage} job(s)",
            flush=True,
        )
    except BaseException as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        for task in tasks:
            result_queue.put(
                TaskResult(
                    task_index=task.index,
                    gpu_slot=gpu_slot,
                    source=task.source,
                    power=task.power,
                    success=False,
                    error=error,
                    stage=stage,
                )
            )
        raise

    for position, task in enumerate(tasks):
        try:
            output_path = runner(task, configuration, torch, gpu_slot)
        except Exception as exc:
            failures += 1
            error = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            print(
                f"GPU worker {gpu_slot} failed {stage} job for {task.source}"
                + (f" at {task.power}%" if task.power is not None else "")
                + f":\n{error}",
                file=sys.stderr,
                flush=True,
            )
            result_queue.put(
                TaskResult(
                    task_index=task.index,
                    gpu_slot=gpu_slot,
                    source=task.source,
                    power=task.power,
                    success=False,
                    error=error,
                    stage=stage,
                )
            )
            if isinstance(exc, ResourceCleanupError):
                remaining_error = (
                    f"GPU worker {gpu_slot} stopped after a cleanup failure in "
                    f"job {task.index + 1}; this job was not started"
                )
                for remaining_task in tasks[position + 1 :]:
                    failures += 1
                    result_queue.put(
                        TaskResult(
                            task_index=remaining_task.index,
                            gpu_slot=gpu_slot,
                            source=remaining_task.source,
                            power=remaining_task.power,
                            success=False,
                            error=remaining_error,
                            stage=stage,
                        )
                    )
                break
        else:
            result_queue.put(
                TaskResult(
                    task_index=task.index,
                    gpu_slot=gpu_slot,
                    source=task.source,
                    power=task.power,
                    success=True,
                    output_path=output_path,
                    stage=stage,
                )
            )

    if failures:
        raise SystemExit(1)


def _run_parallel_stage(
    tasks: Sequence[Union[OriginalTask, SweepTask]],
    selected_gpus: Sequence[tuple[int, str]],
    configuration: SweepConfiguration,
    *,
    stage: str,
) -> tuple[TaskResult, ...]:
    """Spawn isolated GPU workers for one dependency-free pipeline stage."""
    if stage not in {"original", "power"}:
        raise ValueError(f"Unknown SNR stage: {stage}")
    tasks = tuple(tasks)
    if not tasks:
        return ()
    selected_gpus = tuple(selected_gpus)
    assignments = partition_tasks(tasks, [slot for slot, _ in selected_gpus])
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = []

    try:
        for slot, token in selected_gpus:
            assigned = assignments[slot]
            if not assigned:
                continue
            process = context.Process(
                target=_gpu_worker,
                args=(slot, token, assigned, configuration, result_queue, stage),
                name=(
                    f"snr-gpu-{slot}"
                    if stage == "power"
                    else f"snr-original-gpu-{slot}"
                ),
            )
            process.start()
            processes.append((slot, assigned, process))

        results = {}
        while len(results) < len(tasks):
            try:
                result = result_queue.get(timeout=0.5)
            except Empty:
                if processes and all(
                    not process.is_alive() for _, _, process in processes
                ):
                    break
                continue
            results[result.task_index] = result

        for _, _, process in processes:
            process.join()

        # A multiprocessing.Queue feeder can publish its final items just after
        # is_alive() turns false. Joining and draining avoids false "missing"
        # failures for results that were already sent by a worker.
        while len(results) < len(tasks):
            try:
                result = result_queue.get(timeout=0)
            except Empty:
                break
            results[result.task_index] = result
    except BaseException:
        for _, _, process in processes:
            if process.is_alive():
                process.terminate()
        for _, _, process in processes:
            process.join()
        raise
    finally:
        result_queue.close()
        result_queue.join_thread()

    task_to_slot = {
        task.index: slot
        for slot, assigned, _ in processes
        for task in assigned
    }
    for task in tasks:
        if task.index not in results:
            results[task.index] = TaskResult(
                task_index=task.index,
                gpu_slot=task_to_slot[task.index],
                source=task.source,
                power=task.power,
                success=False,
                error="GPU worker exited without reporting a result",
                stage=stage,
            )

    abnormal_slots = {
        slot: process.exitcode
        for slot, _, process in processes
        if process.exitcode not in (0, None)
    }
    for result in tuple(results.values()):
        if result.success and result.gpu_slot in abnormal_slots:
            # A handled task failure makes that worker exit 1, but its successful
            # sibling jobs remain valid. Only flag an abnormal exit when no task
            # failure from that worker explains it.
            if not any(
                not other.success and other.gpu_slot == result.gpu_slot
                for other in results.values()
            ):
                results[result.task_index] = TaskResult(
                    task_index=result.task_index,
                    gpu_slot=result.gpu_slot,
                    source=result.source,
                    power=result.power,
                    success=False,
                    output_path=result.output_path,
                    error=(
                        f"GPU worker {result.gpu_slot} exited with status "
                        f"{abnormal_slots[result.gpu_slot]}"
                    ),
                    stage=stage,
                )

    return tuple(results[index] for index in sorted(results))


def run_parallel_originals(
    tasks: Sequence[OriginalTask],
    selected_gpus: Sequence[tuple[int, str]],
    configuration: SweepConfiguration,
) -> tuple[TaskResult, ...]:
    """Create all denoised originals and wait for the stage to finish."""
    return _run_parallel_stage(
        tasks,
        selected_gpus,
        configuration,
        stage="original",
    )


def run_parallel_sweep(
    tasks: Sequence[SweepTask],
    selected_gpus: Sequence[tuple[int, str]],
    configuration: SweepConfiguration,
) -> tuple[TaskResult, ...]:
    """Run independent power jobs after their originals have been created."""
    return _run_parallel_stage(
        tasks,
        selected_gpus,
        configuration,
        stage="power",
    )


def run_staged_sweep(
    original_tasks: Sequence[OriginalTask],
    power_tasks: Sequence[SweepTask],
    selected_gpus: Sequence[tuple[int, str]],
    configuration: SweepConfiguration,
) -> tuple[TaskResult, ...]:
    """Create originals, calibrate each source once, then run its power jobs."""
    original_tasks = tuple(original_tasks)
    power_tasks = tuple(power_tasks)
    configured_photons_per_unit = None
    if configuration.photons_per_unit is not None:
        try:
            configured_photons_per_unit = float(
                configuration.photons_per_unit
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "photons_per_unit override must be finite and positive"
            ) from exc
        if (
            not np.isfinite(configured_photons_per_unit)
            or configured_photons_per_unit <= 0
        ):
            raise ValueError(
                "photons_per_unit override must be finite and positive"
            )
    try:
        configured_zero_level = float(configuration.photon_zero_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("photon_zero_level must be a finite scalar") from exc
    if not np.isfinite(configured_zero_level):
        raise ValueError("photon_zero_level must be a finite scalar")
    if (
        configuration.use_estimated_photon_pedestal
        and configured_photons_per_unit is not None
    ):
        raise ValueError(
            "use_estimated_photon_pedestal requires automatic photon "
            "calibration (photons_per_unit=None)"
        )
    if (
        configuration.use_estimated_photon_pedestal
        and configured_zero_level != 0.0
    ):
        raise ValueError(
            "Set either photon_zero_level or use_estimated_photon_pedestal, "
            "not both"
        )
    if (
        configured_photons_per_unit is None
        and configuration.dimensionality == "3D"
    ):
        raise ValueError(
            "Static 3D data require an explicit photons_per_unit because the "
            "photon-transfer estimator needs repeated time frames."
        )
    original_results = run_parallel_originals(
        original_tasks,
        selected_gpus,
        configuration,
    )
    original_by_source = {result.source: result for result in original_results}
    original_failures = sum(not result.success for result in original_results)
    print(
        "Original-image stage complete: "
        f"{len(original_results) - original_failures} succeeded, "
        f"{original_failures} failed.",
        flush=True,
    )

    photon_values = {}
    zero_levels = {}
    photon_errors = {}
    for original_result in original_results:
        if not original_result.success or original_result.output_path is None:
            continue
        if configured_photons_per_unit is not None:
            photon_values[original_result.source] = configured_photons_per_unit
            zero_levels[original_result.source] = configured_zero_level
            continue
        try:
            estimate = _estimate_source_photons_per_unit(
                original_result.source,
                Path(original_result.output_path),
                configuration,
            )
            estimated_photons_per_unit = float(estimate.photons_per_unit)
            if (
                not np.isfinite(estimated_photons_per_unit)
                or estimated_photons_per_unit <= 0
            ):
                raise ValueError(
                    "The estimated photons_per_unit is not finite and positive"
                )
            if configuration.use_estimated_photon_pedestal:
                estimated_pedestal = float(estimate.effective_support_pedestal)
                if not np.isfinite(estimated_pedestal):
                    raise ValueError(
                        "The fitted effective SUPPORT pedestal is not finite"
                    )
                selected_zero_level = estimated_pedestal
                print(
                    "WARNING: using the extrapolated fitted effective pedestal "
                    f"{estimated_pedestal:.9g} for "
                    f"{original_result.source.name}; this is not a dark-frame "
                    "measurement.",
                    flush=True,
                )
            else:
                selected_zero_level = configured_zero_level
                print(
                    f"Using configured photon_zero_level "
                    f"{configured_zero_level:.9g} for "
                    f"{original_result.source.name}; fitted candidate was "
                    f"{estimate.effective_support_pedestal:.9g}.",
                    flush=True,
                )
            # Commit both values together so a failed pedestal selection cannot
            # leave a partially calibrated source eligible for power jobs.
            photon_values[original_result.source] = estimated_photons_per_unit
            zero_levels[original_result.source] = selected_zero_level
        except Exception as exc:
            photon_errors[original_result.source] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            print(
                f"Photon calibration failed for {original_result.source}:\n"
                f"{photon_errors[original_result.source]}",
                file=sys.stderr,
                flush=True,
            )

    runnable = []
    results = {}
    for task in power_tasks:
        original_result = original_by_source.get(task.source)
        if (
            original_result is not None
            and original_result.success
            and original_result.output_path is not None
            and task.source in photon_values
            and task.source in zero_levels
        ):
            # Use the output reported by this invocation, not a possibly stale
            # TIFF left behind by an earlier failed run.
            runnable.append(
                replace(
                    task,
                    reference_path=Path(original_result.output_path),
                    photons_per_unit=photon_values[task.source],
                    zero_photon_level=zero_levels[task.source],
                )
            )
            continue

        if original_result is None:
            gpu_slot = -1
            reason = "original-image stage did not report a result"
        elif task.source in photon_errors:
            gpu_slot = original_result.gpu_slot
            final_line = photon_errors[task.source].strip().splitlines()[-1]
            reason = f"photon calibration failed: {final_line}"
        else:
            gpu_slot = original_result.gpu_slot
            final_line = (
                original_result.error.strip().splitlines()[-1]
                if original_result.error
                else "unknown original-image failure"
            )
            reason = f"original-image stage failed: {final_line}"
        results[task.index] = TaskResult(
            task_index=task.index,
            gpu_slot=gpu_slot,
            source=task.source,
            power=task.power,
            success=False,
            error=f"Power job was not started because its {reason}",
            stage="power",
        )

    if runnable:
        for result in run_parallel_sweep(
            runnable,
            selected_gpus,
            configuration,
        ):
            results[result.task_index] = result

    return tuple(results[index] for index in sorted(results))


def _discover_visible_devices() -> tuple[str, ...]:
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None:
        return visible_device_tokens(inherited)

    # The parent may inspect CUDA because workers use the spawn start method and
    # therefore begin in fresh interpreters before being pinned.
    import torch

    return visible_device_tokens(None, torch.cuda.device_count())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repository = (
        args.repository if args.repository is not None else path_support
    ).expanduser().resolve()
    _ensure_repository_importable(repository)
    input_root = (
        args.input_directory if args.input_directory is not None else dir_noisy
    ).expanduser().resolve()
    output_root = (
        args.output_directory
        if args.output_directory is not None
        else repository / "notebooks" / "outputs"
    ).expanduser().resolve()
    configured_dimensionality = (
        args.dimensionality
        if args.dimensionality is not None
        else dimensionality
    )
    configured_power_percent = (
        args.power_percent
        if args.power_percent is not None
        else power_percent
    )
    configured_n_cpu = args.n_cpu if args.n_cpu is not None else n_cpu
    if not repository.is_dir():
        raise NotADirectoryError(f"SUPPORT repository does not exist: {repository}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"SNR input directory does not exist: {input_root}")
    try:
        output_root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "SNR output directory must not be inside the recursively scanned "
            f"input directory: {output_root}"
        )

    parameters_path = (
        args.parameters
        if args.parameters is not None
        else repository / "notebooks" / "parameters.json"
    ).expanduser().resolve()
    if not parameters_path.is_file():
        raise FileNotFoundError(f"Parameter file does not exist: {parameters_path}")

    # Every discovered TIFF is an independent SNR source.  Do not impose a
    # per-directory file-count or channel-count requirement here.
    paths_noisy = sorted(
        path.resolve()
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".tif", ".tiff"}
    )
    if not paths_noisy:
        raise FileNotFoundError(f"No TIFF files were found in {input_root}.")

    original_tasks = build_original_tasks(paths_noisy, output_root)
    tasks = build_sweep_tasks(
        paths_noisy,
        configured_power_percent,
        output_root,
        reference_paths={
            task.source: task.output_path for task in original_tasks
        },
    )
    requested_gpus = args.gpus if args.gpus is not None else gpu_ids
    selected_gpus = select_gpu_slots(requested_gpus, _discover_visible_devices())
    configured_photons_per_unit = (
        args.photons_per_unit
        if args.photons_per_unit is not None
        else photons_per_unit
    )
    if configured_photons_per_unit is not None:
        configured_photons_per_unit = float(configured_photons_per_unit)
        if (
            not np.isfinite(configured_photons_per_unit)
            or configured_photons_per_unit <= 0
        ):
            raise ValueError(
                "photons_per_unit must be None or a finite positive number"
            )
    configured_zero_level = (
        args.photon_zero_level
        if args.photon_zero_level is not None
        else photon_zero_level
    )
    configured_zero_level = float(configured_zero_level)
    if not np.isfinite(configured_zero_level):
        raise ValueError("photon_zero_level must be finite")
    configuration = SweepConfiguration(
        repository=repository,
        input_root=input_root,
        parameters_path=parameters_path,
        output_root=output_root,
        dimensionality=configured_dimensionality,
        n_cpu=configured_n_cpu,
        rm_tmp=rm_tmp,
        rm_dir=rm_dir,
        checkpoints=checkpoints,
        resume=resume,
        rm_intermediaries=rm_intermediaries,
        photons_per_unit=configured_photons_per_unit,
        photon_estimator_frame_start=(
            args.photon_frame_start
            if args.photon_frame_start is not None
            else photon_estimator_frame_start
        ),
        photon_estimator_frame_end_margin=(
            args.photon_frame_end_margin
            if args.photon_frame_end_margin is not None
            else photon_estimator_frame_end_margin
        ),
        photon_estimator_max_pairs=(
            args.photon_max_pairs
            if args.photon_max_pairs is not None
            else photon_estimator_max_pairs
        ),
        photon_estimator_sample_count=(
            args.photon_samples
            if args.photon_samples is not None
            else photon_estimator_sample_count
        ),
        photon_estimator_random_seed=photon_estimator_random_seed,
        allow_provisional_photon_estimate=(
            args.allow_provisional_photon_estimate
            or allow_provisional_photon_estimate
        ),
        photon_zero_level=configured_zero_level,
        use_estimated_photon_pedestal=(
            args.use_estimated_photon_pedestal
            or use_estimated_photon_pedestal
        ),
    )

    print(
        f"Found {len(paths_noisy)} input TIFF(s), {len(original_tasks)} "
        f"original-image job(s), and {len(tasks)} power job(s)."
    )
    print(
        "GPU workers: "
        + ", ".join(
            f"logical {slot} ({token})" for slot, token in selected_gpus
        )
    )
    print(
        f"DataLoader workers: {configured_n_cpu} per GPU, up to "
        f"{configured_n_cpu * len(selected_gpus)} total."
    )
    if configuration.photons_per_unit is None:
        print(
            "Photon scale: automatic per source "
            f"(frame_start={configuration.photon_estimator_frame_start}, "
            "frame_end_margin="
            f"{configuration.photon_estimator_frame_end_margin}, "
            f"max_pairs={configuration.photon_estimator_max_pairs}, "
            f"sample_count={configuration.photon_estimator_sample_count:,})."
        )
    else:
        print(
            "Photon scale: fixed override "
            f"photons_per_unit={configuration.photons_per_unit:.9g}."
        )
    if configuration.use_estimated_photon_pedestal:
        print("Photon zero: fitted effective pedestal (explicit opt-in).")
    else:
        print(
            "Photon zero: fixed detector pedestal "
            f"{configuration.photon_zero_level:.9g} SUPPORT units."
        )
    for path in paths_noisy:
        print(path.parent.stem)

    results = run_staged_sweep(
        original_tasks,
        tasks,
        selected_gpus,
        configuration,
    )
    failures = [result for result in results if not result.success]
    successes = len(results) - len(failures)
    print(f"SNR sweep complete: {successes} succeeded, {len(failures)} failed.")
    for result in failures:
        final_line = (
            result.error.strip().splitlines()[-1]
            if result.error
            else "unknown error"
        )
        print(
            f"FAILED GPU {result.gpu_slot}: {result.source} at "
            f"{result.power}% ({final_line})",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
