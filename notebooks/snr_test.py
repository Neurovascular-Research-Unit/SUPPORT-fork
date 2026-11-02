"""Cooperatively train and test restartable SUPPORT SNR sweep jobs.

Each raw TIFF first gets one persistent denoised ``original`` reference. Power
models derived from a ready reference are then independent. Persistent advisory
locks let multiple local GPU workers or overlapping Slurm submissions share the
same output tree without training the same logical model twice.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import json
import multiprocessing
import os
import socket
import stat
import sys
import tempfile
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Sequence, Union

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
SNR_JOB_LOCK_NAME = ".snr_job.lock"
_COMPLETED_OUTPUT_VALIDATION_CACHE: dict[
    tuple[
        Path,
        tuple[int, int, int, int, int],
        Path,
        tuple[int, int, int, int, int],
    ],
    bool,
] = {}


# Basic configuration. Model parameters are read from parameters.json.
dir_noisy = Path("/Volumes/Amreen_Mughal_lab/20260921_snrTest/01_data")
dimensionality = "3+1D"
power_percent = [100, 90, 80, 70, 60, 50, 40, 30, 20]

# ``None`` estimates one value per raw/SUPPORT pair when a worker first claims
# unfinished power work. Set a positive float to use an explicit override.
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
    train_pin_memory: Optional[bool] = None
    test_pin_memory: Optional[bool] = None
    slack_webhook_file: Optional[Path] = None
    exclude_training_edge_frames: bool = False
    test_pad: Optional[bool] = None


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
    deferred: bool = False
    output_signature: Optional[tuple[int, int, int, int, int]] = None


class ResourceCleanupError(RuntimeError):
    """Raised when a worker cannot safely continue after releasing a job."""


class TaskBusyError(RuntimeError):
    """Raised when another SNR submission owns one logical model."""


class PhotonCalibrationError(RuntimeError):
    """Raised when a source cannot supply parameters for its power jobs."""


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


def _new_slack_notifier(repository: Path, webhook_file: Optional[Path]):
    """Build one isolated notifier without exposing the webhook to parameters."""
    _ensure_repository_importable(repository)
    if __package__:
        from ..utils.slack_notifications import SlackNotifier
    else:
        from utils.slack_notifications import SlackNotifier

    return SlackNotifier.from_environment(
        enabled=webhook_file is not None,
        webhook_file=webhook_file,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cooperatively denoise each source once, then claim independent "
            "SUPPORT SNR power jobs with one worker per selected GPU."
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
        "--no-train-pin-memory",
        dest="train_pin_memory",
        action="store_false",
        help="disable pinned host memory for training DataLoader batches",
    )
    parser.add_argument(
        "--no-test-pin-memory",
        dest="test_pin_memory",
        action="store_false",
        help="disable pinned host memory for inference DataLoader batches",
    )
    parser.set_defaults(train_pin_memory=None, test_pin_memory=None)
    parser.add_argument(
        "--slack-webhook-file",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "private mode-0600 Slack Incoming Webhook file; enables start, "
            "epoch, completion, and failure notifications for each model"
        ),
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


def _power_output_path(task: SweepTask) -> Path:
    """Return the stable final output path for one simulated-power model."""
    return task.out_dir / task.exp_name / f"{task.exp_name}_SPRT.tif"


def _regular_file_signature(path: Path) -> tuple[int, int, int, int, int]:
    """Return a change-sensitive identity for one nonempty regular file."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"file must not be a symbolic link: {path}")
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
        raise ValueError(f"file is missing, empty, or not regular: {path}")
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
    )


def _tiff_shape(path: Path) -> tuple[int, ...]:
    """Read one TIFF's logical shape without materializing the full stack."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"TIFF must not be a symbolic link: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"TIFF is missing or empty: {path}")
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"TIFF contains no image series: {path}")
        series = tif.series[0]
        if not series.pages:
            raise ValueError(f"TIFF contains no image pages: {path}")
        return tuple(int(value) for value in series.shape)


def _verify_completed_support_output(
    output_path: Path,
    input_path: Path,
) -> Path:
    """Validate an atomic SUPPORT TIFF before treating it as complete."""
    output_path = Path(output_path)
    if output_path.is_symlink():
        raise ValueError(f"SUPPORT output must not be a symbolic link: {output_path}")
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise ValueError(f"SUPPORT output is missing or empty: {output_path}")
    output_stat = output_path.stat()
    if not stat.S_ISREG(output_stat.st_mode):
        raise ValueError(f"SUPPORT output is not a regular file: {output_path}")
    expected_shape = _tiff_shape(input_path)

    with tifffile.TiffFile(output_path) as tif:
        if not tif.series:
            raise ValueError(f"SUPPORT output contains no TIFF series: {output_path}")
        series = tif.series[0]
        actual_shape = tuple(int(value) for value in series.shape)
        actual_dtype = np.dtype(series.dtype)
        if actual_shape != expected_shape:
            raise ValueError(
                f"SUPPORT output shape {actual_shape} does not match input "
                f"shape {expected_shape}: {output_path}"
            )
        if actual_dtype != np.dtype(np.float32):
            raise ValueError(
                f"SUPPORT output dtype {actual_dtype} is not float32: "
                f"{output_path}"
            )
        if not series.pages:
            raise ValueError(f"SUPPORT output contains no TIFF pages: {output_path}")
        file_size = output_stat.st_size
        for page_index, page in enumerate(series.pages):
            offsets = tuple(int(value) for value in page.dataoffsets)
            byte_counts = tuple(int(value) for value in page.databytecounts)
            if not offsets or len(offsets) != len(byte_counts):
                raise ValueError(
                    f"SUPPORT output page {page_index} has invalid TIFF data "
                    f"offset metadata: {output_path}"
                )
            if any(
                offset < 0 or count <= 0 or offset + count > file_size
                for offset, count in zip(offsets, byte_counts)
            ):
                raise ValueError(
                    f"SUPPORT output page {page_index} references data outside "
                    f"the TIFF file: {output_path}"
                )
            values = np.asarray(page.asarray())
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(
                    f"SUPPORT output page {page_index} is empty or contains "
                    f"non-finite values: {output_path}"
                )
    return output_path


def _completed_support_output_is_reusable(
    output_path: Path,
    input_path: Path,
) -> bool:
    """Return whether an existing final TIFF is a safe restart boundary."""
    output_path = Path(output_path)
    input_path = Path(input_path)
    cache_key = None
    try:
        output_signature = _regular_file_signature(output_path)
        input_signature = _regular_file_signature(input_path)
        cache_key = (
            output_path.absolute(),
            output_signature,
            input_path.absolute(),
            input_signature,
        )
        if cache_key in _COMPLETED_OUTPUT_VALIDATION_CACHE:
            return _COMPLETED_OUTPUT_VALIDATION_CACHE[cache_key]
        _verify_completed_support_output(output_path, input_path)
    except Exception as exc:
        # Cache deterministic structural/content failures, but allow a later
        # scan to retry transient filesystem I/O errors without requiring the
        # file metadata to change first.
        if cache_key is not None and not isinstance(exc, OSError):
            _COMPLETED_OUTPUT_VALIDATION_CACHE[cache_key] = False
        if output_path.exists():
            print(
                f"WARNING: existing SUPPORT output is not reusable "
                f"({type(exc).__name__}: {exc}); it will be regenerated.",
                file=sys.stderr,
                flush=True,
            )
        return False
    _COMPLETED_OUTPUT_VALIDATION_CACHE[cache_key] = True
    return True


@contextlib.contextmanager
def _exclusive_snr_job_lock(experiment_directory: Path) -> Iterator[Path]:
    """Nonblockingly claim one model across overlapping Slurm submissions."""
    experiment_directory = Path(experiment_directory)
    if experiment_directory.is_symlink():
        raise ValueError(
            f"SNR experiment directory must not be a symlink: "
            f"{experiment_directory}"
        )
    experiment_directory.mkdir(parents=True, exist_ok=True)
    if not experiment_directory.is_dir():
        raise ValueError(
            f"SNR experiment path is not a directory: {experiment_directory}"
        )
    lock_path = experiment_directory / SNR_JOB_LOCK_NAME
    if lock_path.is_symlink():
        raise ValueError(f"SNR job lock must not be a symlink: {lock_path}")

    with lock_path.open("a+", encoding="utf-8") as handle:
        lock_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(lock_stat.st_mode):
            raise ValueError(f"SNR job lock is not a regular file: {lock_path}")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "owner details unavailable"
            raise TaskBusyError(
                f"another submission owns {experiment_directory}: {owner}"
            ) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(
            f"job_id={os.environ.get('SLURM_JOB_ID', '')} "
            f"hostname={socket.gethostname()} pid={os.getpid()}\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def _validated_photon_configuration(
    configuration: SweepConfiguration,
) -> tuple[Optional[float], float]:
    """Validate and normalize sweep-wide photon settings."""
    configured_photons_per_unit = None
    if configuration.photons_per_unit is not None:
        try:
            configured_photons_per_unit = float(configuration.photons_per_unit)
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
    return configured_photons_per_unit, configured_zero_level


def _resolve_source_calibration(
    source: Path,
    reference_path: Path,
    configuration: SweepConfiguration,
) -> tuple[float, float]:
    """Resolve one source's simulation scale after its power lock is claimed."""
    configured_photons_per_unit, configured_zero_level = (
        _validated_photon_configuration(configuration)
    )
    if configured_photons_per_unit is not None:
        return configured_photons_per_unit, configured_zero_level

    try:
        estimate = _estimate_source_photons_per_unit(
            source,
            reference_path,
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
            selected_zero_level = float(estimate.effective_support_pedestal)
            if not np.isfinite(selected_zero_level):
                raise ValueError(
                    "The fitted effective SUPPORT pedestal is not finite"
                )
            print(
                "WARNING: using the extrapolated fitted effective pedestal "
                f"{selected_zero_level:.9g} for {Path(source).name}; this is "
                "not a dark-frame measurement.",
                flush=True,
            )
        else:
            selected_zero_level = configured_zero_level
            print(
                f"Using configured photon_zero_level "
                f"{configured_zero_level:.9g} for {Path(source).name}; fitted "
                f"candidate was {estimate.effective_support_pedestal:.9g}.",
                flush=True,
            )
        return estimated_photons_per_unit, selected_zero_level
    except PhotonCalibrationError:
        raise
    except Exception as exc:
        raise PhotonCalibrationError(
            f"Photon calibration failed for {Path(source)}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


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
            "exclude_training_edge_frames": configuration.exclude_training_edge_frames,
        }
    )
    if configuration.train_pin_memory is not None:
        train_params["pin_memory"] = configuration.train_pin_memory

    # A new notifier per model keeps timing and loss history isolated when one
    # GPU worker runs several original/power jobs in sequence.
    slack_notifier = _new_slack_notifier(
        configuration.repository,
        configuration.slack_webhook_file,
    )
    failure_stage = "training initialization"
    trainer = None
    tester = None
    training_normalization = {}
    try:
        training_succeeded = False
        try:
            trainer = TrainingClass(train_params)
            trainer.vars_dict()
            trainer.log.log_params()
            if slack_notifier.enabled:
                trainer.epoch_start_callback = (
                    slack_notifier.notify_epoch_started
                )
                trainer.epoch_complete_callback = (
                    slack_notifier.notify_epoch_completed
                )
                slack_notifier.notify_run_started(
                    trainer,
                    training_count=1,
                    validation_count=0,
                )
            # TrainingClass currently seeds torch after model construction.
            # Seed here too so GPU assignment cannot change initial weights.
            torch.manual_seed(int(train_params.get("rnd_seed", 0)))
            failure_stage = "training"
            trainer.run()
            training_succeeded = True
            if configuration.exclude_training_edge_frames:
                mean = getattr(trainer, "normalization_mean", None)
                std = getattr(trainer, "normalization_std", None)
                if mean is not None and std is not None:
                    training_normalization = {"normalization_mean": mean, "normalization_std": std}
            model_path = str(trainer.model_path).replace("/vf/users", "/data")
            test_out_dir = str(trainer.out_dir).replace("/vf/users", "/data")
            test_exp_name = trainer.exp_name
        finally:
            if trainer is not None:
                if training_succeeded:
                    failure_stage = "training resource cleanup"
                _release_job_resource(trainer, torch)

        test_pin_memory = test_params.get("pin_memory", True)
        if configuration.test_pin_memory is not None:
            test_pin_memory = configuration.test_pin_memory
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
            "pin_memory": test_pin_memory,
            "depth": train_params["depth"],
            "bc_channels": train_params["bc_channels"],
            "obo_channels": train_params["obo_channels"],
            "ll_channels": train_params["ll_channels"],
            "unet_channels": train_params["unet_channels"],
            "bs_size": train_params["bs_size"],
            "bp": train_params.get("bp", False),
            "n_cpu": train_params["n_cpu"],
            "prefetch_factor": test_params["prefetch_factor"],
            "pad": configuration.test_pad if configuration.test_pad is not None else test_params["pad"],
            **training_normalization,
        }
        failure_stage = "inference initialization"
        inference_succeeded = False
        try:
            tester = TestingClass(test_dict)
            failure_stage = "inference"
            tester.run()
            inference_succeeded = True
            output_path = Path(tester.out_path)
        finally:
            if tester is not None:
                if inference_succeeded:
                    failure_stage = "inference resource cleanup"
                _release_job_resource(tester, torch)

        failure_stage = "completion notification"
        if slack_notifier.enabled:
            slack_notifier.notify_run_completed(trainer)
        return output_path
    except BaseException as exc:
        if slack_notifier.enabled:
            slack_notifier.notify_failure(
                exc,
                trainer=trainer,
                stage=failure_stage,
            )
        raise
    finally:
        if tester is not None:
            del tester
        if trainer is not None:
            del trainer


def _run_original_task(
    task: OriginalTask,
    configuration: SweepConfiguration,
    torch,
    gpu_slot: int,
) -> Path:
    """Denoise one raw source to create the persistent simulation reference."""
    if _completed_support_output_is_reusable(task.output_path, task.source):
        print(
            f"[GPU {gpu_slot}] Reusing completed original for "
            f"{task.source.name}: {task.output_path}",
            flush=True,
        )
        return task.output_path

    print(
        f"[GPU {gpu_slot}] Original job {task.index + 1}: {task.source.name}",
        flush=True,
    )
    with _exclusive_snr_job_lock(task.output_path.parent):
        # A different submission may have published the output between the
        # optimistic check above and this lock acquisition.
        if _completed_support_output_is_reusable(task.output_path, task.source):
            print(
                f"[GPU {gpu_slot}] Reusing newly completed original for "
                f"{task.source.name}: {task.output_path}",
                flush=True,
            )
            return task.output_path
        output_path = _run_support(
            input_path=task.source,
            exp_name=task.exp_name,
            out_dir=task.out_dir,
            output_path=task.output_path,
            configuration=configuration,
            torch=torch,
        )
        _verify_completed_support_output(output_path, task.source)
        print(f"Created denoised original image at {output_path}", flush=True)
        return output_path


def _run_task(
    task: SweepTask,
    configuration: SweepConfiguration,
    torch,
    gpu_slot: int,
    calibration_cache: Optional[dict[Path, tuple[float, float]]] = None,
) -> Path:
    """Simulate one power from its denoised reference, then run SUPPORT."""
    output_path = _power_output_path(task)
    if _completed_support_output_is_reusable(output_path, task.source):
        print(
            f"[GPU {gpu_slot}] Reusing completed {task.power}% power output "
            f"for {task.source.name}: {output_path}",
            flush=True,
        )
        return output_path

    with _exclusive_snr_job_lock(output_path.parent):
        if _completed_support_output_is_reusable(output_path, task.source):
            print(
                f"[GPU {gpu_slot}] Reusing newly completed {task.power}% "
                f"power output for {task.source.name}: {output_path}",
                flush=True,
            )
            if configuration.rm_tmp and task.simulated_path.is_file():
                task.simulated_path.unlink()
            return output_path

        _ensure_repository_importable(configuration.repository)
        if __package__:
            from ..utils.salt import simulate_lower_power
        else:
            from utils.salt import simulate_lower_power

        if task.photons_per_unit is None:
            cache = calibration_cache if calibration_cache is not None else {}
            calibration_key = Path(task.source).resolve()
            if calibration_key not in cache:
                cache[calibration_key] = _resolve_source_calibration(
                    task.source,
                    task.reference_path,
                    configuration,
                )
            task_photons_per_unit, task_zero_photon_level = cache[calibration_key]
        else:
            task_photons_per_unit = task.photons_per_unit
            task_zero_photon_level = (
                task.zero_photon_level
                if task.zero_photon_level is not None
                else configuration.photon_zero_level
            )
        try:
            task_photons_per_unit = float(task_photons_per_unit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Power jobs require a finite positive photons_per_unit"
            ) from exc
        if not np.isfinite(task_photons_per_unit) or task_photons_per_unit <= 0:
            raise ValueError(
                "Power jobs require a finite positive photons_per_unit. Run "
                "them through run_staged_sweep for automatic calibration or "
                "provide an explicit override."
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
            # Measured two-photon excitation power at the specimen scales P^2.
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

        completed_path = _run_support(
            input_path=task.simulated_path,
            exp_name=task.exp_name,
            out_dir=task.out_dir,
            output_path=output_path,
            configuration=configuration,
            torch=torch,
        )
        _verify_completed_support_output(completed_path, task.source)
        if configuration.rm_tmp:
            task.simulated_path.unlink()
        return completed_path


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
        except TaskBusyError as exc:
            print(
                f"GPU worker {gpu_slot} deferred {stage} job for "
                f"{task.source}"
                + (f" at {task.power}%" if task.power is not None else "")
                + f": {exc}",
                flush=True,
            )
            result_queue.put(
                TaskResult(
                    task_index=task.index,
                    gpu_slot=gpu_slot,
                    source=task.source,
                    power=task.power,
                    success=False,
                    error=str(exc),
                    stage=stage,
                    deferred=True,
                )
            )
            continue
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


def _cooperative_gpu_worker(
    gpu_slot: int,
    device_token: str,
    original_tasks: Sequence[OriginalTask],
    power_tasks: Sequence[SweepTask],
    configuration: SweepConfiguration,
    result_queue,
) -> None:
    """Claim source-major work shared by every local or remote worker.

    Locks, rather than a static partition, decide ownership. After every
    completed model the scan restarts at the first source. This makes a worker
    skip a locked, not-yet-denoised source, while letting workers split power
    models as soon as that source's original is complete.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = device_token
    repository = str(configuration.repository)
    if repository not in sys.path:
        sys.path.insert(0, repository)

    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"GPU worker {gpu_slot} could not isolate CUDA device "
                f"{device_token!r}"
            )
        torch.cuda.set_device(0)
        print(
            f"GPU worker {gpu_slot}: {torch.cuda.get_device_name(0)}; "
            f"cooperatively scanning {len(original_tasks)} source(s) and "
            f"{len(power_tasks)} power job(s)",
            flush=True,
        )
    except BaseException as exc:
        error = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        for task in power_tasks:
            result_queue.put(
                TaskResult(
                    task_index=task.index,
                    gpu_slot=gpu_slot,
                    source=task.source,
                    power=task.power,
                    success=False,
                    error=error,
                    stage="power",
                )
            )
        raise

    powers_by_source: dict[Path, list[SweepTask]] = {
        task.source: [] for task in original_tasks
    }
    for task in power_tasks:
        powers_by_source[task.source].append(task)

    ready_originals: set[Path] = set()
    completed_powers: set[int] = set()
    attempted_original_failures: set[int] = set()
    attempted_power_failures: set[int] = set()
    reported_original_successes: set[int] = set()
    reported_power_successes: set[int] = set()
    reported_original_deferrals: set[int] = set()
    reported_power_deferrals: set[int] = set()
    calibration_cache: dict[Path, tuple[float, float]] = {}
    failures = 0
    stop_worker = False

    def report_success(
        task: Union[OriginalTask, SweepTask],
        output_path: Path,
        stage: str,
    ) -> None:
        reported = (
            reported_original_successes
            if stage == "original"
            else reported_power_successes
        )
        if task.index in reported:
            return
        reported.add(task.index)
        try:
            output_signature = _regular_file_signature(output_path)
        except (OSError, ValueError):
            # The coordinator will perform the authoritative final check. This
            # also covers an external deletion immediately after validation.
            output_signature = None
        result_queue.put(
            TaskResult(
                task_index=task.index,
                gpu_slot=gpu_slot,
                source=task.source,
                power=task.power,
                success=True,
                output_path=Path(output_path),
                stage=stage,
                output_signature=output_signature,
            )
        )

    def report_failure(
        task: Union[OriginalTask, SweepTask],
        error: str,
        stage: str,
    ) -> None:
        nonlocal failures
        failures += 1
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

    def report_deferred(
        task: Union[OriginalTask, SweepTask],
        error: str,
        stage: str,
    ) -> None:
        reported = (
            reported_original_deferrals
            if stage == "original"
            else reported_power_deferrals
        )
        if task.index in reported:
            return
        reported.add(task.index)
        result_queue.put(
            TaskResult(
                task_index=task.index,
                gpu_slot=gpu_slot,
                source=task.source,
                power=task.power,
                success=False,
                error=error,
                stage=stage,
                deferred=True,
            )
        )

    while not stop_worker:
        completed_model = False
        for original_task in original_tasks:
            source = original_task.source
            source_power_tasks = powers_by_source[source]

            # Existing power outputs are authoritative and need neither the
            # original reference nor a new photon calibration.
            for task in source_power_tasks:
                if task.index in completed_powers:
                    continue
                output_path = _power_output_path(task)
                if _completed_support_output_is_reusable(output_path, task.source):
                    completed_powers.add(task.index)
                    report_success(task, output_path, "power")

            if all(task.index in completed_powers for task in source_power_tasks):
                continue

            if source not in ready_originals:
                if _completed_support_output_is_reusable(
                    original_task.output_path,
                    original_task.source,
                ):
                    ready_originals.add(source)
                    report_success(
                        original_task,
                        original_task.output_path,
                        "original",
                    )
                elif original_task.index in attempted_original_failures:
                    continue
                else:
                    try:
                        output_path = _run_original_task(
                            original_task,
                            configuration,
                            torch,
                            gpu_slot,
                        )
                    except TaskBusyError as exc:
                        # Another worker owns this prerequisite. Move to the
                        # next raw source instead of waiting on its lock.
                        report_deferred(original_task, str(exc), "original")
                        continue
                    except Exception as exc:
                        attempted_original_failures.add(original_task.index)
                        error = "".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        )
                        print(
                            f"GPU worker {gpu_slot} failed original job for "
                            f"{source}:\n{error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        report_failure(original_task, error, "original")
                        if isinstance(exc, ResourceCleanupError):
                            stop_worker = True
                            break
                        continue
                    else:
                        ready_originals.add(source)
                        report_success(original_task, output_path, "original")
                        completed_model = True
                        break

            if source not in ready_originals:
                continue

            for task in source_power_tasks:
                if (
                    task.index in completed_powers
                    or task.index in attempted_power_failures
                ):
                    continue
                try:
                    output_path = _run_task(
                        task,
                        configuration,
                        torch,
                        gpu_slot,
                        calibration_cache,
                    )
                except TaskBusyError as exc:
                    # A ready source may have several independent power jobs;
                    # try its next power before advancing to the next source.
                    report_deferred(task, str(exc), "power")
                    continue
                except PhotonCalibrationError as exc:
                    error = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                    print(error, file=sys.stderr, flush=True)
                    for blocked_task in source_power_tasks:
                        if (
                            blocked_task.index in completed_powers
                            or blocked_task.index in attempted_power_failures
                        ):
                            continue
                        attempted_power_failures.add(blocked_task.index)
                        blocked_error = (
                            error
                            if blocked_task.index == task.index
                            else "Power job was not started because source photon "
                            f"calibration failed: {exc}"
                        )
                        report_failure(blocked_task, blocked_error, "power")
                    break
                except Exception as exc:
                    attempted_power_failures.add(task.index)
                    error = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                    print(
                        f"GPU worker {gpu_slot} failed power job for "
                        f"{task.source} at {task.power}%:\n{error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    report_failure(task, error, "power")
                    if isinstance(exc, ResourceCleanupError):
                        stop_worker = True
                        break
                    continue
                else:
                    completed_powers.add(task.index)
                    report_success(task, output_path, "power")
                    completed_model = True
                    break

            if stop_worker or completed_model:
                break

        # A full source-major pass with no claim means every unfinished model
        # is either locked by another submission or already failed here. Do
        # not spin or turn ordinary lock contention into a job failure.
        if stop_worker or not completed_model:
            break

    if failures:
        print(
            f"GPU worker {gpu_slot} reported {failures} model failure(s); "
            "the parent process will determine the sweep exit status.",
            file=sys.stderr,
            flush=True,
        )


def _run_cooperative_sweep(
    original_tasks: Sequence[OriginalTask],
    power_tasks: Sequence[SweepTask],
    selected_gpus: Sequence[tuple[int, str]],
    configuration: SweepConfiguration,
) -> tuple[TaskResult, ...]:
    """Run one shared lock-based work inventory on every selected GPU."""
    original_tasks = tuple(original_tasks)
    power_tasks = tuple(power_tasks)
    if not power_tasks:
        return ()
    selected_gpus = tuple(selected_gpus)
    if not selected_gpus:
        raise ValueError("at least one GPU must be selected")

    sources = [task.source for task in original_tasks]
    if len(set(sources)) != len(sources):
        raise ValueError("original task sources must be unique")
    original_indices = [task.index for task in original_tasks]
    if len(set(original_indices)) != len(original_indices):
        raise ValueError("original task indexes must be unique")
    original_by_source = {task.source: task for task in original_tasks}
    missing_sources = sorted(
        {task.source for task in power_tasks} - set(original_by_source),
        key=str,
    )
    if missing_sources:
        raise ValueError(
            f"Power task has no original task for source {missing_sources[0]}"
        )
    mismatched_reference = next(
        (
            task
            for task in power_tasks
            if Path(task.reference_path)
            != Path(original_by_source[task.source].output_path)
        ),
        None,
    )
    if mismatched_reference is not None:
        raise ValueError(
            "Power task reference does not match its original task output: "
            f"{mismatched_reference.reference_path}"
        )
    power_indices = [task.index for task in power_tasks]
    if len(set(power_indices)) != len(power_indices):
        raise ValueError("power task indexes must be unique")

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = []
    reports: list[TaskResult] = []
    try:
        for slot, token in selected_gpus:
            process = context.Process(
                target=_cooperative_gpu_worker,
                args=(
                    slot,
                    token,
                    original_tasks,
                    power_tasks,
                    configuration,
                    result_queue,
                ),
                name=f"snr-cooperative-gpu-{slot}",
            )
            process.start()
            processes.append((slot, process))

        while processes and any(process.is_alive() for _, process in processes):
            try:
                reports.append(result_queue.get(timeout=0.5))
            except Empty:
                continue
        for _, process in processes:
            process.join()
        while True:
            try:
                reports.append(result_queue.get(timeout=0.1))
            except Empty:
                break
    except BaseException:
        for _, process in processes:
            if process.is_alive():
                process.terminate()
        for _, process in processes:
            process.join()
        raise
    finally:
        result_queue.close()
        result_queue.join_thread()

    reports_by_key: dict[tuple[str, int], list[TaskResult]] = {}
    for report in reports:
        reports_by_key.setdefault((report.stage, report.task_index), []).append(
            report
        )
    abnormal_slots = {
        slot: process.exitcode
        for slot, process in processes
        if process.exitcode not in (0, None)
    }

    results = []
    for task in power_tasks:
        output_path = _power_output_path(task)
        original_task = original_by_source[task.source]
        task_reports = reports_by_key.get(("power", task.index), [])
        successful_report = next(
            (report for report in task_reports if report.success),
            None,
        )
        reported_output_is_unchanged = False
        if (
            successful_report is not None
            and successful_report.output_signature is not None
        ):
            try:
                reported_output_is_unchanged = (
                    _regular_file_signature(output_path)
                    == successful_report.output_signature
                )
            except (OSError, ValueError):
                pass
        if reported_output_is_unchanged or _completed_support_output_is_reusable(
            output_path,
            task.source,
        ):
            results.append(
                TaskResult(
                    task_index=task.index,
                    gpu_slot=(
                        successful_report.gpu_slot
                        if successful_report is not None
                        else -1
                    ),
                    source=task.source,
                    power=task.power,
                    success=True,
                    output_path=output_path,
                    stage="power",
                    output_signature=_regular_file_signature(output_path),
                )
            )
            continue

        failure_report = next(
            (
                report
                for report in task_reports
                if not report.success and not report.deferred
            ),
            None,
        )
        if failure_report is None:
            if successful_report is not None:
                failure_report = TaskResult(
                    task_index=task.index,
                    gpu_slot=successful_report.gpu_slot,
                    source=task.source,
                    power=task.power,
                    success=False,
                    output_path=successful_report.output_path,
                    error=(
                        "GPU worker reported success, but the final SUPPORT "
                        "output is missing or invalid"
                    ),
                    stage="power",
                )
        if failure_report is None:
            original_failures = reports_by_key.get(
                ("original", original_task.index),
                [],
            )
            original_failure = next(
                (
                    report
                    for report in original_failures
                    if not report.success and not report.deferred
                ),
                None,
            )
            if original_failure is not None:
                final_line = (
                    original_failure.error.strip().splitlines()[-1]
                    if original_failure.error
                    else "unknown original-image failure"
                )
                failure_report = TaskResult(
                    task_index=task.index,
                    gpu_slot=original_failure.gpu_slot,
                    source=task.source,
                    power=task.power,
                    success=False,
                    error=(
                        "Power job was not started because its original-image "
                        f"job failed: {final_line}"
                    ),
                    stage="power",
                )

        if failure_report is not None:
            results.append(failure_report)
            continue

        if abnormal_slots:
            slot = min(abnormal_slots)
            results.append(
                TaskResult(
                    task_index=task.index,
                    gpu_slot=slot,
                    source=task.source,
                    power=task.power,
                    success=False,
                    error=(
                        f"GPU worker {slot} exited with status "
                        f"{abnormal_slots[slot]} without reporting this task"
                    ),
                    stage="power",
                )
            )
            continue

        power_deferral = next(
            (
                report
                for report in task_reports
                if not report.success and report.deferred
            ),
            None,
        )
        original_deferral = next(
            (
                report
                for report in reports_by_key.get(
                    ("original", original_task.index),
                    [],
                )
                if not report.success and report.deferred
            ),
            None,
        )
        deferral_report = power_deferral or original_deferral
        if deferral_report is not None:
            results.append(
                TaskResult(
                    task_index=task.index,
                    gpu_slot=deferral_report.gpu_slot,
                    source=task.source,
                    power=task.power,
                    success=False,
                    error=deferral_report.error,
                    stage="power",
                    deferred=True,
                )
            )
            continue

        results.append(
            TaskResult(
                task_index=task.index,
                gpu_slot=-1,
                source=task.source,
                power=task.power,
                success=False,
                error=(
                    "Cooperative GPU workers exited without completing, "
                    "failing, or deferring this task"
                ),
                stage="power",
            )
        )

    return tuple(sorted(results, key=lambda result: result.task_index))


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
                not other.success
                and not other.deferred
                and other.gpu_slot == result.gpu_slot
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
    """Validate settings, then cooperatively claim original and power models."""
    _validated_photon_configuration(configuration)
    return _run_cooperative_sweep(
        original_tasks,
        power_tasks,
        selected_gpus,
        configuration,
    )


def _discover_visible_devices() -> tuple[str, ...]:
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None:
        return visible_device_tokens(inherited)

    # The parent may inspect CUDA because workers use the spawn start method and
    # therefore begin in fresh interpreters before being pinned.
    import torch

    return visible_device_tokens(None, torch.cuda.device_count())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    configured_slack_webhook_file = None
    if args.slack_webhook_file is not None:
        candidate = args.slack_webhook_file.expanduser()
        try:
            # Validate permissions and contents before any GPU worker starts.
            _new_slack_notifier(repository, candidate)
        except ValueError as exc:
            parser.error(str(exc))
        configured_slack_webhook_file = candidate.resolve(strict=True)
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
        train_pin_memory=args.train_pin_memory,
        test_pin_memory=args.test_pin_memory,
        slack_webhook_file=configured_slack_webhook_file,
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
    print(
        "Slack notifications: "
        + (
            "enabled for each independent SUPPORT model"
            if configuration.slack_webhook_file is not None
            else "disabled"
        )
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
    failures = [
        result for result in results if not result.success and not result.deferred
    ]
    deferred = [result for result in results if result.deferred]
    successes = sum(result.success for result in results)
    print(
        f"SNR sweep complete: {successes} succeeded, "
        f"{len(deferred)} deferred to another submission, "
        f"{len(failures)} failed."
    )
    for result in deferred:
        print(
            f"DEFERRED: {result.source} at {result.power}% "
            "is owned by another worker or awaits its prerequisite."
        )
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
