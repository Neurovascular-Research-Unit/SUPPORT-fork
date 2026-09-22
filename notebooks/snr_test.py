"""Train and test independent SNR sweep jobs across multiple GPUs.

Each ``(input TIFF, power percentage)`` pair remains a normal single-GPU
SUPPORT run. Multi-GPU execution happens at the job level: one spawned worker
is pinned to each selected GPU and runs its assigned jobs sequentially. This
preserves the batch size and other hyperparameters tuned for one GPU.
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
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Optional, Sequence

import numpy as np
import tifffile


############################################################
#### CHANGE THIS PATH TO THE PATH OF YOUR SUPPORT REPO #####
############################################################
path_support = Path("/Users/garfinkeljb/Downloads/s/SUPPORT")
#############################################################


# Basic configuration. Model parameters are read from parameters.json.
dir_noisy = Path("/Volumes/Amreen_Mughal_lab/20260921_snrTest/01_data")
dimensionality = "3+1D"
power_percent = [100, 90, 80, 70, 60, 50, 40]

# Node-dependent settings. ``None`` uses every GPU visible to this process.
# To keep the sweep on one GPU, use ``gpu_ids = [0]``. IDs are logical positions
# within CUDA_VISIBLE_DEVICES (as assigned by Slurm, when applicable).
gpu_ids = None
n_cpu = 8  # Per GPU; total DataLoader workers scale with selected GPU count.

rm_tmp = True  # removes each simulated raw + noise file after denoising
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


@dataclass(frozen=True)
class SweepTask:
    index: int
    source: Path
    power: int
    exp_name: str
    out_dir: Path
    simulated_path: Path


@dataclass(frozen=True)
class TaskResult:
    task_index: int
    gpu_slot: int
    source: Path
    power: int
    success: bool
    output_path: Optional[Path] = None
    error: str = ""


class ResourceCleanupError(RuntimeError):
    """Raised when a worker cannot safely continue after releasing a job."""


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("GPU IDs must be zero or greater")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent SUPPORT SNR sweep jobs concurrently, with one "
            "single-GPU worker per selected GPU."
        )
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


def build_sweep_tasks(
    paths: Sequence[Path], powers: Sequence[int], output_root: Path
) -> tuple[SweepTask, ...]:
    """Build deterministic jobs and reject names that would write concurrently."""
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

    tasks = []
    destinations = {}
    for source in paths:
        source = Path(source).resolve()
        for power in normalized_powers:
            exp_name = f"{source.stem}_{power:03d}"
            out_dir = Path(output_root).resolve() / source.stem
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
                    power=power,
                    exp_name=exp_name,
                    out_dir=out_dir,
                    simulated_path=simulated_path,
                )
            )
    return tuple(tasks)


def partition_tasks(
    tasks: Sequence[SweepTask], gpu_slots: Sequence[int]
) -> dict[int, tuple[SweepTask, ...]]:
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


def _run_task(
    task: SweepTask,
    configuration: SweepConfiguration,
    torch,
    gpu_slot: int,
) -> Path:
    """Run one complete single-GPU simulation, training, and inference job."""
    from utils.salt import simulate_lower_power
    from utils.test_collection import TestingClass
    from utils.train_collection import TrainingClass

    print(
        f"[GPU {gpu_slot}] Job {task.index + 1}: "
        f"{task.source.name} at {task.power}% power",
        flush=True,
    )
    with tifffile.TiffFile(task.source) as tif:
        stack = tif.asarray()
    stack = simulate_lower_power(
        stack,
        task.power / 100,
        gamma=2.0,
        photons_per_unit=1 / 100,
    )

    with configuration.parameters_path.open(encoding="utf-8") as file:
        params = json.load(file)
    train_params = dict(params[configuration.dimensionality]["train"])
    test_params = dict(params[configuration.dimensionality]["test"])
    train_params.update(
        {
            "dimensionality": configuration.dimensionality,
            "rm_dir": configuration.rm_dir,
            "out_dir": str(task.out_dir),
            "n_cpu": configuration.n_cpu,
            # Each worker sees only its assigned physical GPU as logical GPU 0.
            "cuda_device": "0",
            "in_path": str(task.simulated_path),
            "train_gen": False,
            "exp_name": task.exp_name,
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
        print(f"Saving low-power TIFF to {task.simulated_path}...", flush=True)
        if stack.ndim == 4:
            axes = "TZYX"
        elif configuration.dimensionality == "3D":
            axes = "ZYX"
        else:
            axes = "TYX"
        _write_simulated_tiff(task.simulated_path, stack, axes)
        del stack

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
        "in_path": str(task.simulated_path),
        "model_path": model_path,
        "exp_name": test_exp_name,
        "out_dir": test_out_dir,
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

    if configuration.rm_tmp:
        task.simulated_path.unlink()
    return output_path


def _gpu_worker(
    gpu_slot: int,
    device_token: str,
    tasks: Sequence[SweepTask],
    configuration: SweepConfiguration,
    result_queue,
) -> None:
    """Pin one spawned process to one GPU, then run its jobs serially."""
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
            f"{len(tasks)} independent job(s)",
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
                )
            )
        raise

    for position, task in enumerate(tasks):
        try:
            output_path = _run_task(task, configuration, torch, gpu_slot)
        except Exception as exc:
            failures += 1
            error = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            print(
                f"GPU worker {gpu_slot} failed {task.source} at {task.power}%:\n"
                f"{error}",
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
                )
            )

    if failures:
        raise SystemExit(1)


def run_parallel_sweep(
    tasks: Sequence[SweepTask],
    selected_gpus: Sequence[tuple[int, str]],
    configuration: SweepConfiguration,
) -> tuple[TaskResult, ...]:
    """Spawn isolated GPU workers and return one result for every planned job."""
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
                args=(slot, token, assigned, configuration, result_queue),
                name=f"snr-gpu-{slot}",
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
                )

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
    repository = path_support.expanduser().resolve()
    input_root = dir_noisy.expanduser().resolve()
    if not repository.is_dir():
        raise NotADirectoryError(f"SUPPORT repository does not exist: {repository}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"SNR input directory does not exist: {input_root}")

    parameters_path = repository / "notebooks" / "parameters.json"
    if not parameters_path.is_file():
        raise FileNotFoundError(f"Parameter file does not exist: {parameters_path}")

    paths_noisy = sorted(
        path.resolve()
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".tif", ".tiff"}
    )
    if not paths_noisy:
        raise FileNotFoundError(f"No TIFF files were found in {input_root}.")

    output_root = repository / "notebooks" / "outputs"
    tasks = build_sweep_tasks(paths_noisy, power_percent, output_root)
    requested_gpus = args.gpus if args.gpus is not None else gpu_ids
    selected_gpus = select_gpu_slots(requested_gpus, _discover_visible_devices())
    configuration = SweepConfiguration(
        repository=repository,
        input_root=input_root,
        parameters_path=parameters_path,
        output_root=output_root,
        dimensionality=dimensionality,
        n_cpu=n_cpu,
        rm_tmp=rm_tmp,
        rm_dir=rm_dir,
        checkpoints=checkpoints,
        resume=resume,
        rm_intermediaries=rm_intermediaries,
    )

    print(f"Found {len(paths_noisy)} input TIFF(s) and {len(tasks)} sweep job(s).")
    print(
        "GPU workers: "
        + ", ".join(
            f"logical {slot} ({token})" for slot, token in selected_gpus
        )
    )
    print(
        f"DataLoader workers: {n_cpu} per GPU, up to "
        f"{n_cpu * len(selected_gpus)} total."
    )
    for path in paths_noisy:
        print(path.parent.stem)

    results = run_parallel_sweep(tasks, selected_gpus, configuration)
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
