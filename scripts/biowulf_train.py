#!/usr/bin/env python3
"""Run train_gen-style SUPPORT training and validation from a batch job."""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import socket
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile


REPO_DIRECTORY = Path(__file__).resolve().parent.parent
if str(REPO_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPO_DIRECTORY))

DEFAULT_PARAMETERS = REPO_DIRECTORY / "notebooks" / "gen_parameters.json"
DEFAULT_OUTPUT_DIRECTORY = REPO_DIRECTORY / "notebooks" / "outputs"
RUN_MANIFEST_NAME = ".biowulf_train_config.json"
NAMESPACE_LOCK_NAME = ".biowulf_training_namespace.lock"
NAMESPACE_LOCK_ENVIRONMENT_VARIABLE = "SUPPORT_TRAIN_LOCK_REGISTRY"
TIFF_SUFFIXES = {".tif", ".tiff"}
VALID_NDIMS = {
    "2+1D": {3},
    "3D": {3},
    "3+1D": {3, 4},
}


class TrainingLaunchError(RuntimeError):
    """Raised for a safe-to-summarize launch or locking error."""


@dataclass(frozen=True)
class TiffDescription:
    path: Path
    shape: Tuple[int, ...]
    dtype: str
    size: int
    mtime_ns: int
    device: int
    inode: int


@dataclass(frozen=True)
class TrainingConfiguration:
    training_directory: Path
    validation_path: Path
    training_tiffs: Tuple[TiffDescription, ...]
    validation_tiffs: Tuple[TiffDescription, ...]
    dimensionality: str
    parameters_path: Path
    output_directory: Path
    experiment_name: str
    experiment_directory: Path
    validation_output_directory: Path
    validation_output_files: Tuple[Path, ...]
    training_parameters: Dict[str, Any]
    validation_parameters: Dict[str, Any]
    use_cpu: bool
    parameters_sha256: str
    adopt_existing_run: bool


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return number


def slurm_cpu_count(default: int = 16) -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value is None:
        return default
    try:
        number = int(value)
    except ValueError:
        return default
    return number if number >= 0 else default


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def paths_overlap(first: Path, second: Path) -> bool:
    return path_is_within(first, second) or path_is_within(second, first)


def discover_tiffs(path: Path, require_directory: bool = False) -> List[Path]:
    if require_directory and not path.is_dir():
        raise ValueError(f"training input must be a directory: {path}")
    if path.is_file():
        if path.suffix.casefold() not in TIFF_SUFFIXES:
            raise ValueError(f"input file is not a TIFF: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {path}")
    # Match TrainingClass and ValidationTracker's ordinary Path ordering so the
    # preflight listing is the same order those classes will consume.
    paths = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.casefold() in TIFF_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no TIFF files were found recursively in {path}")
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_tiffs(paths: Sequence[Path], dimensionality: str) -> Tuple[TiffDescription, ...]:
    descriptions: List[TiffDescription] = []
    allowed_ndims = VALID_NDIMS[dimensionality]
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"TIFF inputs must not be symlinks: {path}")
        before = path.stat()
        if before.st_size <= 0:
            raise ValueError(f"TIFF is empty: {path}")
        try:
            with tifffile.TiffFile(path) as tif:
                if not tif.series:
                    raise ValueError("contains no image series")
                series = tif.series[0]
                shape = tuple(int(item) for item in series.shape)
                dtype = np.dtype(series.dtype).name
                if not series.pages:
                    raise ValueError("contains no image pages")
        except Exception as exc:
            raise ValueError(f"could not inspect TIFF {path}: {exc}") from exc
        after = path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity:
            raise ValueError(f"TIFF changed while it was being inspected: {path}")
        if len(shape) not in allowed_ndims:
            expected = " or ".join(str(value) for value in sorted(allowed_ndims))
            raise ValueError(
                f"{dimensionality} expects a {expected}-dimensional TIFF, "
                f"but {path} has shape {shape}"
            )
        descriptions.append(
            TiffDescription(
                path=path,
                shape=shape,
                dtype=dtype,
                size=after.st_size,
                mtime_ns=after.st_mtime_ns,
                device=after.st_dev,
                inode=after.st_ino,
            )
        )
    return tuple(descriptions)


def validation_relative_output_paths(
    validation_path: Path, descriptions: Sequence[TiffDescription]
) -> Dict[Path, Path]:
    """Match ValidationTracker's stable relative names, including collisions."""
    relative_paths = {
        description.path: (
            description.path.relative_to(validation_path)
            if validation_path.is_dir()
            else Path(description.path.name)
        )
        for description in descriptions
    }
    candidate_groups: Dict[str, List[Path]] = {}
    for source, relative_path in relative_paths.items():
        stem = relative_path.stem
        if stem.casefold().endswith("_sprt"):
            stem = stem[:-5]
        candidate = relative_path.with_name(f"{stem}_SPRT.tif")
        candidate_groups.setdefault(candidate.as_posix().casefold(), []).append(source)

    output_paths: Dict[Path, Path] = {}
    reserved = set()
    for sources in candidate_groups.values():
        if len(sources) != 1:
            continue
        source = sources[0]
        relative_path = relative_paths[source]
        stem = relative_path.stem
        if stem.casefold().endswith("_sprt"):
            stem = stem[:-5]
        output_path = relative_path.with_name(f"{stem}_SPRT.tif")
        output_paths[source] = output_path
        reserved.add(output_path.as_posix().casefold())

    for sources in candidate_groups.values():
        if len(sources) == 1:
            continue
        for source in sources:
            relative_path = relative_paths[source]
            suffix_label = source.suffix.casefold().lstrip(".") or "tiff"
            base_name = f"{relative_path.stem}_{suffix_label}_SPRT"
            output_path = relative_path.with_name(f"{base_name}.tif")
            counter = 2
            while output_path.as_posix().casefold() in reserved:
                output_path = relative_path.with_name(f"{base_name}_{counter}.tif")
                counter += 1
            output_paths[source] = output_path
            reserved.add(output_path.as_posix().casefold())
    return output_paths


def validate_directory_destination(path: Path, label: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"{label} exists but is not a directory: {path}")
        writable_ancestor = path
    else:
        writable_ancestor = path.parent
        while not writable_ancestor.exists() and writable_ancestor != writable_ancestor.parent:
            writable_ancestor = writable_ancestor.parent
        if not writable_ancestor.is_dir():
            raise ValueError(
                f"an ancestor of {label} is not a directory: {writable_ancestor}"
            )
    if not os.access(writable_ancestor, os.W_OK | os.X_OK):
        raise ValueError(f"{label} is not writable through {writable_ancestor}: {path}")


def validate_output_file_layout(
    output_root: Path, output_files: Sequence[Path]
) -> None:
    relative_parts = [
        tuple(part.casefold() for part in path.relative_to(output_root).parts)
        for path in output_files
    ]
    owners: Dict[Tuple[str, ...], int] = {}
    for index, parts in enumerate(relative_parts):
        other_index = owners.get(parts)
        if other_index is not None:
            raise ValueError(
                "validation outputs collide after case normalization: "
                f"{output_files[other_index]} and {output_files[index]}"
            )
        owners[parts] = index
    for index, parts in enumerate(relative_parts):
        for prefix_length in range(1, len(parts)):
            other_index = owners.get(parts[:prefix_length])
            if other_index is not None:
                raise ValueError(
                    "validation outputs have a file/directory path collision: "
                    f"{output_files[other_index]} conflicts with {output_files[index]}"
                )


def stack_shape(description: TiffDescription, dimensionality: str) -> Tuple[int, int, int]:
    if dimensionality == "3+1D" and len(description.shape) == 4:
        time, _depth, height, width = description.shape
        return time, height, width
    return description.shape  # type: ignore[return-value]


def validate_patch_compatibility(
    training_tiffs: Sequence[TiffDescription],
    validation_tiffs: Sequence[TiffDescription],
    dimensionality: str,
    patch_size: Sequence[int],
    validation_interval: Sequence[int],
    validation_pad: bool,
) -> None:
    patch = tuple(int(value) for value in patch_size)
    if int(validation_interval[0]) != 1:
        raise ValueError(
            "test.patch_interval[0] must be 1 so stitched validation writes "
            "every temporal/depth plane"
        )
    if validation_pad and patch[0] == 1:
        raise ValueError(
            "validation padding with a temporal patch size of 1 produces an empty crop; "
            "use --no-validation-pad"
        )
    # Stitched validation must cover every output pixel. Training may
    # intentionally sample with a sparser interval, so only enforce this on
    # the validation grid.
    if any(int(gap) > size for gap, size in zip(validation_interval, patch)):
        raise ValueError(
            "test.patch_interval must not exceed train.patch_size on any axis"
        )
    if any(
        (size - int(gap)) % 2 != 0
        for gap, size in zip(validation_interval, patch)
    ):
        raise ValueError(
            "train.patch_size minus test.patch_interval must be even on every axis"
        )

    for description in training_tiffs:
        shape = stack_shape(description, dimensionality)
        if any(size < required for size, required in zip(shape, patch)):
            raise ValueError(
                f"training TIFF {description.path} becomes stack shape {shape}, "
                f"which is smaller than patch size {patch}"
            )

    temporal_margin = patch[0] // 2 if validation_pad else 0
    for description in validation_tiffs:
        raw_shape = stack_shape(description, dimensionality)
        effective_shape = (
            raw_shape[0] + 2 * temporal_margin,
            raw_shape[1],
            raw_shape[2],
        )
        if any(size < required for size, required in zip(effective_shape, patch)):
            padding_note = " after temporal padding" if validation_pad else ""
            raise ValueError(
                f"validation TIFF {description.path} becomes stack shape "
                f"{effective_shape}{padding_note}, which is smaller than patch size {patch}"
            )


def load_parameter_sections(
    parameters_path: Path, dimensionality: str
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    try:
        parameter_bytes = parameters_path.read_bytes()
        parameters = json.loads(parameter_bytes.decode("utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"parameter file does not exist: {parameters_path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read parameter file {parameters_path}: {exc}") from exc

    try:
        section = parameters[dimensionality]
        training = section["train"]
        validation = section["test"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{parameters_path} must contain {dimensionality!r} train and test objects"
        ) from exc
    if not isinstance(training, dict) or not isinstance(validation, dict):
        raise ValueError(
            f"{parameters_path} must contain {dimensionality!r} train and test objects"
        )
    # The notebook deliberately copies these dictionaries and never writes JSON.
    return (
        dict(training),
        dict(validation),
        hashlib.sha256(parameter_bytes).hexdigest(),
    )


def _validate_positive_integer(mapping: Dict[str, Any], name: str, label: str) -> None:
    value = mapping.get(name)
    if type(value) is not int or value < 1:
        raise ValueError(f"{label}.{name} must be a positive integer")


def _validate_triplet(mapping: Dict[str, Any], name: str, label: str) -> None:
    value = mapping.get(name)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(item) is not int or item < 1 for item in value)
    ):
        raise ValueError(f"{label}.{name} must contain three positive integers")


def validate_parameter_sections(
    training: Dict[str, Any], validation: Dict[str, Any]
) -> None:
    for name in ("n_epochs", "batch_size", "prefetch_factor"):
        _validate_positive_integer(training, name, "train")
    if type(training.get("n_cpu")) is not int or training["n_cpu"] < 0:
        raise ValueError("train.n_cpu must be a nonnegative integer")
    if type(training.get("rnd_seed")) is not int or training["rnd_seed"] < 0:
        raise ValueError("train.rnd_seed must be a nonnegative integer")
    for name in ("patch_size", "patch_interval"):
        _validate_triplet(training, name, "train")
    for name in ("load_to_memory", "pin_memory", "use_amp"):
        if type(training.get(name)) is not bool:
            raise ValueError(f"train.{name} must be true or false")

    for name in ("batch_size", "prefetch_factor"):
        _validate_positive_integer(validation, name, "test")
    _validate_triplet(validation, "patch_interval", "test")
    if type(validation.get("pin_memory")) is not bool:
        raise ValueError("test.pin_memory must be true or false")
    if type(validation.get("pad")) is not bool:
        raise ValueError("test.pad must be true or false")

    loss_coef = training.get("loss_coef")
    if (
        not isinstance(loss_coef, (list, tuple))
        or len(loss_coef) != 2
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or item < 0
            for item in loss_coef
        )
        or not any(float(item) > 0 for item in loss_coef)
    ):
        raise ValueError("train.loss_coef must contain two finite nonnegative weights")
    learning_rate = training.get("lr")
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("train.lr must be finite and greater than zero")


def normalize_bool(value: Any, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one generalized SUPPORT model from recursive TIFFs, validate "
            "after every epoch, and retain resumable epoch checkpoints."
        )
    )
    parser.add_argument(
        "--training-directory",
        "--in-path",
        dest="training_directory",
        required=True,
        type=Path,
        metavar="DIRECTORY",
        help="training directory searched recursively for .tif and .tiff files",
    )
    parser.add_argument(
        "--test-directory",
        "--test-dir",
        "--validation-directory",
        dest="test_directory",
        required=True,
        type=Path,
        metavar="PATH",
        help="held-out TIFF file or directory searched recursively for TIFFs",
    )
    parser.add_argument(
        "--dimensionality",
        choices=("2+1D", "3D", "3+1D"),
        default="2+1D",
        help="SUPPORT training mode (default: %(default)s)",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=DEFAULT_PARAMETERS,
        help="read-only train/test parameter JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--output-directory",
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="output root; the experiment name is added beneath it",
    )
    parser.add_argument(
        "--validation-output-directory",
        type=Path,
        default=None,
        help="validation output root (default: EXPERIMENT_DIRECTORY/validation)",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        help="experiment name (default: training directory basename)",
    )
    parser.add_argument(
        "--n-epochs",
        type=positive_int,
        help="final total epoch count, not the number of additional epochs",
    )
    parser.add_argument(
        "--n-cpu",
        type=nonnegative_int,
        default=None,
        help="training workers (default: SLURM_CPUS_PER_TASK, otherwise 16)",
    )
    parser.add_argument(
        "--train-batch-size",
        "--batch-size",
        dest="train_batch_size",
        type=positive_int,
        help="override the training batch size",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=positive_int,
        help="override the validation batch size",
    )
    parser.add_argument(
        "--train-patch-size",
        "--patch-size",
        dest="train_patch_size",
        type=positive_int,
        nargs=3,
        metavar=("T_OR_Z", "Y", "X"),
        help="override the training/model patch size",
    )
    parser.add_argument(
        "--train-patch-interval",
        "--patch-interval",
        dest="train_patch_interval",
        type=positive_int,
        nargs=3,
        metavar=("T_OR_Z", "Y", "X"),
        help="override the training patch interval",
    )
    parser.add_argument(
        "--validation-patch-interval",
        type=positive_int,
        nargs=3,
        metavar=("T_OR_Z", "Y", "X"),
        help="override the validation patch interval",
    )
    parser.add_argument(
        "--train-prefetch-factor",
        "--prefetch-factor",
        dest="train_prefetch_factor",
        type=positive_int,
        help="override the training DataLoader prefetch factor",
    )
    parser.add_argument(
        "--validation-prefetch-factor",
        type=positive_int,
        help="override the validation prefetch factor",
    )
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        help="override the training learning rate",
    )
    parser.add_argument(
        "--loss-coef",
        type=nonnegative_float,
        nargs=2,
        metavar=("L1", "L2"),
        help="override the two training/validation loss weights",
    )
    parser.add_argument(
        "--random-seed",
        type=nonnegative_int,
        help="override the training random seed",
    )

    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--load-to-memory",
        dest="load_to_memory",
        action="store_true",
        help="load all training TIFFs into RAM",
    )
    memory_group.add_argument(
        "--no-load-to-memory",
        dest="load_to_memory",
        action="store_false",
        help="memory-map training TIFFs where possible",
    )
    parser.set_defaults(load_to_memory=None)

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="enable automatic mixed precision",
    )
    amp_group.add_argument(
        "--no-use-amp",
        dest="use_amp",
        action="store_false",
        help="disable automatic mixed precision",
    )
    parser.set_defaults(use_amp=None)

    train_pin_group = parser.add_mutually_exclusive_group()
    train_pin_group.add_argument(
        "--train-pin-memory",
        dest="train_pin_memory",
        action="store_true",
        help="enable pinned host memory for training batches",
    )
    train_pin_group.add_argument(
        "--no-train-pin-memory",
        dest="train_pin_memory",
        action="store_false",
        help="disable pinned host memory for training batches",
    )
    parser.set_defaults(train_pin_memory=None)

    validation_pin_group = parser.add_mutually_exclusive_group()
    validation_pin_group.add_argument(
        "--validation-pin-memory",
        dest="validation_pin_memory",
        action="store_true",
        help="enable pinned host memory for validation batches",
    )
    validation_pin_group.add_argument(
        "--no-validation-pin-memory",
        dest="validation_pin_memory",
        action="store_false",
        help="disable pinned host memory for validation batches",
    )
    parser.set_defaults(validation_pin_memory=None)

    validation_pad_group = parser.add_mutually_exclusive_group()
    validation_pad_group.add_argument(
        "--validation-pad",
        dest="validation_pad",
        action="store_true",
        help="pad validation stacks before inference",
    )
    validation_pad_group.add_argument(
        "--no-validation-pad",
        dest="validation_pad",
        action="store_false",
        help="disable validation padding",
    )
    parser.set_defaults(validation_pad=None)

    parser.add_argument(
        "--keep-only-latest",
        action="store_true",
        help="remove older checkpoint bundles after each new save (default: retain all)",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="automatically resume the newest complete checkpoint (default)",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="start at epoch zero; refused when the experiment directory is nonempty",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--adopt-existing-run",
        action="store_true",
        help=(
            "explicitly attach a missing run manifest to a legacy experiment that "
            "has a complete resumable checkpoint"
        ),
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="train without CUDA (intended only for small debugging runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the resolved run without creating outputs",
    )
    return parser


def build_configuration(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> TrainingConfiguration:
    try:
        raw_training_directory = args.training_directory.expanduser()
        if raw_training_directory.is_symlink():
            raise ValueError(
                f"training input root must not be a symlink: {raw_training_directory}"
            )
        training_directory = raw_training_directory.resolve(strict=True)
        if not training_directory.is_dir():
            raise ValueError(f"training input must be a directory: {training_directory}")
        raw_validation_path = args.test_directory.expanduser()
        if raw_validation_path.is_symlink():
            raise ValueError(
                f"validation input root must not be a symlink: {raw_validation_path}"
            )
        validation_path = raw_validation_path.resolve(strict=True)
        parameters_path = args.parameters.expanduser().resolve(strict=True)
        output_directory = args.output_directory.expanduser().resolve(strict=False)

        experiment_name = args.exp_name or training_directory.name
        if (
            not experiment_name
            or experiment_name in {".", ".."}
            or Path(experiment_name).name != experiment_name
        ):
            raise ValueError("--exp-name must be one nonempty path component")
        # Resolve the appended experiment component too: an existing symlink at
        # OUT_DIR/EXP_NAME must not bypass input/output overlap guards or locks.
        lexical_experiment_directory = output_directory / experiment_name
        if lexical_experiment_directory.is_symlink():
            raise ValueError(
                "the experiment directory itself must not be a symlink: "
                f"{lexical_experiment_directory}"
            )
        experiment_directory = lexical_experiment_directory.resolve(strict=False)
        raw_validation_output_directory = (
            args.validation_output_directory.expanduser()
            if args.validation_output_directory is not None
            else experiment_directory / "validation"
        )
        if raw_validation_output_directory.is_symlink():
            raise ValueError(
                "the validation output root itself must not be a symlink: "
                f"{raw_validation_output_directory}"
            )
        validation_output_directory = raw_validation_output_directory.resolve(
            strict=False
        )

        validate_directory_destination(experiment_directory, "experiment directory")
        validate_directory_destination(
            validation_output_directory, "validation output directory"
        )
        if args.adopt_existing_run and not args.resume:
            raise ValueError("--adopt-existing-run cannot be combined with --no-resume")

        if paths_overlap(training_directory, validation_path):
            raise ValueError(
                "training and validation inputs must be disjoint so validation is held out"
            )
        if paths_overlap(training_directory, experiment_directory):
            raise ValueError(
                "the experiment output directory must not overlap the recursive training tree"
            )
        if paths_overlap(training_directory, validation_output_directory):
            raise ValueError(
                "the validation output directory must not overlap the recursive training tree"
            )
        if paths_overlap(validation_path, experiment_directory):
            raise ValueError(
                "the experiment output directory must not overlap the validation input"
            )
        if paths_overlap(validation_path, validation_output_directory):
            raise ValueError(
                "the validation output directory must not overlap the recursive validation input"
            )

        training_paths = discover_tiffs(training_directory, require_directory=True)
        validation_paths = discover_tiffs(validation_path)
        training_inodes = {(path.stat().st_dev, path.stat().st_ino) for path in training_paths}
        validation_inodes = {(path.stat().st_dev, path.stat().st_ino) for path in validation_paths}
        if training_inodes.intersection(validation_inodes):
            raise ValueError(
                "training and validation inputs share at least one underlying TIFF file"
            )

        training_tiffs = inspect_tiffs(training_paths, args.dimensionality)
        validation_tiffs = inspect_tiffs(validation_paths, args.dimensionality)
        relative_validation_outputs = validation_relative_output_paths(
            validation_path, validation_tiffs
        )
        validation_output_files_list = []
        for description in validation_tiffs:
            output_file = (
                validation_output_directory
                / relative_validation_outputs[description.path]
            ).resolve(strict=False)
            if not path_is_within(output_file, validation_output_directory):
                raise ValueError(
                    "a validation output escapes its output root through a symlink: "
                    f"{output_file}"
                )
            validation_output_files_list.append(output_file)
        validation_output_files = tuple(validation_output_files_list)
        validate_output_file_layout(
            validation_output_directory, validation_output_files
        )
        for output_file in validation_output_files:
            validate_directory_destination(
                output_file.parent, "validation output parent directory"
            )
            if output_file.exists() and not output_file.is_file():
                raise ValueError(
                    f"validation output exists but is not a regular file: {output_file}"
                )
        (
            training_parameters,
            validation_parameters,
            parameters_sha256,
        ) = load_parameter_sections(parameters_path, args.dimensionality)

        # Preserve the underlying classes' fallback behavior for custom JSON
        # files while making the complete effective configuration explicit.
        training_defaults = {
            "n_epochs": 500,
            "patch_size": [61, 128, 128],
            "patch_interval": [1, 64, 64],
            "batch_size": 16,
            "load_to_memory": True,
            "pin_memory": True,
            "prefetch_factor": 2,
            "lr": 5e-4,
            "loss_coef": [0.5, 0.5],
            "use_amp": False,
            "rnd_seed": 0,
        }
        validation_defaults = {
            "patch_interval": [1, 11, 11],
            "batch_size": 16,
            "pin_memory": True,
            "prefetch_factor": 2,
            "pad": True,
        }
        for name, value in training_defaults.items():
            training_parameters.setdefault(name, value)
        for name, value in validation_defaults.items():
            validation_parameters.setdefault(name, value)

        training_overrides = {
            "n_epochs": args.n_epochs,
            "batch_size": args.train_batch_size,
            "patch_size": args.train_patch_size,
            "patch_interval": args.train_patch_interval,
            "prefetch_factor": args.train_prefetch_factor,
            "load_to_memory": args.load_to_memory,
            "pin_memory": args.train_pin_memory,
            "use_amp": args.use_amp,
            "lr": args.learning_rate,
            "loss_coef": args.loss_coef,
            "rnd_seed": args.random_seed,
        }
        training_parameters.update(
            {name: value for name, value in training_overrides.items() if value is not None}
        )
        n_cpu = args.n_cpu if args.n_cpu is not None else slurm_cpu_count()
        training_parameters.update(
            {
                "in_path": str(training_directory),
                "out_dir": str(output_directory),
                "exp_name": experiment_name,
                "dimensionality": args.dimensionality,
                "rm_dir": False,
                "n_cpu": n_cpu,
                "cuda_device": "0",
                "train_gen": True,
                "checkpoints": True,
                "save_every_epoch": True,
                "resume": bool(args.resume),
                "rm_intermediaries": bool(args.keep_only_latest),
                "use_cpu": bool(args.use_cpu),
            }
        )

        validation_overrides = {
            "batch_size": args.validation_batch_size,
            "patch_interval": args.validation_patch_interval,
            "prefetch_factor": args.validation_prefetch_factor,
            "pin_memory": args.validation_pin_memory,
            "pad": args.validation_pad,
        }
        validation_parameters.update(
            {name: value for name, value in validation_overrides.items() if value is not None}
        )
        validation_parameters["pad"] = normalize_bool(
            validation_parameters.get("pad", True), "test.pad"
        )
        validation_parameters["n_cpu"] = n_cpu

        validate_parameter_sections(training_parameters, validation_parameters)
        validate_patch_compatibility(
            training_tiffs=training_tiffs,
            validation_tiffs=validation_tiffs,
            dimensionality=args.dimensionality,
            patch_size=training_parameters["patch_size"],
            validation_interval=validation_parameters["patch_interval"],
            validation_pad=validation_parameters["pad"],
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    return TrainingConfiguration(
        training_directory=training_directory,
        validation_path=validation_path,
        training_tiffs=training_tiffs,
        validation_tiffs=validation_tiffs,
        dimensionality=args.dimensionality,
        parameters_path=parameters_path,
        output_directory=output_directory,
        experiment_name=experiment_name,
        experiment_directory=experiment_directory,
        validation_output_directory=validation_output_directory,
        validation_output_files=validation_output_files,
        training_parameters=training_parameters,
        validation_parameters=validation_parameters,
        use_cpu=args.use_cpu,
        parameters_sha256=parameters_sha256,
        adopt_existing_run=bool(args.adopt_existing_run),
    )


def _json_normalize(value: Any) -> Any:
    """Return the JSON representation used for stable manifest comparisons."""
    return json.loads(json.dumps(value, sort_keys=True))


def _tiff_manifest_entry(description: TiffDescription) -> Dict[str, Any]:
    return {
        "path": str(description.path),
        "shape": list(description.shape),
        "dtype": description.dtype,
        "size": description.size,
        "mtime_ns": description.mtime_ns,
        "device": description.device,
        "inode": description.inode,
    }


def manifest_configuration(configuration: TrainingConfiguration) -> Dict[str, Any]:
    training_parameters = {
        name: value
        for name, value in configuration.training_parameters.items()
        if name not in {"n_epochs", "resume"}
    }
    return _json_normalize(
        {
            "schema_version": 1,
            "training_directory": str(configuration.training_directory),
            "validation_path": str(configuration.validation_path),
            "parameters_path": str(configuration.parameters_path),
            "parameters_sha256": configuration.parameters_sha256,
            "output_directory": str(configuration.output_directory),
            "experiment_name": configuration.experiment_name,
            "experiment_directory": str(configuration.experiment_directory),
            "validation_output_directory": str(
                configuration.validation_output_directory
            ),
            "validation_output_files": [
                str(path) for path in configuration.validation_output_files
            ],
            "dimensionality": configuration.dimensionality,
            "training_parameters": training_parameters,
            "validation_parameters": configuration.validation_parameters,
            "training_tiffs": [
                _tiff_manifest_entry(description)
                for description in configuration.training_tiffs
            ],
            "validation_tiffs": [
                _tiff_manifest_entry(description)
                for description in configuration.validation_tiffs
            ],
        }
    )


def manifest_configuration_sha256(manifest_config: Dict[str, Any]) -> str:
    encoded = json.dumps(
        manifest_config,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_preflight_inputs(configuration: TrainingConfiguration) -> None:
    try:
        current_parameters_sha256 = sha256_file(configuration.parameters_path)
    except OSError as exc:
        raise TrainingLaunchError(
            f"could not recheck parameter file {configuration.parameters_path}: {exc}"
        ) from exc
    if current_parameters_sha256 != configuration.parameters_sha256:
        raise TrainingLaunchError(
            f"parameter file changed during launch: {configuration.parameters_path}; "
            "rerun the command"
        )

    try:
        current_training_paths = tuple(
            discover_tiffs(configuration.training_directory, require_directory=True)
        )
        current_validation_paths = tuple(discover_tiffs(configuration.validation_path))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise TrainingLaunchError(f"could not recheck recursive TIFF inputs: {exc}") from exc
    expected_training_paths = tuple(
        description.path for description in configuration.training_tiffs
    )
    expected_validation_paths = tuple(
        description.path for description in configuration.validation_tiffs
    )
    if current_training_paths != expected_training_paths:
        raise TrainingLaunchError(
            "the recursive training TIFF inventory changed during launch; rerun the command"
        )
    if current_validation_paths != expected_validation_paths:
        raise TrainingLaunchError(
            "the recursive validation TIFF inventory changed during launch; rerun the command"
        )

    for description in (
        *configuration.training_tiffs,
        *configuration.validation_tiffs,
    ):
        if description.path.is_symlink():
            raise TrainingLaunchError(
                f"TIFF became a symlink during launch: {description.path}; "
                "rerun the command"
            )
        try:
            stat_result = description.path.stat()
        except OSError as exc:
            raise TrainingLaunchError(
                f"could not recheck TIFF {description.path}: {exc}"
            ) from exc
        expected = (
            description.device,
            description.inode,
            description.size,
            description.mtime_ns,
        )
        observed = (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
        )
        if observed != expected:
            raise TrainingLaunchError(
                f"TIFF changed during launch: {description.path}; rerun the command"
            )


def validate_output_tree_safety(configuration: TrainingConfiguration) -> None:
    roots = sorted(
        {
            configuration.experiment_directory,
            configuration.validation_output_directory,
        },
        key=lambda path: (len(path.parts), path.as_posix()),
    )
    scan_roots = [
        root
        for index, root in enumerate(roots)
        if not any(path_is_within(root, parent) for parent in roots[:index])
    ]
    try:
        for root in scan_roots:
            if root.is_symlink():
                raise TrainingLaunchError(
                    f"output root must not be a symlink: {root}"
                )
            if not root.exists():
                continue
            for path in root.rglob("*"):
                entry_stat = path.lstat()
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise TrainingLaunchError(
                        "symlinks are not allowed inside training or validation "
                        f"output trees: {path}"
                    )
                if stat.S_ISREG(entry_stat.st_mode) and entry_stat.st_nlink != 1:
                    raise TrainingLaunchError(
                        "hard-linked files are not allowed inside training or "
                        f"validation output trees: {path}"
                    )
                if not (
                    stat.S_ISREG(entry_stat.st_mode)
                    or stat.S_ISDIR(entry_stat.st_mode)
                ):
                    raise TrainingLaunchError(
                        "special filesystem entries are not allowed inside training "
                        f"or validation output trees: {path}"
                    )
    except OSError as exc:
        raise TrainingLaunchError(f"could not inspect output tree symlinks: {exc}") from exc


def meaningful_output_artifacts(configuration: TrainingConfiguration) -> List[Path]:
    ignored_names = {
        RUN_MANIFEST_NAME,
        NAMESPACE_LOCK_NAME,
        ".biowulf_train.lock",
        ".biowulf_validation.lock",
    }
    roots = sorted(
        {
            configuration.experiment_directory,
            configuration.validation_output_directory,
        },
        key=lambda path: (len(path.parts), path.as_posix()),
    )
    scan_roots = [
        root
        for index, root in enumerate(roots)
        if not any(path_is_within(root, parent) for parent in roots[:index])
    ]
    artifacts: List[Path] = []
    try:
        for root in scan_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.name in ignored_names:
                    continue
                if path.is_file() or path.is_symlink():
                    artifacts.append(path)
                    if len(artifacts) == 5:
                        return artifacts
    except OSError as exc:
        raise TrainingLaunchError(f"could not inspect existing outputs: {exc}") from exc
    return artifacts


def find_complete_checkpoint(configuration: TrainingConfiguration) -> Optional[str]:
    checkpoint_directory = configuration.experiment_directory / "pth"
    if not checkpoint_directory.is_dir():
        return None
    require_scaler = bool(configuration.training_parameters["use_amp"])
    candidates: List[Tuple[Tuple[int, int, int], str]] = []

    for model_path in checkpoint_directory.iterdir():
        epoch_match = re.fullmatch(r"model_(\d+)\.pth", model_path.name)
        batch_match = re.fullmatch(r"model_(\d+)_batch_(\d+)\.pth", model_path.name)
        legacy_match = re.fullmatch(r"model(\d+)\.pth", model_path.name)
        if epoch_match is not None:
            epoch = int(epoch_match.group(1))
            label = str(epoch)
            required = [
                model_path,
                checkpoint_directory / f"optimizer_{label}.pth",
            ]
            if require_scaler:
                required.append(checkpoint_directory / f"scaler_{label}.pth")
            incomplete = checkpoint_directory / f".checkpoint_{label}.incomplete"
            if not incomplete.is_file() and all(path.is_file() for path in required):
                candidates.append(((epoch, 2, 0), model_path.name))
        elif batch_match is not None:
            epoch, batch = (int(value) for value in batch_match.groups())
            label = f"{epoch}_batch_{batch}"
            required = [
                model_path,
                checkpoint_directory / f"optimizer_{label}.pth",
            ]
            if require_scaler:
                required.append(checkpoint_directory / f"scaler_{label}.pth")
            incomplete = checkpoint_directory / f".checkpoint_{label}.incomplete"
            if not incomplete.is_file() and all(path.is_file() for path in required):
                candidates.append(((epoch, 1, batch), model_path.name))
        elif legacy_match is not None and int(legacy_match.group(1)) >= 1:
            candidates.append(
                ((int(legacy_match.group(1)) - 1, 0, 0), model_path.name)
            )
    if not candidates:
        return None
    return max(candidates)[1]


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def atomic_write_json(path: Path, document: Dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _manifest_difference_labels(
    stored: Dict[str, Any], current: Dict[str, Any]
) -> str:
    differences = sorted(
        key for key in set(stored).union(current) if stored.get(key) != current.get(key)
    )
    return ", ".join(differences) if differences else "unknown fields"


def validate_run_manifest(
    configuration: TrainingConfiguration, commit: bool
) -> str:
    """Validate resume identity; write/update the manifest only when locked."""
    validate_namespace_registry_location(configuration)
    verify_preflight_inputs(configuration)
    validate_output_tree_safety(configuration)
    manifest_path = configuration.experiment_directory / RUN_MANIFEST_NAME
    manifest_config = manifest_configuration(configuration)
    manifest_digest = manifest_configuration_sha256(manifest_config)
    artifacts = meaningful_output_artifacts(configuration)
    resume = bool(configuration.training_parameters["resume"])
    detected_complete_checkpoint: Optional[str] = None

    if manifest_path.is_symlink():
        raise TrainingLaunchError(f"run manifest must not be a symlink: {manifest_path}")
    if manifest_path.exists() and not manifest_path.is_file():
        raise TrainingLaunchError(
            f"run manifest exists but is not a regular file: {manifest_path}"
        )

    existing_document: Optional[Dict[str, Any]] = None
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                loaded_document = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrainingLaunchError(
                f"could not read run manifest {manifest_path}: {exc}"
            ) from exc
        if not isinstance(loaded_document, dict):
            raise TrainingLaunchError(f"run manifest is not a JSON object: {manifest_path}")
        existing_document = loaded_document

    if not resume:
        if artifacts:
            shown = ", ".join(str(path) for path in artifacts)
            raise TrainingLaunchError(
                "--no-resume requires new or empty experiment and validation "
                f"outputs; found: {shown}"
            )
        state = "new epoch-zero run"
        existing_document = None
    elif existing_document is not None:
        stored_config = existing_document.get("configuration")
        stored_digest = existing_document.get("configuration_sha256")
        if not isinstance(stored_config, dict) or stored_digest != (
            manifest_configuration_sha256(stored_config)
        ):
            raise TrainingLaunchError(
                f"run manifest is incomplete or internally inconsistent: {manifest_path}"
            )
        if stored_config != manifest_config or stored_digest != manifest_digest:
            differences = _manifest_difference_labels(stored_config, manifest_config)
            raise TrainingLaunchError(
                "resume configuration does not match the saved run manifest "
                f"({differences}); use the original settings/data or a new experiment name"
            )
        state = "matching saved run"
    elif artifacts:
        if not configuration.adopt_existing_run:
            shown = ", ".join(str(path) for path in artifacts)
            raise TrainingLaunchError(
                f"existing outputs have no {RUN_MANIFEST_NAME} manifest: {shown}; "
                "refusing automatic resume (inspect the legacy run and pass "
                "--adopt-existing-run only if it matches this configuration)"
            )
        try:
            detected_complete_checkpoint = find_complete_checkpoint(configuration)
        except OSError as exc:
            raise TrainingLaunchError(
                f"could not inspect existing checkpoints: {exc}"
            ) from exc
        if detected_complete_checkpoint is None:
            raise TrainingLaunchError(
                "--adopt-existing-run requires at least one complete resumable "
                "checkpoint in the experiment pth directory"
            )
        state = (
            "adopted legacy run; detected complete checkpoint "
            f"{detected_complete_checkpoint}"
        )
    else:
        state = "new run"

    if commit:
        now = _utc_timestamp()
        document: Dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": (
                existing_document.get("created_at_utc", now)
                if existing_document is not None
                else now
            ),
            "last_validated_at_utc": now,
            "requested_n_epochs": configuration.training_parameters["n_epochs"],
            "configuration_sha256": manifest_digest,
            "configuration": manifest_config,
        }
        if detected_complete_checkpoint is not None:
            document["adopted_existing_run"] = True
            document["detected_complete_checkpoint"] = detected_complete_checkpoint
        elif existing_document is not None:
            for name in (
                "adopted_existing_run",
                "detected_complete_checkpoint",
            ):
                if name in existing_document:
                    document[name] = existing_document[name]
        try:
            atomic_write_json(manifest_path, document)
        except OSError as exc:
            raise TrainingLaunchError(
                f"could not write run manifest {manifest_path}: {exc}"
            ) from exc
    return state


def print_configuration(configuration: TrainingConfiguration, dry_run: bool) -> None:
    prefix = "Dry run: " if dry_run else ""
    print(f"{prefix}training directory: {configuration.training_directory}")
    print(f"Training TIFFs (recursive): {len(configuration.training_tiffs)}")
    print(f"Validation input: {configuration.validation_path}")
    print(f"Validation TIFFs (recursive): {len(configuration.validation_tiffs)}")
    print(f"Dimensionality: {configuration.dimensionality}")
    print(f"Read-only parameters: {configuration.parameters_path}")
    print(f"Experiment directory: {configuration.experiment_directory}")
    print(f"Validation outputs: {configuration.validation_output_directory}")
    print(f"Final epoch count: {configuration.training_parameters['n_epochs']}")
    print(f"Training workers: {configuration.training_parameters['n_cpu']}")
    print(
        "Automatic resume: "
        + ("enabled" if configuration.training_parameters["resume"] else "disabled")
    )
    print(
        "Checkpoint retention: "
        + (
            "latest complete epoch only"
            if configuration.training_parameters["rm_intermediaries"]
            else "every completed epoch"
        )
    )
    if dry_run:
        print("No output directories, checkpoints, logs, or TIFFs were changed.")
        for description in configuration.training_tiffs:
            print(
                f"TRAIN    {description.path} :: "
                f"shape={description.shape}, dtype={description.dtype}"
            )
        for description, output_file in zip(
            configuration.validation_tiffs,
            configuration.validation_output_files,
        ):
            print(
                f"VALIDATE {description.path} :: shape={description.shape}, "
                f"dtype={description.dtype}, output={output_file}"
            )


def training_namespace_registry_path() -> Path:
    configured_registry = os.environ.get(NAMESPACE_LOCK_ENVIRONMENT_VARIABLE)
    registry_candidate = (
        Path(configured_registry).expanduser()
        if configured_registry
        else DEFAULT_OUTPUT_DIRECTORY / NAMESPACE_LOCK_NAME
    )
    if not registry_candidate.is_absolute():
        registry_candidate = Path.cwd() / registry_candidate
    # Resolve the parent but leave the final component unresolved so O_NOFOLLOW
    # can reject a registry symlink instead of silently splitting coordination.
    return registry_candidate.parent.resolve(strict=False) / registry_candidate.name


def validate_namespace_registry_location(
    configuration: TrainingConfiguration,
) -> Path:
    registry_path = training_namespace_registry_path()
    for output_root in (
        configuration.experiment_directory,
        configuration.validation_output_directory,
    ):
        if path_is_within(registry_path, output_root):
            raise TrainingLaunchError(
                "the training namespace registry must be outside every managed "
                f"output root: {registry_path}"
            )
    if path_is_within(registry_path, configuration.training_directory):
        raise TrainingLaunchError(
            f"the training namespace registry must be outside training input: {registry_path}"
        )
    if (
        configuration.validation_path.is_dir()
        and path_is_within(registry_path, configuration.validation_path)
    ) or registry_path == configuration.validation_path:
        raise TrainingLaunchError(
            f"the training namespace registry must be outside validation input: {registry_path}"
        )
    if registry_path == configuration.parameters_path:
        raise TrainingLaunchError(
            f"the training namespace registry must not be the parameter file: {registry_path}"
        )
    try:
        if registry_path.is_symlink():
            raise TrainingLaunchError(
                f"training namespace registry must not be a symlink: {registry_path}"
            )
        if registry_path.exists():
            registry_stat = registry_path.lstat()
            if not stat.S_ISREG(registry_stat.st_mode):
                raise TrainingLaunchError(
                    "training namespace registry is not a regular file: "
                    f"{registry_path}"
                )
            if registry_stat.st_nlink != 1:
                raise TrainingLaunchError(
                    "training namespace registry must have exactly one hard link: "
                    f"{registry_path}"
                )
    except OSError as exc:
        raise TrainingLaunchError(
            f"could not inspect training namespace registry {registry_path}: {exc}"
        ) from exc
    return registry_path


def open_regular_lock_file(path: Path, binary: bool):
    """Open a lock without following a final symlink or accepting special files."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise TrainingLaunchError(f"could not safely open lock file {path}: {exc}") from exc
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise TrainingLaunchError(f"lock path is not a regular file: {path}")
        if lock_stat.st_nlink != 1:
            raise TrainingLaunchError(
                f"lock file must have exactly one hard link: {path}"
            )
        return os.fdopen(
            descriptor,
            "r+b" if binary else "r+",
            encoding=None if binary else "utf-8",
        )
    except Exception:
        os.close(descriptor)
        raise


@contextlib.contextmanager
def exclusive_training_locks(
    experiment_directory: Path, validation_output_directory: Path
):
    namespace_registry = training_namespace_registry_path()
    if any(
        path_is_within(namespace_registry, output_root.resolve(strict=False))
        for output_root in (experiment_directory, validation_output_directory)
    ):
        raise TrainingLaunchError(
            "the training namespace registry must be outside every managed "
            f"output root: {namespace_registry}"
        )
    namespace_modes: Dict[Path, int] = {}
    for output_root in {experiment_directory, validation_output_directory}:
        canonical_root = output_root.resolve(strict=False)
        for namespace_path in (canonical_root, *canonical_root.parents):
            requested_mode = (
                fcntl.LOCK_EX if namespace_path == canonical_root else fcntl.LOCK_SH
            )
            if (
                requested_mode == fcntl.LOCK_EX
                or namespace_path not in namespace_modes
            ):
                namespace_modes[namespace_path] = requested_mode

    # Every path maps to one byte in a repository-wide registry. Exact output
    # roots take an exclusive byte-range lock, while their ancestors take shared
    # locks. Siblings can run together; equal or nested roots conflict.
    namespace_specs: Dict[int, Dict[str, Any]] = {}
    for namespace_path, mode in namespace_modes.items():
        digest = hashlib.sha256(os.fsencode(str(namespace_path))).digest()
        offset = int.from_bytes(digest[:8], "big") & ((1 << 62) - 1)
        spec = namespace_specs.setdefault(
            offset, {"mode": fcntl.LOCK_SH, "paths": []}
        )
        spec["paths"].append(namespace_path)
        if mode == fcntl.LOCK_EX:
            spec["mode"] = fcntl.LOCK_EX

    lock_paths = sorted(
        {
            experiment_directory / ".biowulf_train.lock",
            validation_output_directory / ".biowulf_validation.lock",
        },
        key=lambda path: path.as_posix(),
    )
    namespace_handle = None
    acquired_namespace_offsets: List[int] = []
    handles = []
    try:
        try:
            namespace_handle = open_regular_lock_file(namespace_registry, binary=True)
            for offset, spec in sorted(namespace_specs.items()):
                try:
                    fcntl.lockf(
                        namespace_handle.fileno(),
                        spec["mode"] | fcntl.LOCK_NB,
                        1,
                        offset,
                        os.SEEK_SET,
                    )
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise TrainingLaunchError(
                            f"could not lock namespace registry {namespace_registry}: {exc}"
                        ) from exc
                    paths = ", ".join(str(path) for path in spec["paths"])
                    raise TrainingLaunchError(
                        "another training process holds an equal or nested output "
                        f"namespace affecting {paths} (registry: {namespace_registry})"
                    ) from exc
                acquired_namespace_offsets.append(offset)

            for lock_path in lock_paths:
                lock_handle = open_regular_lock_file(lock_path, binary=False)
                try:
                    fcntl.flock(
                        lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        lock_handle.close()
                        raise TrainingLaunchError(
                            f"could not acquire run lock {lock_path}: {exc}"
                        ) from exc
                    lock_handle.seek(0)
                    owner = lock_handle.read().strip() or "owner details unavailable"
                    lock_handle.close()
                    raise TrainingLaunchError(
                        f"another training process holds {lock_path}: {owner}"
                    ) from exc
                locked_stat = os.fstat(lock_handle.fileno())
                if not stat.S_ISREG(locked_stat.st_mode) or locked_stat.st_nlink != 1:
                    lock_handle.close()
                    raise TrainingLaunchError(
                        "run lock became unsafe before its owner record could be "
                        f"written: {lock_path}"
                    )
                handles.append(lock_handle)
                lock_handle.seek(0)
                lock_handle.truncate()
                lock_handle.write(
                    f"job_id={os.environ.get('SLURM_JOB_ID', '')} "
                    f"hostname={socket.gethostname()} pid={os.getpid()}\n"
                )
                lock_handle.flush()
                os.fsync(lock_handle.fileno())
        except TrainingLaunchError:
            raise
        except OSError as exc:
            raise TrainingLaunchError(f"could not establish training locks: {exc}") from exc
        yield tuple(lock_paths), namespace_registry
    finally:
        for lock_handle in reversed(handles):
            with contextlib.suppress(OSError):
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        if namespace_handle is not None:
            for offset in reversed(acquired_namespace_offsets):
                with contextlib.suppress(OSError):
                    fcntl.lockf(
                        namespace_handle.fileno(),
                        fcntl.LOCK_UN,
                        1,
                        offset,
                        os.SEEK_SET,
                    )
            namespace_handle.close()


def prepare_output_destinations(configuration: TrainingConfiguration) -> None:
    """Create and probe every directory needed before the first costly epoch."""
    directories = {
        configuration.experiment_directory,
        configuration.experiment_directory / "pth",
        configuration.experiment_directory / "onnx",
        configuration.validation_output_directory,
        *(path.parent for path in configuration.validation_output_files),
    }
    for directory in sorted(directories, key=lambda path: path.as_posix()):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if not directory.is_dir():
                raise TrainingLaunchError(
                    f"output path is not a directory: {directory}"
                )
            descriptor, probe_name = tempfile.mkstemp(
                prefix=".support-write-probe.", dir=directory
            )
            os.close(descriptor)
            Path(probe_name).unlink()
        except TrainingLaunchError:
            raise
        except OSError as exc:
            raise TrainingLaunchError(
                f"could not create or write output directory {directory}: {exc}"
            ) from exc
    for output_file in configuration.validation_output_files:
        if output_file.exists() and not output_file.is_file():
            raise TrainingLaunchError(
                f"validation output exists but is not a regular file: {output_file}"
            )


def run_training(configuration: TrainingConfiguration) -> None:
    validate_namespace_registry_location(configuration)
    # Set these before importing torch through the training modules.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import torch

    if not configuration.use_cpu and not torch.cuda.is_available():
        raise TrainingLaunchError(
            "no CUDA GPU is visible; request a GPU or pass --use-cpu"
        )

    from utils.train_collection import TrainingClass
    from utils.validation import ValidationTracker

    trainer = None
    validation_tracker = None
    with exclusive_training_locks(
        configuration.experiment_directory,
        configuration.validation_output_directory,
    ) as (lock_paths, namespace_registry):
        print("Exclusive run locks: " + ", ".join(str(path) for path in lock_paths))
        print(f"Hierarchical namespace registry: {namespace_registry}")
        manifest_state = validate_run_manifest(configuration, commit=True)
        print(
            f"Run manifest: {configuration.experiment_directory / RUN_MANIFEST_NAME} "
            f"({manifest_state})"
        )
        prepare_output_destinations(configuration)
        try:
            trainer = TrainingClass(configuration.training_parameters)
            trainer._frozen_input_paths = tuple(
                str(description.path) for description in configuration.training_tiffs
            )
            trainer.vars_dict()
            validation_tracker = ValidationTracker(
                test_dir=str(configuration.validation_path),
                test_params=configuration.validation_parameters,
                output_dir=str(configuration.validation_output_directory),
                validation_paths=tuple(
                    str(description.path)
                    for description in configuration.validation_tiffs
                ),
            )
            trainer.epoch_end_callback = validation_tracker
            print(
                f"Found {len(validation_tracker.validation_paths)} held-out "
                "validation TIFF(s)."
            )
            trainer.run()
        finally:
            if trainer is not None:
                trainer.release_resources()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if trainer is not None:
        print(f"Training complete. Checkpoints: {trainer.pth_dir}")
        print(f"Latest model: {trainer.model_path}")
    if validation_tracker is not None:
        history_csv = validation_tracker.history_csv or (
            configuration.experiment_directory
            / f"{configuration.experiment_name}_train_validation_losses.csv"
        )
        per_file_csv = validation_tracker.per_file_csv or (
            configuration.experiment_directory
            / f"{configuration.experiment_name}_validation_files.csv"
        )
        print(f"Paired loss history: {history_csv}")
        print(f"Per-file validation history: {per_file_csv}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configuration = build_configuration(args, parser)
    try:
        manifest_state = validate_run_manifest(configuration, commit=False)
    except TrainingLaunchError as exc:
        parser.error(str(exc))
    print_configuration(configuration, args.dry_run)
    print(f"Run-state preflight: {manifest_state}")
    if args.dry_run:
        return 0
    try:
        run_training(configuration)
    except TrainingLaunchError as exc:
        print(f"Training configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
