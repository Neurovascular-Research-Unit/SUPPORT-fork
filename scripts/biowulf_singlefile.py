#!/usr/bin/env python3
"""Train and test one non-generalized SUPPORT model per logical 2DT stack.

Two on-disk layouts are supported below one input root:

* Prairie View OME-TIFF acquisitions split into temporal chunks and channels.
  Chunks are grouped and concatenated in job-local scratch by channel before a
  separate model is trained and tested for each channel.
* Directories containing exactly two ordinary ``.tif`` files.  Each TIFF is
  trained and tested independently, matching the legacy ``run_batch.py`` flow.

Only TIFFs whose full path contains the configured marker (``2DT`` by default)
are considered.  Acquisition ``References`` directories are intentionally
excluded.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import shutil
import socket
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tifffile


REPO_DIRECTORY = Path(__file__).resolve().parent.parent
if str(REPO_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPO_DIRECTORY))

DEFAULT_PARAMETERS = REPO_DIRECTORY / "notebooks" / "parameters.json"
DEFAULT_OUTPUT_DIRECTORY = REPO_DIRECTORY / "notebooks" / "outputs"
DEFAULT_WORK_DIRECTORY_NAME = ".support_singlefile_models"
RUN_MANIFEST_NAME = ".biowulf_singlefile_config.json"
COMPLETION_RECORD_NAME = ".biowulf_singlefile_complete.json"
RUN_LOCK_NAME = ".biowulf_singlefile.lock"
TIFF_SUFFIXES = {".tif", ".tiff"}
REFERENCE_DIRECTORY_NAMES = {"reference", "references"}
SCENARIO_OME = "ome_chunks"
SCENARIO_SINGLE = "single_tiff"
VALID_NDIMS = {
    "2+1D": {3},
    "3D": {3},
    "3+1D": {3, 4},
}

# The acquisition suffix (for example ``-001``), Cycle, channel, and final
# chunk number are deliberately parsed separately.  Do not broadly remove
# digits: dates, FOV2, 250Deep, 3X, and 2DT are meaningful identity tokens.
OME_CHUNK_PATTERN = re.compile(
    r"^(?P<acquisition>.+?)"
    r"(?:_Cycle(?P<cycle>\d+))?"
    r"_Ch(?P<channel>\d+)"
    r"_(?P<part>\d+)\.ome\.tiff?$",
    re.IGNORECASE,
)
ACQUISITION_RUN_SUFFIX = re.compile(r"-\d+$")


class SingleFileConfigurationError(ValueError):
    """Raised when discovery or a saved run is unsafe or ambiguous."""


class WorkItemBusyError(RuntimeError):
    """Raised when another process currently owns a logical job's run lock."""


class SavedRunMismatchError(SingleFileConfigurationError):
    """Raised when saved checkpoints are incompatible with the current job."""


@dataclass(frozen=True)
class OmeChunkName:
    acquisition: str
    cycle: int
    channel: int
    part: int


@dataclass(frozen=True)
class WorkItem:
    """One independent train-then-test job and its final output identity."""

    scenario: str
    source_paths: Tuple[Path, ...]
    output_stem: str
    channel_number: Optional[int] = None
    acquisition: Optional[str] = None
    temporal_keys: Tuple[Tuple[int, int], ...] = ()


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
class PreparedWorkItem:
    item: WorkItem
    sources: Tuple[TiffDescription, ...]
    logical_shape: Tuple[int, ...]
    logical_dtype: str

    @property
    def input_bytes(self) -> int:
        return math.prod(self.logical_shape) * np.dtype(self.logical_dtype).itemsize

    @property
    def output_bytes(self) -> int:
        # TestingClass always publishes float32 TIFFs.
        return math.prod(self.logical_shape) * np.dtype(np.float32).itemsize


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


def natural_key(value: object) -> Tuple[Tuple[int, object], ...]:
    """Return a deterministic, case-insensitive key with numeric ordering."""
    parts = re.split(r"(\d+)", str(value).casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def paths_overlap(first: Path, second: Path) -> bool:
    return path_is_within(first, second) or path_is_within(second, first)


def normalize_bool(value: Any, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise SingleFileConfigurationError(f"{name} must be true or false")


def slurm_cpu_count(default: int = 16) -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value is None:
        return default
    try:
        count = int(value)
    except ValueError:
        return default
    return count if count >= 0 else default


def parse_ome_chunk_name(path: Path) -> Optional[OmeChunkName]:
    match = OME_CHUNK_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return OmeChunkName(
        acquisition=match.group("acquisition"),
        cycle=int(match.group("cycle") or 0),
        channel=int(match.group("channel")),
        part=int(match.group("part")),
    )


def normalize_acquisition_name(value: str) -> str:
    return ACQUISITION_RUN_SUFFIX.sub("", value)


def _contains_reference_directory(path: Path, input_root: Path) -> bool:
    relative = path.relative_to(input_root)
    return any(
        component.casefold() in REFERENCE_DIRECTORY_NAMES
        for component in relative.parts[:-1]
    )


def _ome_output_stem(
    input_root: Path,
    parent: Path,
    acquisition: str,
    channel: int,
) -> str:
    normalized_acquisition = normalize_acquisition_name(acquisition)
    relative_parent = parent.relative_to(input_root)
    parent_parts = [] if relative_parent == Path(".") else list(relative_parent.parts)

    # Raw copies are sometimes stored inside a directory named exactly like
    # the acquisition (for example .../FOV2_..._2DT-001/*.ome.tif).  Omitting
    # that last directory avoids duplicating the acquisition in the flattened
    # name.  A meaningful containing directory such as FOV2_..._2D remains.
    if (
        parent_parts
        and normalize_acquisition_name(parent_parts[-1]).casefold()
        == normalized_acquisition.casefold()
    ):
        parent_parts.pop()

    components = [*parent_parts, normalized_acquisition, f"Ch{channel}"]
    if any(component in {"", ".", ".."} for component in components):
        raise SingleFileConfigurationError(
            f"could not construct a safe output stem for {parent}"
        )
    return "_".join(components)


def _ordinary_output_stem(path: Path) -> str:
    stem = path.stem
    if stem.casefold().endswith("_sprt"):
        stem = stem[:-5]
    if not stem:
        raise SingleFileConfigurationError(f"TIFF has an empty output stem: {path}")
    return stem


def discover_work_items(
    input_root: Path,
    output_directory: Path,
    marker: str = "2DT",
) -> List[WorkItem]:
    """Discover both supported layouts and return independent logical jobs.

    Scenario-1 chunks are grouped by direct parent, acquisition prefix, and
    channel, then sorted numerically by ``(Cycle, part)``.  Scenario-2 is
    intentionally strict: its directory must contain exactly two direct-child
    ordinary ``.tif`` files and no OME chunks.
    """
    input_root = Path(input_root).expanduser().resolve(strict=True)
    output_directory = Path(output_directory).expanduser().resolve(strict=False)
    if not input_root.is_dir():
        raise NotADirectoryError(f"input root is not a directory: {input_root}")
    if not marker or not marker.strip():
        raise SingleFileConfigurationError("the path marker must not be empty")
    if paths_overlap(input_root, output_directory):
        raise SingleFileConfigurationError(
            "input and output directories must not overlap; generated _SPRT TIFFs "
            "could otherwise be rediscovered as noisy inputs"
        )

    marker_folded = marker.casefold()
    candidates: List[Path] = []
    excluded_references = 0
    for path in input_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in TIFF_SUFFIXES:
            continue
        if marker_folded not in path.as_posix().casefold():
            continue
        if _contains_reference_directory(path, input_root):
            excluded_references += 1
            continue
        if path.is_symlink():
            raise SingleFileConfigurationError(
                f"TIFF inputs must not be symbolic links: {path}"
            )
        candidates.append(path.resolve(strict=True))

    if not candidates:
        suffix = (
            f" ({excluded_references} reference TIFFs were intentionally excluded)"
            if excluded_references
            else ""
        )
        raise FileNotFoundError(
            f"no eligible TIFF files with {marker!r} in their path were found "
            f"below {input_root}{suffix}"
        )

    by_parent: Dict[Path, List[Path]] = {}
    for path in candidates:
        by_parent.setdefault(path.parent, []).append(path)

    work_items: List[WorkItem] = []
    for parent in sorted(by_parent, key=natural_key):
        paths = sorted(by_parent[parent], key=lambda path: natural_key(path.name))
        parsed = [(path, parse_ome_chunk_name(path)) for path in paths]
        ome_paths = [(path, parsed_name) for path, parsed_name in parsed if parsed_name]

        if ome_paths:
            malformed = [path for path, parsed_name in parsed if parsed_name is None]
            if malformed:
                raise SingleFileConfigurationError(
                    "a dataset directory mixes recognized acquisition OME chunks "
                    "with ordinary or malformed TIFFs: "
                    f"{parent}; unexpected: {', '.join(path.name for path in malformed)}"
                )

            grouped: Dict[Tuple[str, int], List[Tuple[Path, OmeChunkName]]] = {}
            for path, parsed_name in ome_paths:
                assert parsed_name is not None
                grouped.setdefault(
                    (parsed_name.acquisition, parsed_name.channel), []
                ).append((path, parsed_name))

            for (acquisition, channel), chunks in grouped.items():
                seen_keys: Dict[Tuple[int, int], Path] = {}
                for path, chunk in chunks:
                    temporal_key = (chunk.cycle, chunk.part)
                    if temporal_key in seen_keys:
                        raise SingleFileConfigurationError(
                            "duplicate OME temporal index for acquisition "
                            f"{acquisition!r} Ch{channel}: {seen_keys[temporal_key]} "
                            f"and {path} both use Cycle={chunk.cycle}, part={chunk.part}"
                        )
                    seen_keys[temporal_key] = path
                ordered = sorted(
                    chunks,
                    key=lambda pair: (
                        pair[1].cycle,
                        pair[1].part,
                        natural_key(pair[0].name),
                    ),
                )
                work_items.append(
                    WorkItem(
                        scenario=SCENARIO_OME,
                        source_paths=tuple(path for path, _ in ordered),
                        output_stem=_ome_output_stem(
                            input_root, parent, acquisition, channel
                        ),
                        channel_number=channel,
                        acquisition=normalize_acquisition_name(acquisition),
                        temporal_keys=tuple(
                            (chunk.cycle, chunk.part) for _, chunk in ordered
                        ),
                    )
                )
            continue

        # A filename ending in .ome.tif but not matching the acquisition grammar
        # is not an ordinary two-file dataset.  Reject it instead of silently
        # training on a mislabeled or incomplete channel collection.
        malformed_ome = [
            path for path in paths if path.name.casefold().endswith((".ome.tif", ".ome.tiff"))
        ]
        if malformed_ome:
            raise SingleFileConfigurationError(
                "OME-TIFF filenames do not contain the required acquisition, "
                f"ChN, and temporal-index tokens in {parent}: "
                + ", ".join(path.name for path in malformed_ome)
            )
        if len(paths) != 2 or any(path.suffix.casefold() != ".tif" for path in paths):
            raise SingleFileConfigurationError(
                "an ordinary 2DT dataset directory must contain exactly two "
                f"direct-child .tif files; found {len(paths)} eligible TIFF(s) in {parent}"
            )
        for path in paths:
            work_items.append(
                WorkItem(
                    scenario=SCENARIO_SINGLE,
                    source_paths=(path,),
                    output_stem=_ordinary_output_stem(path),
                )
            )

    work_items.sort(
        key=lambda item: (
            natural_key(item.output_stem),
            natural_key(item.source_paths[0]),
        )
    )

    destinations: Dict[str, WorkItem] = {}
    for item in work_items:
        output_name = f"{item.output_stem}_SPRT.tif"
        collision_key = output_name.casefold()
        previous = destinations.get(collision_key)
        if previous is not None:
            raise SingleFileConfigurationError(
                "flattened output-name collision: "
                f"{previous.source_paths} and {item.source_paths} would both write "
                f"{output_directory / output_name}"
            )
        destinations[collision_key] = item

    return work_items


def inspect_tiff(path: Path) -> TiffDescription:
    if path.is_symlink():
        raise SingleFileConfigurationError(f"TIFF inputs must not be symlinks: {path}")
    before = path.stat()
    if before.st_size <= 0:
        raise SingleFileConfigurationError(f"TIFF is empty: {path}")
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise ValueError("contains no image series")
            series = tif.series[0]
            shape = tuple(int(value) for value in series.shape)
            dtype = np.dtype(series.dtype).str
            if not series.pages:
                raise ValueError("contains no image pages")
    except Exception as exc:
        raise SingleFileConfigurationError(f"could not inspect TIFF {path}: {exc}") from exc
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
        raise SingleFileConfigurationError(
            f"TIFF changed while its header was being inspected: {path}"
        )
    return TiffDescription(
        path=path,
        shape=shape,
        dtype=dtype,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        device=after.st_dev,
        inode=after.st_ino,
    )


def _training_stack_shape(
    shape: Tuple[int, ...], dimensionality: str
) -> Tuple[int, int, int]:
    if dimensionality == "3+1D" and len(shape) == 4:
        return shape[0], shape[2], shape[3]
    return shape  # type: ignore[return-value]


def prepare_work_items(
    work_items: Sequence[WorkItem],
    dimensionality: str,
    patch_size: Sequence[int],
) -> List[PreparedWorkItem]:
    prepared: List[PreparedWorkItem] = []
    for item in work_items:
        descriptions = tuple(inspect_tiff(path) for path in item.source_paths)
        if item.scenario == SCENARIO_OME:
            if dimensionality != "2+1D":
                raise SingleFileConfigurationError(
                    "temporally chunked 2DT OME acquisitions require "
                    "--dimensionality 2+1D"
                )
            if any(len(description.shape) != 3 for description in descriptions):
                raise SingleFileConfigurationError(
                    f"OME chunks for {item.output_stem} must all be TYX stacks"
                )
            spatial_shapes = {description.shape[1:] for description in descriptions}
            dtypes = {description.dtype for description in descriptions}
            if len(spatial_shapes) != 1:
                raise SingleFileConfigurationError(
                    f"OME chunks for {item.output_stem} have different YX shapes: "
                    + ", ".join(str(description.shape) for description in descriptions)
                )
            if len(dtypes) != 1:
                raise SingleFileConfigurationError(
                    f"OME chunks for {item.output_stem} have different dtypes: "
                    + ", ".join(sorted(dtypes))
                )
            spatial_shape = next(iter(spatial_shapes))
            logical_shape = (
                sum(description.shape[0] for description in descriptions),
                *spatial_shape,
            )
            logical_dtype = descriptions[0].dtype
        else:
            description = descriptions[0]
            allowed_ndims = VALID_NDIMS[dimensionality]
            if len(description.shape) not in allowed_ndims:
                allowed = " or ".join(str(value) for value in sorted(allowed_ndims))
                raise SingleFileConfigurationError(
                    f"{dimensionality} expects a {allowed}-dimensional TIFF, but "
                    f"{description.path} has shape {description.shape}"
                )
            logical_shape = description.shape
            logical_dtype = description.dtype

        stack_shape = _training_stack_shape(logical_shape, dimensionality)
        if any(actual < required for actual, required in zip(stack_shape, patch_size)):
            raise SingleFileConfigurationError(
                f"training patch {list(patch_size)} is larger than logical stack "
                f"{stack_shape} for {item.output_stem}"
            )
        prepared.append(
            PreparedWorkItem(
                item=item,
                sources=descriptions,
                logical_shape=tuple(int(value) for value in logical_shape),
                logical_dtype=logical_dtype,
            )
        )
    return prepared


def verify_sources_unchanged(prepared: PreparedWorkItem) -> None:
    for description in prepared.sources:
        current = description.path.stat()
        identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        )
        expected = (
            description.device,
            description.inode,
            description.size,
            description.mtime_ns,
        )
        if identity != expected:
            raise SingleFileConfigurationError(
                f"input TIFF changed after preflight: {description.path}"
            )


def _close_memmap(array: np.ndarray) -> None:
    mapped_file = getattr(array, "_mmap", None)
    if mapped_file is not None:
        mapped_file.close()


def _copy_tiff_into(
    source: Path,
    destination: np.memmap,
    offset: int,
    expected_shape: Tuple[int, ...],
) -> None:
    """Copy one TIFF with bounded RAM, preferring direct memory mapping."""
    source_array = None
    try:
        try:
            source_array = tifffile.memmap(source, mode="r", series=0)
        except (OSError, ValueError):
            source_array = None

        if source_array is not None:
            if tuple(source_array.shape) != expected_shape:
                raise SingleFileConfigurationError(
                    f"TIFF shape changed before staging: {source}"
                )
            frames_per_copy = max(1, (64 * 1024 * 1024) // max(1, source_array[0].nbytes))
            for start in range(0, expected_shape[0], frames_per_copy):
                stop = min(expected_shape[0], start + frames_per_copy)
                destination[offset + start : offset + stop] = source_array[start:stop]
            return

        with tifffile.TiffFile(source) as tif:
            series = tif.series[0]
            if tuple(series.shape) != expected_shape:
                raise SingleFileConfigurationError(
                    f"TIFF shape changed before staging: {source}"
                )
            if len(series.pages) != expected_shape[0]:
                raise SingleFileConfigurationError(
                    f"TIFF pages do not match its leading dimension: {source}"
                )
            for frame_index, page in enumerate(series.pages):
                destination[offset + frame_index] = page.asarray()
    finally:
        if source_array is not None:
            _close_memmap(source_array)


def concatenate_ome_chunks(paths: Sequence[Path], destination: Path) -> Path:
    """Concatenate one channel's ordered OME chunks into a scratch TYX BigTIFF."""
    source_paths = tuple(Path(path).expanduser().resolve(strict=True) for path in paths)
    if not source_paths:
        raise SingleFileConfigurationError("at least one OME chunk is required")
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"staged destination already exists: {destination}")

    descriptions = tuple(inspect_tiff(path) for path in source_paths)
    if any(len(description.shape) != 3 for description in descriptions):
        raise SingleFileConfigurationError("OME chunks must all be three-dimensional TYX stacks")
    spatial_shapes = {description.shape[1:] for description in descriptions}
    dtypes = {description.dtype for description in descriptions}
    if len(spatial_shapes) != 1:
        raise SingleFileConfigurationError("OME chunks have incompatible spatial shapes")
    if len(dtypes) != 1:
        raise SingleFileConfigurationError("OME chunks have incompatible dtypes")
    shape = (
        sum(description.shape[0] for description in descriptions),
        *next(iter(spatial_shapes)),
    )
    dtype = np.dtype(descriptions[0].dtype)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial.tif",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    # tifffile.memmap creates the TIFF layout and therefore requires a new path.
    temporary_path.unlink()
    staged_array = None
    try:
        staged_array = tifffile.memmap(
            temporary_path,
            shape=shape,
            dtype=dtype,
            bigtiff=True,
            photometric="minisblack",
            metadata={"axes": "TYX"},
        )
        offset = 0
        for description in descriptions:
            print(
                f"  staging {description.path.name}: frames "
                f"{offset}..{offset + description.shape[0] - 1}",
                flush=True,
            )
            _copy_tiff_into(
                description.path,
                staged_array,
                offset,
                description.shape,
            )
            offset += description.shape[0]
        staged_array.flush()
        _close_memmap(staged_array)
        staged_array = None
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        with tifffile.TiffFile(temporary_path) as tif:
            series = tif.series[0]
            if (
                not tif.is_bigtiff
                or tuple(series.shape) != shape
                or np.dtype(series.dtype) != dtype
                or series.axes != "TYX"
            ):
                raise SingleFileConfigurationError(
                    f"staged TIFF verification failed: {temporary_path}"
                )
        os.replace(temporary_path, destination)
        return destination
    finally:
        if staged_array is not None:
            _close_memmap(staged_array)
        if temporary_path.exists():
            temporary_path.unlink()


def load_parameter_sections(
    path: Path, dimensionality: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        section = document[dimensionality]
        training = section["train"]
        testing = section["test"]
    except FileNotFoundError as exc:
        raise SingleFileConfigurationError(f"parameter file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SingleFileConfigurationError(
            f"could not read {dimensionality!r} train/test parameters from {path}: {exc}"
        ) from exc
    if not isinstance(training, dict) or not isinstance(testing, dict):
        raise SingleFileConfigurationError(
            f"{dimensionality!r} train and test parameter sections must be objects"
        )
    return dict(training), dict(testing)


def _validate_triplet(value: Any, name: str) -> List[int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(item) is not int or item < 1 for item in value)
    ):
        raise SingleFileConfigurationError(
            f"{name} must contain exactly three positive integers"
        )
    return [int(item) for item in value]


def build_effective_parameters(
    args: argparse.Namespace,
    parameters_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    training, testing = load_parameter_sections(parameters_path, args.dimensionality)

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
        "rnd_seed": args.random_seed,
    }
    testing_overrides = {
        "batch_size": args.test_batch_size,
        "patch_interval": args.test_patch_interval,
        "prefetch_factor": args.test_prefetch_factor,
        "pin_memory": args.test_pin_memory,
        "pad": args.test_pad,
    }
    training.update(
        {name: value for name, value in training_overrides.items() if value is not None}
    )
    testing.update(
        {name: value for name, value in testing_overrides.items() if value is not None}
    )

    defaults = {
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
    test_defaults = {
        "patch_interval": [1, 11, 11],
        "batch_size": 16,
        "pin_memory": True,
        "prefetch_factor": 2,
        "pad": True,
    }
    for name, value in defaults.items():
        training.setdefault(name, value)
    for name, value in test_defaults.items():
        testing.setdefault(name, value)

    training["patch_size"] = _validate_triplet(training["patch_size"], "train.patch_size")
    training["patch_interval"] = _validate_triplet(
        training["patch_interval"], "train.patch_interval"
    )
    testing["patch_interval"] = _validate_triplet(
        testing["patch_interval"], "test.patch_interval"
    )
    if testing["patch_interval"][0] != 1:
        raise SingleFileConfigurationError(
            "test.patch_interval[0] must be 1 so every temporal plane is written"
        )
    for mapping, label in ((training, "train"), (testing, "test")):
        for name in ("batch_size", "prefetch_factor"):
            value = mapping[name]
            if type(value) is not int or value < 1:
                raise SingleFileConfigurationError(f"{label}.{name} must be positive")
    if type(training["n_epochs"]) is not int or training["n_epochs"] < 1:
        raise SingleFileConfigurationError("train.n_epochs must be positive")
    learning_rate = training["lr"]
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise SingleFileConfigurationError("train.lr must be finite and positive")
    training["load_to_memory"] = normalize_bool(
        training["load_to_memory"], "train.load_to_memory"
    )
    training["pin_memory"] = normalize_bool(training["pin_memory"], "train.pin_memory")
    training["use_amp"] = normalize_bool(training["use_amp"], "train.use_amp")
    training["bp"] = normalize_bool(training.get("bp", False), "train.bp")
    testing["pin_memory"] = normalize_bool(testing["pin_memory"], "test.pin_memory")
    testing["pad"] = normalize_bool(testing["pad"], "test.pad")
    return training, testing


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _configuration_signature(
    prepared: PreparedWorkItem,
    dimensionality: str,
    training: Mapping[str, Any],
    testing: Mapping[str, Any],
) -> Dict[str, Any]:
    training_keys = (
        "patch_size",
        "patch_interval",
        "batch_size",
        "load_to_memory",
        "pin_memory",
        "prefetch_factor",
        "lr",
        "loss_coef",
        "use_amp",
        "rnd_seed",
        "depth",
        "bc_channels",
        "obo_channels",
        "ll_channels",
        "unet_channels",
        "bs_size",
        "bp",
    )
    test_keys = ("patch_interval", "batch_size", "pin_memory", "prefetch_factor", "pad")
    return {
        "version": 1,
        "scenario": prepared.item.scenario,
        "output_stem": prepared.item.output_stem,
        "channel": prepared.item.channel_number,
        "acquisition": prepared.item.acquisition,
        "temporal_keys": prepared.item.temporal_keys,
        "dimensionality": dimensionality,
        "logical_shape": prepared.logical_shape,
        "logical_dtype": prepared.logical_dtype,
        "sources": [
            {
                "path": str(description.path),
                "shape": description.shape,
                "dtype": description.dtype,
                "size": description.size,
                "mtime_ns": description.mtime_ns,
                "device": description.device,
                "inode": description.inode,
            }
            for description in prepared.sources
        ],
        # n_epochs is intentionally excluded so a completed run can be extended.
        "training": {key: training.get(key) for key in training_keys},
        "testing": {key: testing.get(key) for key in test_keys},
    }


_VOLATILE_SOURCE_KEYS = frozenset({"device"})
_RUNTIME_TRAINING_KEYS = frozenset({"pin_memory", "prefetch_factor"})
_RUNTIME_TESTING_KEYS = frozenset(
    {"batch_size", "pin_memory", "prefetch_factor"}
)


def _mapping_without_keys(value: Any, ignored: frozenset[str]) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        str(key): _json_value(item)
        for key, item in sorted(value.items())
        if str(key) not in ignored
    }


def _stable_sources(value: Any) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    return [
        _mapping_without_keys(source, _VOLATILE_SOURCE_KEYS)
        for source in value
    ]


def _checkpoint_identity(signature: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only fields that must match before checkpoint weights are reused."""
    normalized = dict(_json_value(signature))
    # Inference settings do not affect whether training weights can resume.
    # All other top-level fields, including unknown future fields and the
    # schema version, remain fail-closed checkpoint identity.
    normalized.pop("testing", None)
    # st_dev can vary across mounts or compute nodes even when the same shared
    # file is addressed.  Inode/size/mtime/shape/dtype/path remain persistent
    # guards; live identity is rechecked again before training.
    normalized["sources"] = _stable_sources(normalized.get("sources"))
    # Pinning and prefetching affect throughput rather than model/checkpoint
    # compatibility.  load_to_memory stays in the identity because its
    # normalization path can affect training numerics.
    normalized["training"] = _mapping_without_keys(
        normalized.get("training"), _RUNTIME_TRAINING_KEYS
    )
    return normalized


def _completion_identity(signature: Mapping[str, Any]) -> Dict[str, Any]:
    """Return stable fields that determine whether a published output is reusable."""
    normalized = dict(_json_value(signature))
    normalized["sources"] = _stable_sources(normalized.get("sources"))
    normalized["training"] = _mapping_without_keys(
        normalized.get("training"), _RUNTIME_TRAINING_KEYS
    )
    normalized["testing"] = _mapping_without_keys(
        normalized.get("testing"), _RUNTIME_TESTING_KEYS
    )
    return normalized


def _differing_paths(first: Any, second: Any, prefix: str = "") -> List[str]:
    """Return concise JSON-style paths whose values differ."""
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        differences: List[str] = []
        for key in sorted(set(first) | set(second), key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in first or key not in second:
                differences.append(child)
            else:
                differences.extend(_differing_paths(first[key], second[key], child))
        return differences
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        differences = []
        if len(first) != len(second):
            differences.append(f"{prefix}.length" if prefix else "length")
        for index, (left, right) in enumerate(zip(first, second)):
            differences.extend(
                _differing_paths(left, right, f"{prefix}[{index}]")
            )
        return differences
    return [] if first == second else [prefix or "<root>"]


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_value(document), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _configuration_sha256(signature: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _completion_identity(signature),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_configuration_sha256(signature: Mapping[str, Any]) -> str:
    """Hash signatures exactly as completion-record version 1 originally did."""
    encoded = json.dumps(
        _json_value(signature),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _completion_configuration_matches(
    experiment_directory: Path,
    signature: Mapping[str, Any],
    recorded_hash: Any,
) -> bool:
    """Accept stable hashes plus provably equivalent legacy v1 hashes."""
    if recorded_hash in {
        _configuration_sha256(signature),
        _legacy_configuration_sha256(signature),
    }:
        return True

    # A deployed v1 record may hash the older manifest containing a different
    # prefetch count or mount device.  Accept it only when that saved manifest
    # has the same stable output identity as the current job.
    manifest_path = experiment_directory / RUN_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            saved_signature = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(saved_signature, dict)
        and _completion_identity(saved_signature) == _completion_identity(signature)
        and recorded_hash == _legacy_configuration_sha256(saved_signature)
    )


def validate_or_create_manifest(
    experiment_directory: Path,
    signature: Mapping[str, Any],
    resume: bool,
) -> str:
    manifest_path = experiment_directory / RUN_MANIFEST_NAME
    expected = _json_value(signature)
    if manifest_path.is_symlink():
        raise SingleFileConfigurationError(f"run manifest must not be a symlink: {manifest_path}")
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SingleFileConfigurationError(
                f"could not read saved run manifest {manifest_path}: {exc}"
            ) from exc
        if not isinstance(existing, dict):
            raise SingleFileConfigurationError(
                f"saved run manifest must contain a JSON object: {manifest_path}"
            )
        if not resume and any((experiment_directory / "pth").glob("model*.pth")):
            raise SingleFileConfigurationError(
                f"--no-resume cannot reuse existing checkpoints in {experiment_directory}"
            )
        if existing != expected:
            existing_identity = _checkpoint_identity(existing)
            expected_identity = _checkpoint_identity(expected)
            if existing_identity == expected_identity:
                # Preserve the original manifest for auditability and for
                # validating legacy completion-record hashes.
                manifest_state = "compatible saved run; runtime differences accepted"
            else:
                differences = _differing_paths(existing_identity, expected_identity)
                shown = differences[:12]
                difference_text = ", ".join(shown) or "unknown fields"
                if len(differences) > len(shown):
                    difference_text += f", and {len(differences) - len(shown)} more"
                raise SavedRunMismatchError(
                    "saved checkpoint identity does not match the current job "
                    f"({difference_text}): {manifest_path}"
                )
        else:
            manifest_state = "matching saved run"
        return manifest_state

    meaningful = [
        path
        for path in experiment_directory.iterdir()
        if path.name not in {RUN_LOCK_NAME}
    ]
    if meaningful:
        raise SingleFileConfigurationError(
            "refusing to attach a new run manifest to a nonempty experiment "
            f"directory: {experiment_directory}"
        )
    _atomic_write_json(manifest_path, expected)
    return "new run"


def latest_complete_checkpoint_epoch(
    experiment_directory: Path, use_amp: bool
) -> Optional[int]:
    checkpoint_directory = experiment_directory / "pth"
    if not checkpoint_directory.is_dir():
        return None
    pattern = re.compile(r"^model_(\d+)\.pth$")
    legacy_pattern = re.compile(r"^model(\d+)\.pth$")
    epochs: List[int] = []
    for model_path in checkpoint_directory.iterdir():
        match = pattern.fullmatch(model_path.name)
        if match is None:
            legacy_match = legacy_pattern.fullmatch(model_path.name)
            if legacy_match is not None and model_path.is_file():
                completed_epochs = int(legacy_match.group(1))
                if completed_epochs >= 1:
                    epochs.append(completed_epochs - 1)
            continue
        epoch = int(match.group(1))
        required = [
            model_path,
            checkpoint_directory / f"optimizer_{epoch}.pth",
        ]
        if use_amp:
            required.append(checkpoint_directory / f"scaler_{epoch}.pth")
        incomplete = checkpoint_directory / f".checkpoint_{epoch}.incomplete"
        if not incomplete.exists() and all(path.is_file() for path in required):
            epochs.append(epoch)
    return max(epochs) if epochs else None


def write_completion_record(
    experiment_directory: Path,
    signature: Mapping[str, Any],
    completed_epochs: int,
    model_path: Path,
    output_path: Path,
) -> None:
    model_stat = model_path.stat()
    output_stat = output_path.stat()
    record = {
        "version": 1,
        "configuration_sha256": _configuration_sha256(signature),
        "completed_epochs": int(completed_epochs),
        "model": {
            "path": str(model_path),
            "size": model_stat.st_size,
            "mtime_ns": model_stat.st_mtime_ns,
            "device": model_stat.st_dev,
            "inode": model_stat.st_ino,
        },
        "output": {
            "path": str(output_path),
            "size": output_stat.st_size,
            "mtime_ns": output_stat.st_mtime_ns,
            "device": output_stat.st_dev,
            "inode": output_stat.st_ino,
        },
    }
    _atomic_write_json(experiment_directory / COMPLETION_RECORD_NAME, record)


def completed_output_is_current(
    experiment_directory: Path,
    signature: Mapping[str, Any],
    requested_epochs: int,
    output_path: Path,
    expected_shape: Tuple[int, ...],
    use_amp: bool = False,
) -> bool:
    record_path = experiment_directory / COMPLETION_RECORD_NAME
    if record_path.is_symlink():
        raise SingleFileConfigurationError(
            f"completion record must not be a symlink: {record_path}"
        )
    if not record_path.is_file():
        return False
    try:
        with record_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if not isinstance(record, dict):
            return False
        if (
            record.get("version") != 1
            or not _completion_configuration_matches(
                experiment_directory,
                signature,
                record.get("configuration_sha256"),
            )
            or int(record.get("completed_epochs", -1)) != int(requested_epochs)
        ):
            return False

        output_record = record["output"]
        if Path(output_record["path"]) != output_path or not output_path.is_file():
            return False
        output_stat = output_path.stat()
        if (
            output_stat.st_size != int(output_record["size"])
            or output_stat.st_mtime_ns != int(output_record["mtime_ns"])
            or output_stat.st_ino != int(output_record["inode"])
        ):
            return False

        model_record = record["model"]
        model_path = Path(model_record["path"])
        expected_model_directory = (experiment_directory / "pth").resolve(
            strict=False
        )
        model_match = re.fullmatch(r"model_(\d+)\.pth", model_path.name)
        if (
            model_match is None
            or int(model_match.group(1)) + 1 != int(requested_epochs)
            or not path_is_within(
                model_path.resolve(strict=False), expected_model_directory
            )
            or not model_path.is_file()
        ):
            return False
        model_epoch = int(model_match.group(1))
        required_bundle = [
            expected_model_directory / f"optimizer_{model_epoch}.pth",
        ]
        if use_amp:
            required_bundle.append(
                expected_model_directory / f"scaler_{model_epoch}.pth"
            )
        if (
            (expected_model_directory / f".checkpoint_{model_epoch}.incomplete").exists()
            or not all(path.is_file() for path in required_bundle)
        ):
            return False
        model_stat = model_path.stat()
        if (
            model_stat.st_size != int(model_record["size"])
            or model_stat.st_mtime_ns != int(model_record["mtime_ns"])
            or model_stat.st_ino != int(model_record["inode"])
        ):
            return False
        verify_output_tiff(output_path, expected_shape)
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"WARNING: completion record is not reusable ({type(exc).__name__}); "
            f"the output will be regenerated: {record_path}",
            file=sys.stderr,
        )
        return False


@contextlib.contextmanager
def exclusive_item_lock(experiment_directory: Path) -> Iterator[Path]:
    experiment_directory.mkdir(parents=True, exist_ok=True)
    if experiment_directory.is_symlink() or not experiment_directory.is_dir():
        raise SingleFileConfigurationError(
            f"experiment path must be a real directory: {experiment_directory}"
        )
    lock_path = experiment_directory / RUN_LOCK_NAME
    if lock_path.is_symlink():
        raise SingleFileConfigurationError(f"run lock must not be a symlink: {lock_path}")
    with lock_path.open("a+", encoding="utf-8") as handle:
        lock_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(lock_stat.st_mode):
            raise SingleFileConfigurationError(f"run lock is not a regular file: {lock_path}")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "owner details unavailable"
            raise WorkItemBusyError(
                f"another process is training {experiment_directory}: {owner}"
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


def _scratch_filename(output_stem: str) -> str:
    digest = hashlib.sha256(output_stem.encode("utf-8")).hexdigest()[:12]
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_stem)[:80]
    return f"{safe_prefix}-{digest}.tif"


@contextlib.contextmanager
def staged_input(
    prepared: PreparedWorkItem,
    scratch_directory: Path,
) -> Iterator[Path]:
    if prepared.item.scenario == SCENARIO_SINGLE:
        yield prepared.item.source_paths[0]
        return

    scratch_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(scratch_directory).free
    if free_bytes < prepared.input_bytes:
        raise SingleFileConfigurationError(
            f"not enough scratch space in {scratch_directory}: need at least "
            f"{prepared.input_bytes / (1024 ** 3):.2f} GiB, have "
            f"{free_bytes / (1024 ** 3):.2f} GiB"
        )
    with tempfile.TemporaryDirectory(
        prefix="support-singlefile-", dir=scratch_directory
    ) as temporary_directory:
        destination = Path(temporary_directory) / _scratch_filename(
            prepared.item.output_stem
        )
        print(
            f"Concatenating {len(prepared.sources)} OME chunks for "
            f"{prepared.item.output_stem} into {destination}",
            flush=True,
        )
        concatenate_ome_chunks(prepared.item.source_paths, destination)
        staged = inspect_tiff(destination)
        if staged.shape != prepared.logical_shape or staged.dtype != prepared.logical_dtype:
            raise SingleFileConfigurationError(
                f"staged input does not match preflight for {prepared.item.output_stem}"
            )
        yield destination


def build_training_parameters(
    base: Mapping[str, Any],
    args: argparse.Namespace,
    input_path: Path,
    work_directory: Path,
    output_stem: str,
) -> Dict[str, Any]:
    training = dict(base)
    training.update(
        {
            "in_path": str(input_path),
            "out_dir": str(work_directory),
            "exp_name": output_stem,
            "dimensionality": args.dimensionality,
            "rm_dir": False,
            "epoch": 0,
            "checkpoints": True,
            "resume": bool(args.resume),
            "rm_intermediaries": bool(args.keep_only_latest),
            "train_gen": False,
            "save_every_epoch": False,
            "n_cpu": args.n_cpu if args.n_cpu is not None else slurm_cpu_count(),
            "cuda_device": "0",
            "use_cpu": bool(args.use_cpu),
        }
    )
    return training


def build_testing_parameters(
    test_base: Mapping[str, Any],
    trainer: Any,
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "in_path": str(input_path),
        "model_path": str(trainer.model_path),
        "exp_name": trainer.exp_name,
        "out_dir": str(output_path.parent),
        "output_path": str(output_path),
        "dimensionality": trainer.dimensionality,
        "cuda_device": trainer.cuda_device,
        "patch_size": list(trainer.patch_size),
        "patch_interval": list(test_base["patch_interval"]),
        "batch_size": int(test_base["batch_size"]),
        "pin_memory": bool(test_base["pin_memory"]),
        "depth": trainer.depth,
        "bc_channels": trainer.bc_channels,
        "obo_channels": list(trainer.obo_channels),
        "ll_channels": list(trainer.ll_channels),
        "unet_channels": list(trainer.unet_channels),
        "bs_size": list(trainer.bs_size),
        "bp": trainer.bp,
        "loss_coef": list(trainer.loss_coef),
        "n_cpu": trainer.n_cpu,
        "prefetch_factor": int(test_base["prefetch_factor"]),
        "pad": bool(test_base["pad"]),
        "use_cpu": bool(args.use_cpu),
        "track_loss": False,
        "rm_dir": False,
    }


def verify_output_tiff(path: Path, expected_shape: Tuple[int, ...]) -> None:
    if not path.is_file():
        raise SingleFileConfigurationError(f"inference did not create output TIFF: {path}")
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise SingleFileConfigurationError(f"output contains no TIFF series: {path}")
        series = tif.series[0]
        actual_shape = tuple(int(value) for value in series.shape)
        actual_dtype = np.dtype(series.dtype)
    if actual_shape != expected_shape:
        raise SingleFileConfigurationError(
            f"output shape {actual_shape} does not match input {expected_shape}: {path}"
        )
    if actual_dtype != np.dtype(np.float32):
        raise SingleFileConfigurationError(
            f"output dtype {actual_dtype} is not float32: {path}"
        )


def release_resources_preserving_error(resource: Any, label: str) -> None:
    """Release a model object without hiding an exception already in flight."""
    active_exception = sys.exc_info()[0] is not None
    try:
        resource.release_resources()
    except BaseException as exc:
        if not active_exception:
            raise
        print(
            f"WARNING: {label} resource cleanup also failed ({type(exc).__name__}); "
            "preserving the original error.",
            file=sys.stderr,
        )


def run_work_item(
    prepared: PreparedWorkItem,
    args: argparse.Namespace,
    training_base: Mapping[str, Any],
    testing_base: Mapping[str, Any],
    output_directory: Path,
    work_directory: Path,
    scratch_directory: Path,
    slack_notifier: Any,
) -> Path:
    item = prepared.item
    output_path = output_directory / f"{item.output_stem}_SPRT.tif"
    experiment_directory = work_directory / item.output_stem
    trainer = None
    try:
        with exclusive_item_lock(experiment_directory):
            # Do this before creating a new manifest.  Otherwise a source that
            # changed after batch preflight could strand a stale manifest and
            # turn every later submission into a persistent mismatch skip.
            verify_sources_unchanged(prepared)
            signature = _configuration_signature(
                prepared,
                args.dimensionality,
                training_base,
                testing_base,
            )
            manifest_state = validate_or_create_manifest(
                experiment_directory, signature, bool(args.resume)
            )
            print(f"Run manifest: {manifest_state} ({experiment_directory / RUN_MANIFEST_NAME})")
            requested_epochs = int(training_base["n_epochs"])
            latest_epoch = latest_complete_checkpoint_epoch(
                experiment_directory,
                bool(training_base["use_amp"]),
            )
            if latest_epoch is not None and latest_epoch + 1 > requested_epochs:
                raise SingleFileConfigurationError(
                    f"saved checkpoint epoch target {latest_epoch + 1} is newer than "
                    f"the requested --n-epochs {requested_epochs} for {item.output_stem}; "
                    "raise the target or use a new work directory"
                )
            if (
                not args.rerun_completed
                and completed_output_is_current(
                    experiment_directory,
                    signature,
                    requested_epochs,
                    output_path,
                    prepared.logical_shape,
                    use_amp=bool(training_base["use_amp"]),
                )
            ):
                print(
                    f"Skipping completed and verified output: {output_path}",
                    flush=True,
                )
                return output_path
            with staged_input(prepared, scratch_directory) as input_path:
                from utils.test_collection import TestingClass
                from utils.train_collection import TrainingClass

                training_parameters = build_training_parameters(
                    training_base,
                    args,
                    input_path,
                    work_directory,
                    item.output_stem,
                )
                trainer = TrainingClass(training_parameters)
                trainer.vars_dict()
                if slack_notifier.enabled:
                    trainer.epoch_start_callback = slack_notifier.notify_epoch_started

                    def notify_epoch_completed(
                        active_trainer: Any,
                        epoch: int,
                        training_metrics: Mapping[str, Any],
                        validation_metrics: Optional[Mapping[str, Any]],
                    ) -> None:
                        slack_notifier.notify_epoch_completed(
                            active_trainer,
                            epoch,
                            training_metrics,
                            validation_metrics,
                        )

                    trainer.epoch_complete_callback = notify_epoch_completed
                slack_notifier.notify_run_started(
                    trainer,
                    training_count=len(item.source_paths),
                    validation_count=0,
                )
                try:
                    trainer.run()
                    if not trainer.model_path or not Path(trainer.model_path).is_file():
                        raise SingleFileConfigurationError(
                            f"training produced no usable model for {item.output_stem}"
                        )
                finally:
                    release_resources_preserving_error(trainer, "trainer")
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                testing_parameters = build_testing_parameters(
                    testing_base,
                    trainer,
                    input_path,
                    output_path,
                    args,
                )
                tester = TestingClass(testing_parameters)
                try:
                    tester.run()
                finally:
                    release_resources_preserving_error(tester, "tester")
                    del tester
                    gc.collect()
                verify_output_tiff(output_path, prepared.logical_shape)
                write_completion_record(
                    experiment_directory,
                    signature,
                    requested_epochs,
                    Path(trainer.model_path),
                    output_path,
                )
                slack_notifier.notify_run_completed(trainer)
                print(f"Completed {item.output_stem}: {output_path}", flush=True)
                return output_path
    except (WorkItemBusyError, SavedRunMismatchError):
        # Lock contention and an incompatible saved run are safe per-item
        # skips.  The caller reports them and continues, so neither should be
        # reported to Slack as a training failure.
        raise
    except BaseException as exc:
        slack_notifier.notify_failure(
            exc,
            trainer=trainer,
            stage="per-dataset training or inference",
        )
        raise


def _format_gib(byte_count: int) -> str:
    return f"{byte_count / (1024 ** 3):.2f} GiB"


def print_plan(
    prepared_items: Sequence[PreparedWorkItem],
    output_directory: Path,
    work_directory: Path,
    scratch_directory: Path,
    args: argparse.Namespace,
) -> None:
    prefix = "Dry run: " if args.dry_run else ""
    print(f"{prefix}input root: {args.input_directory.expanduser().resolve()}")
    print(f"Path marker: {args.marker!r} (case-insensitive)")
    print(f"Logical train/test jobs: {len(prepared_items)}")
    print(f"Mode: {args.dimensionality}; train_gen=False (one model per listed job)")
    print(f"Clean output directory: {output_directory}")
    print(f"Persistent model/checkpoint directory: {work_directory}")
    print(f"OME concatenation scratch: {scratch_directory}")
    for number, prepared in enumerate(prepared_items, start=1):
        item = prepared.item
        channel = f", Ch{item.channel_number}" if item.channel_number is not None else ""
        print(
            f"[{number}/{len(prepared_items)}] {item.scenario}{channel}: "
            f"{len(item.source_paths)} source TIFF(s) -> "
            f"{output_directory / (item.output_stem + '_SPRT.tif')}"
        )
        print(
            f"  logical shape={prepared.logical_shape}, "
            f"dtype={np.dtype(prepared.logical_dtype).name}, "
            f"scratch input={_format_gib(prepared.input_bytes)}, "
            f"float32 output={_format_gib(prepared.output_bytes)}"
        )
        for source in prepared.sources:
            print(f"  {source.path} :: shape={source.shape}, dtype={np.dtype(source.dtype).name}")
    if args.dry_run:
        print("No directories, staged data, checkpoints, Slack messages, or outputs were changed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find 2DT TIFF datasets, then train and test a separate "
            "non-generalized SUPPORT model for every ordinary TIFF or OME channel."
        )
    )
    parser.add_argument(
        "--input-directory",
        "--in-path",
        required=True,
        type=Path,
        help="root searched recursively for eligible TIFF datasets",
    )
    parser.add_argument(
        "--output-directory",
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="flat destination for final _SPRT.tif files",
    )
    parser.add_argument(
        "--work-directory",
        type=Path,
        default=None,
        help=(
            "persistent checkpoint/log root (default: "
            "OUTPUT_DIRECTORY/.support_singlefile_models)"
        ),
    )
    parser.add_argument(
        "--scratch-directory",
        type=Path,
        default=None,
        help=(
            "scratch root for per-channel OME concatenation (default: "
            "SUPPORT_SINGLEFILE_SCRATCH_DIR, TMPDIR, or system temporary directory)"
        ),
    )
    parser.add_argument(
        "--marker",
        default="2DT",
        help="case-insensitive token required anywhere in each TIFF path",
    )
    parser.add_argument(
        "--dimensionality",
        choices=("2+1D", "3D", "3+1D"),
        default="2+1D",
        help="SUPPORT mode; chunked 2DT OME inputs require 2+1D",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=DEFAULT_PARAMETERS,
        help="read-only normal-mode train/test parameter JSON",
    )
    parser.add_argument("--n-epochs", type=positive_int)
    parser.add_argument("--n-cpu", type=nonnegative_int, default=None)
    parser.add_argument("--train-batch-size", "--batch-size", type=positive_int)
    parser.add_argument("--test-batch-size", type=positive_int)
    parser.add_argument(
        "--train-patch-size",
        "--patch-size",
        type=positive_int,
        nargs=3,
        metavar=("T_OR_Z", "Y", "X"),
    )
    parser.add_argument(
        "--train-patch-interval",
        "--patch-interval",
        type=positive_int,
        nargs=3,
        metavar=("T_OR_Z", "Y", "X"),
    )
    parser.add_argument(
        "--test-patch-interval",
        type=positive_int,
        nargs=3,
        metavar=("T_OR_Z", "Y", "X"),
    )
    parser.add_argument("--train-prefetch-factor", type=positive_int)
    parser.add_argument("--test-prefetch-factor", type=positive_int)
    parser.add_argument("--learning-rate", type=positive_float)
    parser.add_argument("--random-seed", type=nonnegative_int)

    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--load-to-memory", dest="load_to_memory", action="store_true"
    )
    memory_group.add_argument(
        "--no-load-to-memory", dest="load_to_memory", action="store_false"
    )
    parser.set_defaults(load_to_memory=None)

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--use-amp", dest="use_amp", action="store_true")
    amp_group.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=None)

    train_pin_group = parser.add_mutually_exclusive_group()
    train_pin_group.add_argument(
        "--train-pin-memory", dest="train_pin_memory", action="store_true"
    )
    train_pin_group.add_argument(
        "--no-train-pin-memory", dest="train_pin_memory", action="store_false"
    )
    parser.set_defaults(train_pin_memory=None)

    test_pin_group = parser.add_mutually_exclusive_group()
    test_pin_group.add_argument(
        "--test-pin-memory", dest="test_pin_memory", action="store_true"
    )
    test_pin_group.add_argument(
        "--no-test-pin-memory", dest="test_pin_memory", action="store_false"
    )
    parser.set_defaults(test_pin_memory=None)

    test_pad_group = parser.add_mutually_exclusive_group()
    test_pad_group.add_argument("--test-pad", dest="test_pad", action="store_true")
    test_pad_group.add_argument("--no-test-pad", dest="test_pad", action="store_false")
    parser.set_defaults(test_pad=None)

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)

    retention_group = parser.add_mutually_exclusive_group()
    retention_group.add_argument(
        "--keep-only-latest",
        dest="keep_only_latest",
        action="store_true",
        help="retain only the newest complete epoch checkpoint (default)",
    )
    retention_group.add_argument(
        "--keep-all-checkpoints",
        dest="keep_only_latest",
        action="store_false",
    )
    parser.set_defaults(keep_only_latest=True)

    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help=(
            "run inference again even when the saved completion record, model, "
            "and output all verify (default: skip completed logical jobs)"
        ),
    )

    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument(
        "--slack-notifications",
        action="store_true",
        help="enable start, epoch, completion, and failure Slack messages",
    )
    parser.add_argument(
        "--slack-webhook-file",
        type=Path,
        default=None,
        help="private mode-0600 Slack Incoming Webhook file; implies notifications",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect, group, validate, and print every job without writing",
    )
    return parser


def _resolve_scratch_directory(args: argparse.Namespace) -> Path:
    selected = (
        args.scratch_directory
        or os.environ.get("SUPPORT_SINGLEFILE_SCRATCH_DIR")
        or os.environ.get("TMPDIR")
        or tempfile.gettempdir()
    )
    return Path(selected).expanduser().resolve(strict=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from utils.slack_notifications import SlackConfigurationError, SlackNotifier

    slack_enabled = bool(args.slack_notifications or args.slack_webhook_file)

    def new_slack_notifier() -> Any:
        return SlackNotifier.from_environment(
            enabled=slack_enabled,
            webhook_file=args.slack_webhook_file,
        )

    try:
        preflight_notifier = new_slack_notifier()
    except SlackConfigurationError as exc:
        parser.error(str(exc))

    try:
        input_directory = args.input_directory.expanduser().resolve(strict=True)
        output_directory = args.output_directory.expanduser().resolve(strict=False)
        work_directory = (
            args.work_directory.expanduser().resolve(strict=False)
            if args.work_directory is not None
            else output_directory / DEFAULT_WORK_DIRECTORY_NAME
        )
        scratch_directory = _resolve_scratch_directory(args)
        parameters_path = args.parameters.expanduser().resolve(strict=True)
        if not input_directory.is_dir():
            raise SingleFileConfigurationError(
                f"input directory is not a directory: {input_directory}"
            )
        if paths_overlap(input_directory, output_directory):
            raise SingleFileConfigurationError(
                "input and output directories must not overlap"
            )
        if paths_overlap(input_directory, work_directory):
            raise SingleFileConfigurationError(
                "input and persistent work directories must not overlap"
            )
        if scratch_directory == input_directory or path_is_within(
            scratch_directory, input_directory
        ):
            raise SingleFileConfigurationError(
                "the OME scratch directory must not be inside the recursive input tree"
            )
        training_base, testing_base = build_effective_parameters(args, parameters_path)
        work_items = discover_work_items(
            input_directory,
            output_directory,
            marker=args.marker,
        )
        prepared_items = prepare_work_items(
            work_items,
            args.dimensionality,
            training_base["patch_size"],
        )
    except (FileNotFoundError, OSError, SingleFileConfigurationError) as exc:
        if preflight_notifier.enabled and not args.dry_run:
            preflight_notifier.notify_failure(exc, stage="single-file preflight")
        parser.error(str(exc))

    print_plan(
        prepared_items,
        output_directory,
        work_directory,
        scratch_directory,
        args,
    )
    print(
        "Slack notifications: "
        + ("enabled (fresh notifier per logical job)" if slack_enabled else "disabled")
    )
    if args.dry_run:
        return 0

    output_directory.mkdir(parents=True, exist_ok=True)
    work_directory.mkdir(parents=True, exist_ok=True)
    scratch_directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    # Set allocator and plotting options before importing torch via the model classes.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if not args.use_cpu:
        import torch

        if not torch.cuda.is_available():
            exc = SingleFileConfigurationError(
                "no CUDA GPU is visible; request a GPU or pass --use-cpu"
            )
            preflight_notifier.notify_failure(exc, stage="GPU preflight")
            print(str(exc), file=sys.stderr)
            return 2

    completed: List[Path] = []
    skipped_busy: List[str] = []
    skipped_incompatible: List[str] = []
    for number, prepared in enumerate(prepared_items, start=1):
        print(
            f"\n=== Logical job {number}/{len(prepared_items)}: "
            f"{prepared.item.output_stem} ===",
            flush=True,
        )
        try:
            # Recreate this per logical job so epoch/loss history from one
            # channel or TIFF can never leak into the next notification set.
            notifier = new_slack_notifier()
            completed.append(
                run_work_item(
                    prepared,
                    args,
                    training_base,
                    testing_base,
                    output_directory,
                    work_directory,
                    scratch_directory,
                    notifier,
                )
            )
        except WorkItemBusyError as exc:
            skipped_busy.append(prepared.item.output_stem)
            print(
                f"Skipping busy logical job {prepared.item.output_stem}: {exc}",
                flush=True,
            )
            continue
        except SavedRunMismatchError as exc:
            skipped_incompatible.append(prepared.item.output_stem)
            print(
                "Skipping logical job with incompatible saved checkpoints "
                f"{prepared.item.output_stem}: {exc}",
                flush=True,
            )
            continue
        except SlackConfigurationError as exc:
            print(f"Slack configuration changed during the batch: {exc}", file=sys.stderr)
            return 2
        except SingleFileConfigurationError as exc:
            print(f"Single-file run failed: {exc}", file=sys.stderr)
            return 2

    print(
        "Batch finished successfully: "
        f"{len(completed)} logical job(s) completed or already verified; "
        f"{len(skipped_busy)} skipped because another process is actively training them; "
        f"{len(skipped_incompatible)} skipped because saved checkpoints are incompatible."
    )
    for path in completed:
        print(path)
    if skipped_busy:
        print("Busy logical jobs skipped by this process:")
        for output_stem in skipped_busy:
            print(output_stem)
    if skipped_incompatible:
        print("Logical jobs skipped due to incompatible saved checkpoints:")
        for output_stem in skipped_incompatible:
            print(output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
