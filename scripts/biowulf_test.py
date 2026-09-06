#!/usr/bin/env python3
"""Concurrency-safe recursive SUPPORT inference for Biowulf batch jobs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile


REPO_DIRECTORY = Path(__file__).resolve().parent.parent
if str(REPO_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPO_DIRECTORY))

TIFF_SUFFIXES = {".tif", ".tiff"}
DEFAULT_CHANNEL_MAP = {1: "leukocyte", 2: "plasma", 3: "calcium"}
DIMENSIONALITY_MAP = {"4D": "3+1D", "3D": "3D", "2D": "2+1D"}
EXPECTED_NDIM = {"4D": 4, "3D": 3, "2D": 3}
DIMENSION_TOKEN = re.compile(r"(?<![A-Za-z0-9])([234])D(?![A-Za-z0-9])", re.IGNORECASE)
CHANNEL_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])ch(?:annel)?[_ -]?0*(\d+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
INFO_DIRECTORY = re.compile(r"^info(?:$|[_ .-])", re.IGNORECASE)

CSV_FIELDS = [
    "source_file",
    "relative_path",
    "output_file",
    "log_file",
    "channel",
    "type",
    "data_dims",
    "support_dimensionality",
    "model_path",
    "model_epoch",
    "model_training_date",
    "input_shape",
    "input_dtype",
    "status",
    "attempt",
    "begun_at_utc",
    "heartbeat_at_utc",
    "output_verified_at_utc",
    "source_deleted_at_utc",
    "finished_at_utc",
    "slurm_job_id",
    "hostname",
    "pid",
    "claim_id",
    "output_existed_at_claim",
    "source_device",
    "source_inode",
    "source_size",
    "source_mtime_ns",
    "error",
]


class LostClaimError(RuntimeError):
    """Raised when another worker owns a reclaimed file."""


class SourceChangedError(RuntimeError):
    """Raised when a source TIFF changes after inference begins."""


class ShutdownRequested(KeyboardInterrupt):
    """Raised when Slurm or the user requests a graceful stop."""

    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class SourceFingerprint:
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ModelChoice:
    path: Path
    epoch: int
    training_date: str


@dataclass
class FilePlan:
    source: Path
    relative_path: Path
    output: Path
    log_path: Path
    channel_number: Optional[int] = None
    data_type: str = ""
    data_dims: str = ""
    support_dimensionality: str = ""
    model: Optional[ModelChoice] = None
    input_shape: Tuple[int, ...] = ()
    input_dtype: str = ""
    error: str = ""


@dataclass(frozen=True)
class Claim:
    plan: FilePlan
    claim_id: str
    fingerprint: SourceFingerprint
    previous_status: str
    recover_existing_output: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def natural_key(value: object) -> Tuple[object, ...]:
    text = str(value)
    parts = re.split(r"(\d+)", text.casefold())
    numeric_key = tuple(int(part) if part.isdigit() else part for part in parts)
    return numeric_key, text.casefold(), text


def shape_to_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(item)) for item in shape)


def text_to_shape(value: str) -> Tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(item) for item in value.split("x"))


def plan_from_progress_row(fallback: FilePlan, row: Dict[str, str]) -> FilePlan:
    """Rebuild the plan that produced a recoverable output without relabeling it."""
    channel_number = fallback.channel_number
    channel_match = re.fullmatch(r"(?i)Ch(\d+)", row.get("channel", ""))
    if channel_match is not None:
        channel_number = int(channel_match.group(1))

    recorded_shape = fallback.input_shape
    try:
        if row.get("input_shape"):
            recorded_shape = text_to_shape(row["input_shape"])
    except ValueError:
        pass

    model: Optional[ModelChoice] = None
    if row.get("model_path"):
        try:
            epoch = int(row.get("model_epoch") or 0)
        except ValueError:
            epoch = 0
        model = ModelChoice(
            path=Path(row["model_path"]),
            epoch=epoch,
            training_date=row.get("model_training_date", ""),
        )

    return FilePlan(
        source=fallback.source,
        relative_path=Path(row.get("relative_path") or fallback.relative_path),
        output=Path(row.get("output_file") or fallback.output),
        log_path=Path(row.get("log_file") or fallback.log_path),
        channel_number=channel_number,
        data_type=row.get("type") or fallback.data_type,
        data_dims=row.get("data_dims") or fallback.data_dims,
        support_dimensionality=(
            row.get("support_dimensionality") or fallback.support_dimensionality
        ),
        model=model,
        input_shape=recorded_shape,
        input_dtype=row.get("input_dtype") or fallback.input_dtype,
    )


def as_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, received {value!r}")


def parse_channel_map(values: Optional[Sequence[str]]) -> Dict[int, str]:
    if not values:
        return dict(DEFAULT_CHANNEL_MAP)

    result: Dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"channel mapping {value!r} must use CHANNEL=TYPE, for example Ch3=calcium"
            )
        channel_text, data_type = (part.strip() for part in value.split("=", 1))
        match = re.fullmatch(r"(?i)(?:ch(?:annel)?)?0*(\d+)", channel_text)
        if match is None or int(match.group(1)) < 1:
            raise argparse.ArgumentTypeError(f"invalid channel in mapping {value!r}")
        if not data_type:
            raise argparse.ArgumentTypeError(f"missing type in mapping {value!r}")
        channel_number = int(match.group(1))
        if channel_number in result:
            raise argparse.ArgumentTypeError(f"channel Ch{channel_number} was mapped more than once")
        result[channel_number] = data_type
    return result


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be a finite number greater than zero")
    return number


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def ensure_separate_trees(test_directory: Path, output_directory: Path) -> None:
    if path_is_within(output_directory, test_directory) or path_is_within(
        test_directory, output_directory
    ):
        raise ValueError(
            "test_directory and output_directory must not overlap; recursive discovery and "
            "source deletion would otherwise be unsafe"
        )


def fingerprint(path: Path) -> SourceFingerprint:
    stat_result = path.stat()
    return SourceFingerprint(
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
    )


def fingerprint_from_row(row: Dict[str, str]) -> Optional[SourceFingerprint]:
    try:
        return SourceFingerprint(
            device=int(row["source_device"]),
            inode=int(row["source_inode"]),
            size=int(row["source_size"]),
            mtime_ns=int(row["source_mtime_ns"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def raise_walk_error(error: OSError) -> None:
    raise error


def is_info_directory(name: str) -> bool:
    return INFO_DIRECTORY.match(name) is not None


def discover_tiffs(test_directory: Path) -> List[Path]:
    paths: List[Path] = []
    for directory, directory_names, file_names in os.walk(
        test_directory, followlinks=False, onerror=raise_walk_error
    ):
        directory_names[:] = sorted(
            (
                name
                for name in directory_names
                if not is_info_directory(name)
                and name != ".support_denoising"
            ),
            key=natural_key,
        )
        current_directory = Path(directory)
        for name in sorted(file_names, key=natural_key):
            path = current_directory / name
            if path.suffix.casefold() not in TIFF_SUFFIXES:
                continue
            if path.stem.casefold().endswith("_sprt"):
                continue
            if path.is_symlink():
                print(f"Skipping symbolic-link TIFF for deletion safety: {path}", file=sys.stderr)
                continue
            paths.append(path)
    return sorted(
        paths,
        key=lambda path: natural_key(path.relative_to(test_directory).as_posix()),
    )


def find_channel(relative_path: Path) -> int:
    matches = {int(match.group(1)) for match in CHANNEL_TOKEN.finditer(relative_path.as_posix())}
    if not matches:
        raise ValueError("no Ch# token was found in the relative path")
    if len(matches) != 1:
        labels = ", ".join(f"Ch{number}" for number in sorted(matches))
        raise ValueError(f"conflicting channel tokens were found: {labels}")
    return next(iter(matches))


def find_data_dims(relative_path: Path, override: Optional[str]) -> str:
    if override:
        return override.upper()
    matches = {f"{match.group(1)}D".upper() for match in DIMENSION_TOKEN.finditer(relative_path.as_posix())}
    if not matches:
        raise ValueError("no _2D, _3D, or _4D token was found in the relative path")
    if len(matches) != 1:
        raise ValueError(f"conflicting dimensionality tokens were found: {', '.join(sorted(matches))}")
    return next(iter(matches))


def inspect_input_tiff(path: Path) -> Tuple[Tuple[int, ...], str]:
    if path.stat().st_size <= 0:
        raise ValueError("TIFF is empty")
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError("TIFF contains no image series")
        series = tif.series[0]
        shape = tuple(int(item) for item in series.shape)
        dtype = np.dtype(series.dtype).name
        if not series.pages:
            raise ValueError("TIFF contains no decodable pages")
    return shape, dtype


def verify_output_tiff(
    path: Path,
    expected_shape: Sequence[int],
    decode_all_pages: bool = True,
) -> Tuple[Tuple[int, ...], str]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"output TIFF is missing or empty: {path}")
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"output TIFF contains no image series: {path}")
        series = tif.series[0]
        shape = tuple(int(item) for item in series.shape)
        dtype = np.dtype(series.dtype).name
        if shape != tuple(expected_shape):
            raise ValueError(
                f"output shape {shape} does not match input shape {tuple(expected_shape)}"
            )
        if dtype != "float32":
            raise ValueError(f"output dtype {dtype} is not float32")
        if not series.pages:
            raise ValueError("output TIFF contains no pages")
        page_indices = range(len(series.pages)) if decode_all_pages else ()
        for page_index in page_indices:
            page = np.asarray(series.pages[page_index].asarray())
            if page.size == 0 or not np.isfinite(page).all():
                raise ValueError(f"output page {page_index} is empty or contains non-finite values")
    return shape, dtype


def choose_model(model_directory: Path, data_type: str, data_dims: str) -> ModelChoice:
    pattern = re.compile(
        rf"^{re.escape(data_type)}.*?"
        rf"(?<![A-Za-z0-9]){re.escape(data_dims)}(?![A-Za-z0-9]).*?"
        rf"epoch(?P<epoch>\d{{3}})(?!\d).*?"
        rf"(?P<date>\d{{8}})(?!\d).*\.pth$",
        re.IGNORECASE,
    )
    candidates: List[ModelChoice] = []
    for path in model_directory.rglob("*"):
        if not path.is_file() or path.suffix.casefold() != ".pth":
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        training_date = match.group("date")
        try:
            datetime.strptime(training_date, "%Y%m%d")
        except ValueError:
            continue
        candidates.append(
            ModelChoice(
                path=path.resolve(),
                epoch=int(match.group("epoch")),
                training_date=training_date,
            )
        )
    if not candidates:
        raise FileNotFoundError(
            f"no model matched type={data_type!r}, dims={data_dims}; expected a name like "
            f"{data_type}_{data_dims}_epoch013_20260826.pth"
        )
    return max(
        candidates,
        key=lambda model: (model.training_date, model.epoch, natural_key(model.path.as_posix())),
    )


def plan_files(
    paths: Sequence[Path],
    test_directory: Path,
    output_directory: Path,
    model_directory: Path,
    channel_map: Dict[int, str],
    data_dims_override: Optional[str],
) -> List[FilePlan]:
    model_cache: Dict[Tuple[str, str], ModelChoice] = {}
    plans: List[FilePlan] = []
    for source in paths:
        relative_path = source.relative_to(test_directory)
        token_context = Path(test_directory.name) / relative_path
        output = output_directory / relative_path.parent / f"{source.stem}_SPRT.tif"
        digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:10]
        log_path = output.parent / "info" / f"{source.stem}_denoise_{digest}.log"
        plan = FilePlan(
            source=source,
            relative_path=relative_path,
            output=output,
            log_path=log_path,
        )
        try:
            plan.channel_number = find_channel(token_context)
            if plan.channel_number not in channel_map:
                raise ValueError(f"Ch{plan.channel_number} is not present in the channel map")
            plan.data_type = channel_map[plan.channel_number]
            plan.data_dims = find_data_dims(token_context, data_dims_override)
            plan.support_dimensionality = DIMENSIONALITY_MAP[plan.data_dims]
            plan.input_shape, plan.input_dtype = inspect_input_tiff(source)
            expected_ndim = EXPECTED_NDIM[plan.data_dims]
            if len(plan.input_shape) != expected_ndim:
                raise ValueError(
                    f"nominal {plan.data_dims} input must have {expected_ndim} array dimensions, "
                    f"but the TIFF shape is {plan.input_shape}"
                )
            cache_key = (plan.data_type.casefold(), plan.data_dims)
            if cache_key not in model_cache:
                model_cache[cache_key] = choose_model(
                    model_directory, plan.data_type, plan.data_dims
                )
            plan.model = model_cache[cache_key]
        except Exception as exc:
            plan.error = str(exc)
        plans.append(plan)

    collisions: Dict[Path, List[FilePlan]] = {}
    for plan in plans:
        collisions.setdefault(plan.output, []).append(plan)
    for output, matching_plans in collisions.items():
        if len(matching_plans) > 1:
            sources = ", ".join(plan.relative_path.as_posix() for plan in matching_plans)
            error = f"multiple inputs map to the same output {output}: {sources}"
            for plan in matching_plans:
                plan.error = error
    return plans


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _same_regular_file(first: Path, second: Path) -> bool:
    try:
        first_stat = first.stat()
        second_stat = second.stat()
    except OSError:
        return False
    if first_stat.st_size != second_stat.st_size:
        return False
    with first.open("rb") as first_handle, second.open("rb") as second_handle:
        while True:
            first_block = first_handle.read(1024 * 1024)
            second_block = second_handle.read(1024 * 1024)
            if first_block != second_block:
                return False
            if not first_block:
                return True


def collect_info_copies(test_directory: Path, output_directory: Path) -> List[Tuple[Path, Path]]:
    copies: Dict[Path, Path] = {}
    for directory, directory_names, _ in os.walk(
        test_directory, followlinks=False, onerror=raise_walk_error
    ):
        directory_names.sort(key=natural_key)
        current_directory = Path(directory)
        info_names = [name for name in directory_names if is_info_directory(name)]
        for info_name in info_names:
            source_info = current_directory / info_name
            if source_info.is_symlink():
                raise ValueError(f"refusing to copy symbolic-link metadata directory: {source_info}")
            relative_parent = current_directory.relative_to(test_directory)
            destination_info = output_directory / relative_parent / "info"
            for metadata_directory, nested_directories, file_names in os.walk(
                source_info, followlinks=False, onerror=raise_walk_error
            ):
                nested_directories[:] = sorted(nested_directories, key=natural_key)
                metadata_directory_path = Path(metadata_directory)
                nested_relative = metadata_directory_path.relative_to(source_info)
                for file_name in sorted(file_names, key=natural_key):
                    source_file = metadata_directory_path / file_name
                    if source_file.is_symlink() or not source_file.is_file():
                        raise ValueError(f"refusing to copy non-regular metadata file: {source_file}")
                    destination_file = destination_info / nested_relative / file_name
                    previous = copies.get(destination_file)
                    if previous is not None and not _same_regular_file(previous, source_file):
                        raise ValueError(
                            f"metadata collision: both {previous} and {source_file} map to "
                            f"{destination_file}"
                        )
                    copies[destination_file] = source_file
            directory_names.remove(info_name)
    return sorted(
        ((source, destination) for destination, source in copies.items()),
        key=lambda pair: natural_key(pair[1].as_posix()),
    )


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".copying", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def copy_info_files(copies: Iterable[Tuple[Path, Path]]) -> int:
    copied = 0
    for source, destination in copies:
        if destination.exists() and _same_regular_file(source, destination):
            continue
        atomic_copy(source, destination)
        copied += 1
    return copied


class ProgressTracker:
    """A CSV tracker protected by a stable advisory-lock sidecar file."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock_path = Path(f"{csv_path}.lock")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Materialize a header-only tracker even when discovery finds no TIFFs.
        with self._edit():
            pass

    @contextlib.contextmanager
    def _edit(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            rows = self._read_unlocked()
            try:
                yield rows
            except BaseException:
                raise
            else:
                self._write_unlocked(rows)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @contextlib.contextmanager
    def _read(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            try:
                yield self._read_unlocked()
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> Dict[str, Dict[str, str]]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return {}
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "source_file" not in reader.fieldnames:
                raise ValueError(
                    f"progress CSV {self.csv_path} does not contain a source_file column"
                )
            rows: Dict[str, Dict[str, str]] = {}
            for raw_row in reader:
                source_file = (raw_row.get("source_file") or "").strip()
                if not source_file:
                    continue
                if source_file in rows:
                    raise ValueError(f"duplicate source_file row in {self.csv_path}: {source_file}")
                rows[source_file] = {
                    field: raw_row.get(field, "") or "" for field in CSV_FIELDS
                }
            return rows

    def _write_unlocked(self, rows: Dict[str, Dict[str, str]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.csv_path.name}.", suffix=".tmp", dir=self.csv_path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for row in sorted(
                    rows.values(),
                    key=lambda item: natural_key(item.get("relative_path") or item["source_file"]),
                ):
                    writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.csv_path)
            _fsync_directory(self.csv_path.parent)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _plan_values(plan: FilePlan) -> Dict[str, str]:
        model = plan.model
        return {
            "source_file": str(plan.source),
            "relative_path": plan.relative_path.as_posix(),
            "output_file": str(plan.output),
            "log_file": str(plan.log_path),
            "channel": f"Ch{plan.channel_number}" if plan.channel_number is not None else "",
            "type": plan.data_type,
            "data_dims": plan.data_dims,
            "support_dimensionality": plan.support_dimensionality,
            "model_path": str(model.path) if model is not None else "",
            "model_epoch": str(model.epoch) if model is not None else "",
            "model_training_date": model.training_date if model is not None else "",
            "input_shape": shape_to_text(plan.input_shape),
            "input_dtype": plan.input_dtype,
        }

    @staticmethod
    def _owner_values() -> Dict[str, str]:
        return {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "hostname": socket.gethostname(),
            "pid": str(os.getpid()),
        }

    def record_skipped(self, plan: FilePlan) -> None:
        key = str(plan.source)
        with self._edit() as rows:
            existing = rows.get(key, {})
            if existing.get("status") in {
                "begun",
                "output_saved",
                "finished",
                "source_changed",
            }:
                return
            row = {field: existing.get(field, "") for field in CSV_FIELDS}
            row.update(self._plan_values(plan))
            row.update(self._owner_values())
            row.update({"status": "skipped", "error": plan.error})
            rows[key] = row

    def claim(
        self,
        plan: FilePlan,
        reclaim_after_hours: Optional[float],
        retry_source_changed: bool = False,
        overwrite_existing: bool = False,
    ) -> Optional[Claim]:
        key = str(plan.source)
        current_fingerprint = fingerprint(plan.source)
        claim_id = uuid.uuid4().hex
        result: Optional[Claim] = None
        with self._edit() as rows:
            existing = rows.get(key, {})
            previous_status = existing.get("status", "")
            if previous_status == "finished":
                return None
            if previous_status == "source_changed" and not retry_source_changed:
                return None
            has_active_owner = previous_status == "begun" or (
                previous_status == "output_saved" and bool(existing.get("claim_id"))
            )
            if has_active_owner:
                if reclaim_after_hours is None:
                    return None
                timestamp = existing.get("heartbeat_at_utc") or existing.get("begun_at_utc")
                try:
                    begun = datetime.fromisoformat(timestamp)
                    if begun.tzinfo is None:
                        begun = begun.replace(tzinfo=timezone.utc)
                    age_hours = (datetime.now(timezone.utc) - begun).total_seconds() / 3600
                except (TypeError, ValueError):
                    return None
                if age_hours < reclaim_after_hours:
                    return None

            old_fingerprint = fingerprint_from_row(existing)
            if previous_status in {"begun", "output_saved"} and old_fingerprint is not None:
                if old_fingerprint != current_fingerprint:
                    row = {field: existing.get(field, "") for field in CSV_FIELDS}
                    row.update(
                        {
                            "status": "source_changed",
                            "error": "source fingerprint changed before recovery; source was preserved",
                        }
                    )
                    rows[key] = row
                    return None

            recorded_output = Path(existing["output_file"]) if existing.get("output_file") else None
            recover_existing_output = (
                previous_status in {"begun", "output_saved"}
                and recorded_output is not None
                and recorded_output.is_file()
                and existing.get("output_existed_at_claim") == "false"
                and not overwrite_existing
            )
            work_plan = (
                plan_from_progress_row(plan, existing)
                if recover_existing_output
                else plan
            )

            if work_plan.output.exists() and not recover_existing_output and not overwrite_existing:
                row = {field: existing.get(field, "") for field in CSV_FIELDS}
                row.update(self._plan_values(plan))
                row.update(self._owner_values())
                row.update(
                    {
                        "status": "skipped",
                        "claim_id": "",
                        "error": (
                            f"output already exists without recorded provenance: "
                            f"{work_plan.output}; inspect it before using --overwrite-existing"
                        ),
                    }
                )
                rows[key] = row
                return None

            for other_source, other_row in rows.items():
                if other_source == key:
                    continue
                if other_row.get("output_file") == str(work_plan.output):
                    # A collision discovered during an earlier scan may no longer
                    # exist after the user removes or renames one of the inputs.
                    # Keep every stateful reservation, but do not let a stale
                    # skipped row permanently block the surviving source.
                    if (
                        other_row.get("status") == "skipped"
                        and not work_plan.output.exists()
                    ):
                        continue
                    row = {field: existing.get(field, "") for field in CSV_FIELDS}
                    row.update(self._plan_values(plan))
                    row.update(self._owner_values())
                    row.update(
                        {
                            "status": "skipped",
                            "claim_id": "",
                            "error": (
                                f"output path is already reserved by another source: "
                                f"{other_source}"
                            ),
                        }
                    )
                    rows[key] = row
                    return None

            now = utc_now()
            row = {field: existing.get(field, "") for field in CSV_FIELDS}
            if not recover_existing_output:
                row.update(self._plan_values(work_plan))
            elif not row.get("log_file"):
                row["log_file"] = str(work_plan.log_path)
            row.update(self._owner_values())
            row.update(
                {
                    "status": "output_saved" if recover_existing_output else "begun",
                    "attempt": str(int(existing.get("attempt") or 0) + 1),
                    "begun_at_utc": now,
                    "heartbeat_at_utc": now,
                    "finished_at_utc": "",
                    "source_deleted_at_utc": "",
                    "claim_id": claim_id,
                    "source_device": str(current_fingerprint.device),
                    "source_inode": str(current_fingerprint.inode),
                    "source_size": str(current_fingerprint.size),
                    "source_mtime_ns": str(current_fingerprint.mtime_ns),
                    "output_existed_at_claim": (
                        "false" if recover_existing_output else str(work_plan.output.exists()).lower()
                    ),
                    "error": "",
                }
            )
            if not recover_existing_output:
                row["output_verified_at_utc"] = ""
            rows[key] = row
            result = Claim(
                work_plan,
                claim_id,
                current_fingerprint,
                previous_status,
                recover_existing_output,
            )
        return result

    @staticmethod
    def _require_claim(row: Dict[str, str], claim: Claim) -> None:
        if row.get("claim_id") != claim.claim_id or row.get("status") not in {
            "begun",
            "output_saved",
        }:
            raise LostClaimError(f"claim is no longer owned for {claim.plan.source}")

    def heartbeat(self, claim: Claim) -> None:
        with self._edit() as rows:
            row = rows.get(str(claim.plan.source), {})
            self._require_claim(row, claim)
            row["heartbeat_at_utc"] = utc_now()

    def adopt_existing_output(self, claim: Claim) -> None:
        changed = False
        with self._edit() as rows:
            row = rows.get(str(claim.plan.source), {})
            self._require_claim(row, claim)
            if fingerprint(claim.plan.source) != claim.fingerprint:
                row.update(
                    {
                        "status": "source_changed",
                        "error": "source fingerprint changed before output recovery; source was preserved",
                    }
                )
                changed = True
            else:
                row.update(
                    {
                        "status": "output_saved",
                        "output_verified_at_utc": utc_now(),
                        "heartbeat_at_utc": utc_now(),
                        "output_existed_at_claim": "false",
                        "error": "",
                    }
                )
        if changed:
            raise SourceChangedError(f"source changed during recovery: {claim.plan.source}")

    def publish_stage(self, claim: Claim, stage: Path) -> None:
        changed = False
        with self._edit() as rows:
            row = rows.get(str(claim.plan.source), {})
            self._require_claim(row, claim)
            if fingerprint(claim.plan.source) != claim.fingerprint:
                row.update(
                    {
                        "status": "source_changed",
                        "error": "source fingerprint changed during inference; source was preserved",
                    }
                )
                changed = True
            else:
                claim.plan.output.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage, claim.plan.output)
                _fsync_directory(claim.plan.output.parent)
                row.update(
                    {
                        "status": "output_saved",
                        "output_verified_at_utc": utc_now(),
                        "heartbeat_at_utc": utc_now(),
                        "output_existed_at_claim": "false",
                        "error": "",
                    }
                )
        if changed:
            raise SourceChangedError(f"source changed during inference: {claim.plan.source}")

    def delete_source_and_finish(self, claim: Claim) -> None:
        changed = False
        with self._edit() as rows:
            row = rows.get(str(claim.plan.source), {})
            self._require_claim(row, claim)
            if claim.plan.source.exists():
                if fingerprint(claim.plan.source) != claim.fingerprint:
                    row.update(
                        {
                            "status": "source_changed",
                            "error": "source fingerprint changed before deletion; source was preserved",
                        }
                    )
                    changed = True
                else:
                    claim.plan.source.unlink()
                    _fsync_directory(claim.plan.source.parent)
            if not changed:
                now = utc_now()
                row.update(
                    {
                        "status": "finished",
                        "source_deleted_at_utc": now,
                        "finished_at_utc": now,
                        "heartbeat_at_utc": now,
                        "error": "",
                    }
                )
        if changed:
            raise SourceChangedError(f"source changed before deletion: {claim.plan.source}")

    def mark_failed(self, claim: Claim, error: str) -> None:
        with self._edit() as rows:
            row = rows.get(str(claim.plan.source), {})
            if row.get("claim_id") != claim.claim_id or row.get("status") == "finished":
                return
            status = row.get("status")
            if status not in {"output_saved", "source_changed"}:
                status = "failed"
            row.update({"status": status, "heartbeat_at_utc": utc_now(), "error": error})
            if status == "output_saved":
                # Inference is complete and verified, but this worker could not finish
                # cleanup. Release only the cleanup claim so another worker can verify
                # the final TIFF and delete the unchanged source without re-inference.
                row["claim_id"] = ""

    def reconcile_missing_sources(self) -> int:
        with self._read() as rows:
            candidates = [dict(row) for row in rows.values()]
        reconciled = 0
        for candidate in candidates:
            if candidate.get("status") not in {"begun", "output_saved"}:
                continue
            source = Path(candidate["source_file"])
            output = Path(candidate["output_file"])
            expected_shape = text_to_shape(candidate.get("input_shape", ""))
            if (
                source.exists()
                or not expected_shape
                or candidate.get("output_existed_at_claim") != "false"
            ):
                continue
            try:
                verify_output_tiff(output, expected_shape)
            except Exception:
                continue
            with self._edit() as rows:
                row = rows.get(candidate["source_file"], {})
                if row.get("status") not in {"begun", "output_saved"} or source.exists():
                    continue
                now = utc_now()
                row.update(
                    {
                        "status": "finished",
                        "source_deleted_at_utc": row.get("source_deleted_at_utc") or now,
                        "finished_at_utc": now,
                        "heartbeat_at_utc": now,
                        "error": "reconciled after output verification; source was already absent",
                    }
                )
                reconciled += 1
        return reconciled

    def status_counts(self) -> Dict[str, int]:
        with self._read() as rows:
            counts: Dict[str, int] = {}
            for row in rows.values():
                status = row.get("status") or "unknown"
                counts[status] = counts.get(status, 0) + 1
            return counts


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def log_line(handle, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def load_parameters(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)
    for dimensionality in DIMENSIONALITY_MAP.values():
        if dimensionality not in parameters:
            raise KeyError(f"{path} does not contain a {dimensionality!r} parameter section")
    return parameters


def build_test_parameters(
    plan: FilePlan,
    parameters: Dict[str, object],
    stage: Path,
    use_cpu: bool,
    batch_size: Optional[int],
) -> Dict[str, object]:
    section = parameters[plan.support_dimensionality]
    if not isinstance(section, dict):
        raise TypeError(f"parameter section {plan.support_dimensionality!r} must be an object")
    train = section.get("train")
    test = section.get("test")
    if not isinstance(train, dict) or not isinstance(test, dict):
        raise TypeError(f"parameter section {plan.support_dimensionality!r} needs train and test objects")
    if plan.model is None:
        raise RuntimeError("file plan has no selected model")

    result = {
        "in_path": str(plan.source),
        "model_path": str(plan.model.path),
        "exp_name": plan.source.stem,
        "out_dir": str(plan.output.parent),
        "output_path": str(stage),
        "dimensionality": plan.support_dimensionality,
        "patch_size": train["patch_size"],
        "patch_interval": test["patch_interval"],
        "batch_size": batch_size if batch_size is not None else test["batch_size"],
        "pin_memory": as_bool(test.get("pin_memory", True), "pin_memory"),
        "depth": train["depth"],
        "bc_channels": train["bc_channels"],
        "obo_channels": train["obo_channels"],
        "ll_channels": train["ll_channels"],
        "unet_channels": train["unet_channels"],
        "bs_size": train["bs_size"],
        "bp": as_bool(train.get("bp", False), "bp"),
        "prefetch_factor": test.get("prefetch_factor", 2),
        "pad": as_bool(test.get("pad", True), "pad"),
        "use_cpu": use_cpu,
        "track_loss": False,
    }
    return result


def run_support_inference(test_parameters: Dict[str, object]) -> None:
    from utils.test_collection import TestingClass

    tester = TestingClass(test_parameters)
    try:
        tester.run()
    finally:
        tester.release_resources()
        del tester
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _process_claim(
    claim: Claim,
    tracker: ProgressTracker,
    parameters: Dict[str, object],
    use_cpu: bool,
    batch_size: Optional[int],
    overwrite_existing: bool,
    inference_function: Callable[[Dict[str, object]], None] = run_support_inference,
) -> str:
    plan = claim.plan
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    plan.log_path.parent.mkdir(parents=True, exist_ok=True)
    stage = plan.output.parent / (
        f".{plan.output.stem}.{claim.claim_id}.partial{plan.output.suffix}"
    )
    recovered = False
    publish_attempted = False
    with plan.log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_line(
            log_handle,
            f"BEGIN source={plan.source} model={plan.model.path if plan.model else ''} "
            f"claim={claim.claim_id}",
        )
        try:
            current_shape, _ = inspect_input_tiff(plan.source)
            if current_shape != plan.input_shape:
                raise SourceChangedError(
                    f"source shape changed from {plan.input_shape} to {current_shape} before inference"
                )

            if plan.output.exists():
                if claim.recover_existing_output:
                    verify_output_tiff(plan.output, plan.input_shape)
                    tracker.adopt_existing_output(claim)
                    recovered = True
                    log_line(log_handle, f"RECOVERED verified existing output={plan.output}")
                elif not overwrite_existing:
                    raise FileExistsError(
                        f"output already exists without a recoverable progress record: {plan.output}; "
                        "use --overwrite-existing only after inspecting it"
                    )

            if not recovered:
                test_parameters = build_test_parameters(
                    plan, parameters, stage, use_cpu, batch_size
                )
                tracker.heartbeat(claim)
                stdout_tee = Tee(sys.stdout, log_handle)
                stderr_tee = Tee(sys.stderr, log_handle)
                with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                    inference_function(test_parameters)
                verify_output_tiff(stage, plan.input_shape)
                _fsync_file(stage)
                publish_attempted = True
                tracker.publish_stage(claim, stage)
                log_line(log_handle, f"SAVED verified output={plan.output}")

            # The fully decoded and fsynced stage was atomically renamed to this
            # path. Reopen the final path and recheck its TIFF structure without
            # reading the same multi-gigabyte payload a second time.
            verify_output_tiff(plan.output, plan.input_shape, decode_all_pages=False)
            _fsync_file(plan.output)
            tracker.delete_source_and_finish(claim)
            log_line(log_handle, f"FINISHED deleted_source={plan.source}")
            return "finished"
        except ShutdownRequested as exc:
            error = str(exc)
            if publish_attempted and not stage.exists() and plan.output.exists():
                try:
                    verify_output_tiff(plan.output, plan.input_shape)
                    tracker.adopt_existing_output(claim)
                except BaseException:
                    pass
            tracker.mark_failed(claim, error)
            log_line(log_handle, f"INTERRUPTED source preserved: {error}")
            raise
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if publish_attempted and not stage.exists() and plan.output.exists():
                try:
                    verify_output_tiff(plan.output, plan.input_shape)
                    tracker.adopt_existing_output(claim)
                except BaseException:
                    pass
            tracker.mark_failed(claim, error)
            log_line(log_handle, f"FAILED source preserved: {error}")
            traceback.print_exc(file=log_handle)
            return "failed"
        finally:
            if stage.exists():
                stage.unlink()


def process_claim(
    claim: Claim,
    tracker: ProgressTracker,
    parameters: Dict[str, object],
    use_cpu: bool,
    batch_size: Optional[int],
    overwrite_existing: bool,
    inference_function: Callable[[Dict[str, object]], None] = run_support_inference,
) -> str:
    """Run one claimed file and release recoverably if setup itself fails."""
    try:
        return _process_claim(
            claim,
            tracker,
            parameters,
            use_cpu,
            batch_size,
            overwrite_existing,
            inference_function,
        )
    except ShutdownRequested as exc:
        tracker.mark_failed(claim, str(exc))
        raise
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        tracker.mark_failed(claim, error)
        print(
            f"FAILED before per-file logging was available; source preserved: "
            f"{claim.plan.source}: {error}",
            file=sys.stderr,
        )
        return "failed"


def format_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def print_dry_run(
    plans: Sequence[FilePlan],
    info_copies: Sequence[Tuple[Path, Path]],
    channel_map: Dict[int, str],
) -> None:
    print("Dry run: no CSV, outputs, metadata, or source files were changed.")
    print(
        "Channel map: "
        + ", ".join(f"Ch{channel}={data_type}" for channel, data_type in sorted(channel_map.items()))
    )
    print(f"Metadata files to copy: {len(info_copies)}")
    for plan in plans:
        if plan.error:
            print(f"SKIP  {plan.relative_path.as_posix()} :: {plan.error}")
        else:
            print(
                f"READY {plan.relative_path.as_posix()} :: Ch{plan.channel_number}="
                f"{plan.data_type}, {plan.data_dims}, model={plan.model.path.name}, "
                f"output={plan.output}"
            )


def build_parser(repo_directory: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively denoise channel-separated TIFFs with shared CSV claims, mirrored "
            "outputs, verified saves, and post-save source deletion."
        )
    )
    parser.add_argument(
        "--test-directory",
        "--test-dir",
        required=True,
        type=Path,
        help="root directory searched recursively for .tif and .tiff files",
    )
    parser.add_argument(
        "--channel-map",
        action="append",
        metavar="CHANNEL=TYPE",
        help=(
            "repeatable channel/type mapping; supplying any replaces the defaults "
            "Ch3=calcium, Ch2=plasma, Ch1=leukocyte"
        ),
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=repo_directory / "pretrained_models",
        help="directory searched recursively for pretrained .pth files",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repo_directory / "notebooks" / "outputs",
        help="root of the mirrored output tree",
    )
    parser.add_argument(
        "--progress-csv",
        type=Path,
        default=None,
        help="shared progress CSV (default: OUTPUT_DIRECTORY/denoising_progress.csv)",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=repo_directory / "notebooks" / "gen_parameters.json",
        help="read-only SUPPORT architecture/test parameter file",
    )
    parser.add_argument(
        "--data-dims",
        choices=("2D", "3D", "4D", "2d", "3d", "4d"),
        default=None,
        help="force one nominal dimension for every TIFF instead of parsing _2D/_3D/_4D",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="override the selected test batch size"
    )
    parser.add_argument(
        "--use-cpu", action="store_true", help="run without CUDA (intended only for debugging)"
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="replace a preexisting output that has no recoverable CSV record",
    )
    parser.add_argument(
        "--reclaim-after-hours",
        type=positive_float,
        default=None,
        help=(
            "reclaim a begun or actively-owned output_saved row older than this many hours; "
            "use only after confirming its Slurm job is no longer active"
        ),
    )
    parser.add_argument(
        "--retry-source-changed",
        action="store_true",
        help="establish a new fingerprint and retry rows marked source_changed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show discovery, mappings, models, and outputs without changing any files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_directory = REPO_DIRECTORY
    parser = build_parser(repo_directory)
    args = parser.parse_args(argv)
    try:
        channel_map = parse_channel_map(args.channel_map)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    try:
        test_directory = args.test_directory.expanduser().resolve(strict=True)
        if not test_directory.is_dir():
            raise NotADirectoryError(f"test directory is not a directory: {test_directory}")
        model_directory = args.model_directory.expanduser().resolve(strict=True)
        if not model_directory.is_dir():
            raise NotADirectoryError(f"model directory is not a directory: {model_directory}")
        parameter_path = args.parameters.expanduser().resolve(strict=True)
        output_directory = args.output_directory.expanduser().resolve(strict=False)
        ensure_separate_trees(test_directory, output_directory)
        progress_csv = (
            args.progress_csv.expanduser().resolve(strict=False)
            if args.progress_csv is not None
            else output_directory / "denoising_progress.csv"
        )
        paths = discover_tiffs(test_directory)
        plans = plan_files(
            paths,
            test_directory,
            output_directory,
            model_directory,
            channel_map,
            args.data_dims.upper() if args.data_dims else None,
        )
        info_copies = collect_info_copies(test_directory, output_directory)
        parameters = load_parameters(parameter_path)
    except Exception as exc:
        print(f"Configuration error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print_dry_run(plans, info_copies, channel_map)
        return 0

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        copied_count = copy_info_files(info_copies)
        print(f"Copied or refreshed {copied_count} metadata file(s).")
        tracker = ProgressTracker(progress_csv)
        reconciled = tracker.reconcile_missing_sources()
        if reconciled:
            print(f"Reconciled {reconciled} completed row(s) whose source was already absent.")
    except Exception as exc:
        print(f"Initialization error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    def handle_signal(signum, _frame):
        raise ShutdownRequested(signum)

    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    local_counts = {"finished": 0, "failed": 0, "skipped": 0, "busy_or_done": 0}
    active_claim: Optional[Claim] = None
    try:
        for plan in plans:
            if plan.error:
                tracker.record_skipped(plan)
                local_counts["skipped"] += 1
                print(f"SKIPPED {plan.relative_path}: {plan.error}", file=sys.stderr)
                continue
            try:
                claim = tracker.claim(
                    plan,
                    args.reclaim_after_hours,
                    retry_source_changed=args.retry_source_changed,
                    overwrite_existing=args.overwrite_existing,
                )
            except FileNotFoundError:
                local_counts["busy_or_done"] += 1
                continue
            if claim is None:
                local_counts["busy_or_done"] += 1
                continue
            active_claim = claim
            result = process_claim(
                claim,
                tracker,
                parameters,
                args.use_cpu,
                args.batch_size,
                args.overwrite_existing,
            )
            active_claim = None
            local_counts[result] += 1
    except ShutdownRequested as exc:
        if active_claim is not None:
            tracker.mark_failed(active_claim, str(exc))
        print(f"Stopping after {exc}; the active source was preserved.", file=sys.stderr)
        return 128 + exc.signum
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)

    print(f"This worker: {format_counts(local_counts)}")
    print(f"Shared CSV: {progress_csv}")
    try:
        print(f"Shared statuses: {format_counts(tracker.status_counts())}")
    except Exception as exc:
        print(f"Could not summarize progress CSV: {exc}", file=sys.stderr)
        return 2
    return 1 if local_counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
