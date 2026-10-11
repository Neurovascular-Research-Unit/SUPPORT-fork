"""Focused tests for the per-dataset Biowulf input planner."""

from __future__ import annotations

import copy
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

from scripts import biowulf_singlefile as singlefile
from scripts.biowulf_singlefile import (
    PreparedWorkItem,
    SavedRunMismatchError,
    SingleFileConfigurationError,
    WorkItem,
    WorkItemBusyError,
    completed_output_is_current,
    concatenate_ome_chunks,
    discover_work_items,
    exclusive_item_lock,
    latest_complete_checkpoint_epoch,
    prepare_work_items,
    run_work_item,
    validate_or_create_manifest,
    write_completion_record,
)


class BiowulfSingleFileDiscoveryTests(unittest.TestCase):
    @staticmethod
    def _write_tiff(path: Path, value: int = 0) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.full((3, 4, 5), value, dtype=np.uint16)
        tifffile.imwrite(path, data, metadata={"axes": "TYX"})
        return path

    def test_ome_chunks_are_grouped_by_channel_ordered_and_named_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "data_noisy"
            output_directory = root / "clean"
            dataset = (
                input_root
                / "C57_293_ChronicCrani"
                / "2026_0805_BaselineImaging1"
                / "FOV2_250Deep_3X_2D"
            )
            acquisition = "FOV2_250Deep_3X_2DT-001"

            # Create these out of order so discovery, rather than directory
            # insertion order, establishes the temporal sequence.
            for channel in (3, 2):
                for index in (3, 1, 2):
                    self._write_tiff(
                        dataset
                        / (
                            f"{acquisition}_Cycle00001_Ch{channel}_"
                            f"{index:06d}.ome.tif"
                        ),
                        value=channel * 10 + index,
                    )

            # These are TIFFs below a path containing 2DT, but they are
            # acquisition reference images rather than temporal chunks.
            self._write_tiff(
                dataset
                / "References"
                / f"{acquisition}-Ch2-16bit-Reference.tif"
            )
            self._write_tiff(
                dataset
                / "References"
                / f"{acquisition}-Ch3-16bit-Reference.tif"
            )

            # A valid-looking pair outside any path containing the marker is
            # not part of this batch.
            self._write_tiff(input_root / "unrelated" / "first.tif")
            self._write_tiff(input_root / "unrelated" / "second.tif")
            output_directory.mkdir()

            items = discover_work_items(input_root, output_directory)

            self.assertEqual(len(items), 2)
            by_channel = {item.channel_number: item for item in items}
            self.assertEqual(set(by_channel), {2, 3})

            common_stem = (
                "C57_293_ChronicCrani_2026_0805_BaselineImaging1_"
                "FOV2_250Deep_3X_2D_FOV2_250Deep_3X_2DT"
            )
            for channel in (2, 3):
                item = by_channel[channel]
                self.assertIn("ome", str(item.scenario).casefold())
                self.assertEqual(item.output_stem, f"{common_stem}_Ch{channel}")
                self.assertEqual(
                    [path.name for path in item.source_paths],
                    [
                        f"{acquisition}_Cycle00001_Ch{channel}_000001.ome.tif",
                        f"{acquisition}_Cycle00001_Ch{channel}_000002.ome.tif",
                        f"{acquisition}_Cycle00001_Ch{channel}_000003.ome.tif",
                    ],
                )
                self.assertTrue(
                    all(path.parent == dataset.resolve() for path in item.source_paths)
                )

    def test_any_positive_plain_tiff_count_creates_independent_work_items(self) -> None:
        for file_count in (1, 2, 3):
            with self.subTest(file_count=file_count):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    input_root = root / "data_noisy"
                    output_directory = root / "clean"
                    dataset = input_root / "animal" / "session_2DT-004"
                    sources = {
                        self._write_tiff(
                            dataset / f"already_named_Ch{index}.tif", value=index
                        ).resolve()
                        for index in range(1, file_count + 1)
                    }
                    output_directory.mkdir()

                    items = discover_work_items(input_root, output_directory)

                    self.assertEqual(len(items), file_count)
                    self.assertEqual(
                        {item.source_paths[0] for item in items}, sources
                    )
                    self.assertEqual(
                        {item.output_stem for item in items},
                        {
                            f"already_named_Ch{index}"
                            for index in range(1, file_count + 1)
                        },
                    )
                    for item in items:
                        self.assertEqual(len(item.source_paths), 1)
                        self.assertIn("single", str(item.scenario).casefold())

    def test_marker_may_appear_only_in_a_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "input"
            output_directory = root / "output"
            dataset = input_root / "nested" / "contains_2DT_marker"
            self._write_tiff(dataset / "first.tif")
            self._write_tiff(dataset / "second.tif")
            output_directory.mkdir()

            items = discover_work_items(input_root, output_directory)

            self.assertEqual(len(items), 2)
            self.assertEqual(
                {item.source_paths[0].name for item in items},
                {"first.tif", "second.tif"},
            )

    def test_ome_chunk_order_uses_numeric_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "input"
            output_directory = root / "output"
            dataset = input_root / "numeric_2DT"
            for channel in (2, 3):
                for index in (10, 2, 1):
                    self._write_tiff(
                        dataset
                        / (
                            "numeric_2DT-001_Cycle00001_"
                            f"Ch{channel}_{index}.ome.tif"
                        )
                    )
            output_directory.mkdir()

            items = discover_work_items(input_root, output_directory)

            self.assertEqual(len(items), 2)
            for item in items:
                self.assertEqual(
                    [
                        int(path.name.rsplit("_", 1)[1].split(".", 1)[0])
                        for path in item.source_paths
                    ],
                    [1, 2, 10],
                )

    def test_mixed_or_malformed_dataset_layouts_are_rejected(self) -> None:
        cases = {
            "mixed plain and OME TIFFs": (
                "scan_2DT",
                [
                    "plain.tif",
                    "scan_2DT-001_Cycle00001_Ch2_000001.ome.tif",
                ],
            ),
            "OME TIFF without a channel token": (
                "scan_2DT",
                [
                    "scan_2DT-001_Cycle00001_000001.ome.tif",
                    "scan_2DT-001_Cycle00001_000002.ome.tif",
                ],
            ),
        }

        for label, (directory_name, filenames) in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    input_root = root / "input"
                    dataset = input_root / directory_name
                    output_directory = root / "output"
                    for filename in filenames:
                        self._write_tiff(dataset / filename)
                    output_directory.mkdir()

                    with self.assertRaises(ValueError):
                        discover_work_items(input_root, output_directory)

    def test_flattened_ome_output_name_collisions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "input"
            output_directory = root / "output"
            # Both relative directory paths flatten to ``a_b_c_2DT``.
            datasets = (
                input_root / "a_b" / "c_2DT",
                input_root / "a" / "b_c_2DT",
            )
            for dataset in datasets:
                for channel in (2, 3):
                    for index in (1, 2):
                        self._write_tiff(
                            dataset
                            / (
                                "scan_2DT-001_Cycle00001_"
                                f"Ch{channel}_{index:06d}.ome.tif"
                            )
                        )
            output_directory.mkdir()

            with self.assertRaises(ValueError):
                discover_work_items(input_root, output_directory)


class OmeChunkConcatenationTests(unittest.TestCase):
    def test_chunks_are_concatenated_in_planned_temporal_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks = []
            arrays = []
            for index, frame_count in ((1, 2), (2, 3), (10, 4)):
                path = root / (
                    "scan_2DT-001_Cycle00001_Ch2_"
                    f"{index:06d}.ome.tif"
                )
                array = np.full(
                    (frame_count, 4, 5),
                    index,
                    dtype=np.uint16,
                )
                tifffile.imwrite(path, array, metadata={"axes": "TYX"})
                chunks.append(path)
                arrays.append(array)

            destination = root / "combined.tif"
            result = concatenate_ome_chunks(chunks, destination)

            self.assertEqual(result, destination)
            self.assertTrue(destination.is_file())
            with tifffile.TiffFile(destination) as tif:
                self.assertTrue(tif.is_bigtiff)
                self.assertEqual(tif.series[0].axes, "TYX")
                self.assertEqual(tif.series[0].dtype, np.dtype(np.uint16))
                actual = tif.asarray()
            np.testing.assert_array_equal(actual, np.concatenate(arrays, axis=0))

            # Staging a combined stack must not mutate or remove its sources.
            for path, expected in zip(chunks, arrays):
                self.assertTrue(path.is_file())
                np.testing.assert_array_equal(tifffile.imread(path), expected)

    def test_incompatible_chunk_shapes_or_dtypes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            good = root / "scan_2DT-001_Cycle00001_Ch2_000001.ome.tif"
            wrong_shape = root / "scan_2DT-001_Cycle00001_Ch2_000002.ome.tif"
            wrong_dtype = root / "scan_2DT-001_Cycle00001_Ch2_000003.ome.tif"
            tifffile.imwrite(
                good,
                np.zeros((2, 4, 5), dtype=np.uint16),
                metadata={"axes": "TYX"},
            )
            tifffile.imwrite(
                wrong_shape,
                np.zeros((2, 4, 6), dtype=np.uint16),
                metadata={"axes": "TYX"},
            )
            tifffile.imwrite(
                wrong_dtype,
                np.zeros((2, 4, 5), dtype=np.float32),
                metadata={"axes": "TYX"},
            )

            for label, paths in (
                ("shape", [good, wrong_shape]),
                ("dtype", [good, wrong_dtype]),
            ):
                with self.subTest(label=label):
                    destination = root / f"combined_{label}.tif"
                    with self.assertRaises(ValueError):
                        concatenate_ome_chunks(paths, destination)
                    self.assertFalse(destination.exists())


class CompletionRecordTests(unittest.TestCase):
    def test_verified_completion_can_skip_and_detects_replaced_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = root / "work" / "scan"
            checkpoint_directory = experiment / "pth"
            checkpoint_directory.mkdir(parents=True)
            model = checkpoint_directory / "model_2.pth"
            optimizer = checkpoint_directory / "optimizer_2.pth"
            model.write_bytes(b"model")
            optimizer.write_bytes(b"optimizer")
            output = root / "scan_SPRT.tif"
            data = np.zeros((3, 4, 5), dtype=np.float32)
            tifffile.imwrite(
                output,
                data,
                imagej=True,
                metadata={"axes": "TYX"},
            )
            signature = {"version": 1, "output_stem": "scan"}

            self.assertEqual(latest_complete_checkpoint_epoch(experiment, False), 2)
            write_completion_record(
                experiment,
                signature,
                completed_epochs=3,
                model_path=model,
                output_path=output,
            )
            self.assertTrue(
                completed_output_is_current(
                    experiment,
                    signature,
                    requested_epochs=3,
                    output_path=output,
                    expected_shape=data.shape,
                )
            )

            optimizer.unlink()
            self.assertFalse(
                completed_output_is_current(
                    experiment,
                    signature,
                    requested_epochs=3,
                    output_path=output,
                    expected_shape=data.shape,
                )
            )
            optimizer.write_bytes(b"optimizer")

            # Atomic output replacement changes its recorded identity and must
            # force inference instead of trusting a stale completion marker.
            replacement = root / "replacement.tif"
            tifffile.imwrite(
                replacement,
                data + 1,
                imagej=True,
                metadata={"axes": "TYX"},
            )
            replacement.replace(output)
            self.assertFalse(
                completed_output_is_current(
                    experiment,
                    signature,
                    requested_epochs=3,
                    output_path=output,
                    expected_shape=data.shape,
                )
            )

    def test_incomplete_checkpoint_bundle_is_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment = Path(temporary_directory) / "experiment"
            checkpoint_directory = experiment / "pth"
            checkpoint_directory.mkdir(parents=True)
            (checkpoint_directory / "model_4.pth").write_bytes(b"model")
            self.assertIsNone(latest_complete_checkpoint_epoch(experiment, False))

            (checkpoint_directory / "optimizer_4.pth").write_bytes(b"optimizer")
            (checkpoint_directory / ".checkpoint_4.incomplete").touch()
            self.assertIsNone(latest_complete_checkpoint_epoch(experiment, False))

            (checkpoint_directory / ".checkpoint_4.incomplete").unlink()
            self.assertEqual(latest_complete_checkpoint_epoch(experiment, False), 4)

            # Legacy checkpoint numbers are one-based completed-epoch counts.
            (checkpoint_directory / "model007.pth").write_bytes(b"legacy")
            self.assertEqual(latest_complete_checkpoint_epoch(experiment, False), 6)

    def test_non_object_completion_json_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = root / "experiment"
            experiment.mkdir()
            (experiment / ".biowulf_singlefile_complete.json").write_text(
                "[]\n", encoding="utf-8"
            )
            self.assertFalse(
                completed_output_is_current(
                    experiment,
                    {"version": 1},
                    requested_epochs=1,
                    output_path=root / "missing.tif",
                    expected_shape=(1, 2, 3),
                )
            )


class ManifestCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _signature() -> dict:
        return {
            "version": 1,
            "scenario": "ome_chunks",
            "output_stem": "animal_session_2DT_Ch3",
            "channel": 3,
            "acquisition": "session_2DT",
            "temporal_keys": [[1, 1], [1, 2]],
            "dimensionality": "2+1D",
            "logical_shape": [10, 32, 32],
            "logical_dtype": "uint16",
            "sources": [
                {
                    "path": "/shared/session_2DT_Ch3_000001.ome.tif",
                    "shape": [5, 32, 32],
                    "dtype": "uint16",
                    "size": 10240,
                    "mtime_ns": 123456789,
                    "device": 10,
                    "inode": 20,
                }
            ],
            "training": {
                "patch_size": [16, 64, 64],
                "patch_interval": [8, 32, 32],
                "batch_size": 4,
                "load_to_memory": False,
                "pin_memory": True,
                "prefetch_factor": 8,
                "lr": 0.0001,
                "use_amp": True,
                "depth": 5,
            },
            "testing": {
                "patch_interval": [8, 32, 32],
                "batch_size": 4,
                "pin_memory": True,
                "prefetch_factor": 8,
                "pad": True,
            },
        }

    def test_runtime_and_compute_node_differences_refresh_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment = Path(temporary_directory) / "experiment"
            experiment.mkdir()
            original = self._signature()
            self.assertEqual(
                validate_or_create_manifest(experiment, original, resume=True),
                "new run",
            )
            checkpoint = experiment / "pth" / "sentinel.pth"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"preserve me")

            current = copy.deepcopy(original)
            current["sources"][0]["device"] = 999
            current["training"]["pin_memory"] = False
            current["training"]["prefetch_factor"] = 4
            current["testing"]["batch_size"] = 1
            current["testing"]["pin_memory"] = False
            current["testing"]["prefetch_factor"] = 4

            state = validate_or_create_manifest(experiment, current, resume=True)

            self.assertIn("runtime differences accepted", state)
            with (experiment / singlefile.RUN_MANIFEST_NAME).open(
                "r", encoding="utf-8"
            ) as handle:
                self.assertEqual(json.load(handle), original)
            self.assertEqual(checkpoint.read_bytes(), b"preserve me")

    def test_semantic_mismatch_is_typed_and_preserves_saved_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment = Path(temporary_directory) / "experiment"
            experiment.mkdir()
            original = self._signature()
            validate_or_create_manifest(experiment, original, resume=True)
            manifest = experiment / singlefile.RUN_MANIFEST_NAME
            checkpoint = experiment / "pth" / "sentinel.pth"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"preserve me")
            original_manifest = manifest.read_bytes()
            changed = copy.deepcopy(original)
            changed["training"]["lr"] = 0.001

            with self.assertRaisesRegex(SavedRunMismatchError, r"training\.lr"):
                validate_or_create_manifest(experiment, changed, resume=True)

            self.assertEqual(manifest.read_bytes(), original_manifest)
            self.assertEqual(checkpoint.read_bytes(), b"preserve me")

    def test_unknown_or_versioned_identity_changes_fail_closed(self) -> None:
        for label, mutate in (
            ("schema version", lambda value: value.__setitem__("version", 2)),
            (
                "source inode",
                lambda value: value["sources"][0].__setitem__("inode", 999),
            ),
            (
                "normalization path",
                lambda value: value["training"].__setitem__(
                    "load_to_memory", True
                ),
            ),
            (
                "unknown top-level identity",
                lambda value: value.__setitem__("future_semantic_field", "changed"),
            ),
        ):
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    experiment = Path(temporary_directory) / "experiment"
                    experiment.mkdir()
                    original = self._signature()
                    validate_or_create_manifest(experiment, original, resume=True)
                    manifest = experiment / singlefile.RUN_MANIFEST_NAME
                    original_manifest = manifest.read_bytes()
                    changed = copy.deepcopy(original)
                    mutate(changed)

                    with self.assertRaises(SavedRunMismatchError):
                        validate_or_create_manifest(experiment, changed, resume=True)

                    self.assertEqual(manifest.read_bytes(), original_manifest)

    def test_completion_identity_ignores_runtime_and_mount_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = root / "experiment"
            checkpoint_directory = experiment / "pth"
            checkpoint_directory.mkdir(parents=True)
            model = checkpoint_directory / "model_0.pth"
            optimizer = checkpoint_directory / "optimizer_0.pth"
            model.write_bytes(b"model")
            optimizer.write_bytes(b"optimizer")
            output = root / "output.tif"
            data = np.zeros((10, 32, 32), dtype=np.float32)
            tifffile.imwrite(output, data, imagej=True, metadata={"axes": "TYX"})
            original = self._signature()
            write_completion_record(experiment, original, 1, model, output)
            current = copy.deepcopy(original)
            current["sources"][0]["device"] = 500
            current["training"]["prefetch_factor"] = 4
            current["testing"]["prefetch_factor"] = 4

            self.assertTrue(
                completed_output_is_current(
                    experiment,
                    current,
                    requested_epochs=1,
                    output_path=output,
                    expected_shape=data.shape,
                    use_amp=False,
                )
            )

            # Completion records also persist st_dev for diagnostics, but a
            # different mount device on another node must not force inference.
            record_path = experiment / singlefile.COMPLETION_RECORD_NAME
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["model"]["device"] = 700
            record["output"]["device"] = 800
            record_path.write_text(json.dumps(record), encoding="utf-8")
            self.assertTrue(
                completed_output_is_current(
                    experiment,
                    current,
                    requested_epochs=1,
                    output_path=output,
                    expected_shape=data.shape,
                    use_amp=False,
                )
            )

    def test_legacy_completion_hash_is_reused_for_equivalent_runtime_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = root / "experiment"
            experiment.mkdir()
            original = self._signature()
            validate_or_create_manifest(experiment, original, resume=True)
            checkpoint_directory = experiment / "pth"
            checkpoint_directory.mkdir()
            model = checkpoint_directory / "model_0.pth"
            optimizer = checkpoint_directory / "optimizer_0.pth"
            model.write_bytes(b"model")
            optimizer.write_bytes(b"optimizer")
            output = root / "output.tif"
            data = np.zeros((10, 32, 32), dtype=np.float32)
            tifffile.imwrite(output, data, imagej=True, metadata={"axes": "TYX"})
            write_completion_record(experiment, original, 1, model, output)
            record_path = experiment / singlefile.COMPLETION_RECORD_NAME
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["configuration_sha256"] = singlefile._legacy_configuration_sha256(
                original
            )
            record_path.write_text(json.dumps(record), encoding="utf-8")
            current = copy.deepcopy(original)
            current["sources"][0]["device"] = 900
            current["training"]["prefetch_factor"] = 4
            current["testing"]["prefetch_factor"] = 4

            self.assertTrue(
                completed_output_is_current(
                    experiment,
                    current,
                    requested_epochs=1,
                    output_path=output,
                    expected_shape=data.shape,
                    use_amp=False,
                )
            )

            # Output-affecting inference settings still invalidate reuse.
            current["testing"]["pad"] = False
            self.assertFalse(
                completed_output_is_current(
                    experiment,
                    current,
                    requested_epochs=1,
                    output_path=output,
                    expected_shape=data.shape,
                    use_amp=False,
                )
            )


class WorkItemLockTests(unittest.TestCase):
    @staticmethod
    def _prepared(output_stem: str) -> PreparedWorkItem:
        return PreparedWorkItem(
            item=WorkItem(
                scenario="single_tiff",
                source_paths=(Path(f"/{output_stem}.tif"),),
                output_stem=output_stem,
            ),
            sources=(),
            logical_shape=(1, 2, 3),
            logical_dtype="uint16",
        )

    def test_only_an_actively_held_lock_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment = Path(temporary_directory) / "experiment"

            with exclusive_item_lock(experiment):
                with self.assertRaisesRegex(
                    WorkItemBusyError,
                    r"another process is training .*job_id=.*hostname=.*pid=",
                ):
                    with exclusive_item_lock(experiment):
                        self.fail("a second process must not acquire an active lock")

            # The persistent sidecar is intentionally retained.  Once its
            # advisory lock is released, it must not look like active work.
            self.assertTrue((experiment / singlefile.RUN_LOCK_NAME).is_file())
            with exclusive_item_lock(experiment):
                pass

    def test_skippable_item_states_do_not_send_a_slack_failure(self) -> None:
        for patched_name, error in (
            (
                "exclusive_item_lock",
                WorkItemBusyError("another process owns this item"),
            ),
            (
                "validate_or_create_manifest",
                SavedRunMismatchError("saved state is incompatible"),
            ),
        ):
            with self.subTest(error=type(error).__name__):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    notifier = mock.Mock()
                    with mock.patch.object(
                        singlefile,
                        patched_name,
                        side_effect=error,
                    ):
                        with self.assertRaises(type(error)):
                            run_work_item(
                                self._prepared("busy_2DT_Ch2"),
                                mock.Mock(),
                                {},
                                {},
                                root / "output",
                                root / "work",
                                root / "scratch",
                                notifier,
                            )

                    notifier.notify_failure.assert_not_called()
                    notifier.notify_run_started.assert_not_called()

    def test_source_change_before_item_turn_does_not_create_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "input" / "scan_2DT.tif"
            source.parent.mkdir()
            tifffile.imwrite(
                source,
                np.zeros((3, 4, 5), dtype=np.uint16),
                metadata={"axes": "TYX"},
            )
            item = WorkItem(
                scenario="single_tiff",
                source_paths=(source,),
                output_stem="scan_2DT",
            )
            prepared = prepare_work_items([item], "2+1D", (1, 1, 1))[0]
            source_stat = source.stat()
            os.utime(
                source,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000),
            )
            work_directory = root / "work"
            notifier = mock.Mock()

            with self.assertRaisesRegex(
                SingleFileConfigurationError, "input TIFF changed after preflight"
            ):
                run_work_item(
                    prepared,
                    mock.Mock(dimensionality="2+1D", resume=True),
                    {},
                    {},
                    root / "output",
                    work_directory,
                    root / "scratch",
                    notifier,
                )

            self.assertFalse(
                (work_directory / item.output_stem / singlefile.RUN_MANIFEST_NAME).exists()
            )
            notifier.notify_failure.assert_called_once()

    def test_main_skips_busy_item_and_runs_the_next_item(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_directory = root / "input"
            output_directory = root / "output"
            scratch_directory = root / "scratch"
            parameters = root / "parameters.json"
            input_directory.mkdir()
            parameters.write_text("{}\n", encoding="utf-8")
            prepared_items = [
                self._prepared("busy_2DT_Ch2"),
                self._prepared("incompatible_2DT_Ch3"),
                self._prepared("runnable_2DT_Ch3"),
            ]
            runnable_output = output_directory / "runnable_2DT_Ch3_SPRT.tif"

            with (
                mock.patch.object(
                    singlefile,
                    "build_effective_parameters",
                    return_value=({"patch_size": (1, 2, 3)}, {}),
                ),
                mock.patch.object(
                    singlefile,
                    "discover_work_items",
                    return_value=[prepared.item for prepared in prepared_items],
                ),
                mock.patch.object(
                    singlefile,
                    "prepare_work_items",
                    return_value=prepared_items,
                ),
                mock.patch.object(singlefile, "print_plan"),
                mock.patch.object(
                    singlefile,
                    "run_work_item",
                    side_effect=[
                        WorkItemBusyError("owned by another Slurm job"),
                        SavedRunMismatchError("training.lr differs"),
                        runnable_output,
                    ],
                ) as mocked_run,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                result = singlefile.main(
                    [
                        "--input-directory",
                        str(input_directory),
                        "--output-directory",
                        str(output_directory),
                        "--scratch-directory",
                        str(scratch_directory),
                        "--parameters",
                        str(parameters),
                        "--use-cpu",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(mocked_run.call_count, 3)
            output = stdout.getvalue()
            self.assertIn("Skipping busy logical job busy_2DT_Ch2", output)
            self.assertIn(
                "Skipping logical job with incompatible saved checkpoints "
                "incompatible_2DT_Ch3",
                output,
            )
            self.assertIn("1 logical job(s) completed or already verified", output)
            self.assertIn("1 skipped because another process", output)
            self.assertIn("1 skipped because saved checkpoints", output)
            self.assertIn(str(runnable_output), output)


if __name__ == "__main__":
    unittest.main()
