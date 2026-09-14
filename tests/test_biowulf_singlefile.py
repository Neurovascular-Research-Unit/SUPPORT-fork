"""Focused tests for the per-dataset Biowulf input planner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from scripts.biowulf_singlefile import (
    completed_output_is_current,
    concatenate_ome_chunks,
    discover_work_items,
    latest_complete_checkpoint_epoch,
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

    def test_two_plain_tiffs_are_independent_single_file_work_items(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "data_noisy"
            output_directory = root / "clean"
            dataset = input_root / "animal" / "session_2DT-004"
            first = self._write_tiff(dataset / "already_named_Ch2.tif", value=2)
            second = self._write_tiff(dataset / "already_named_Ch3.tif", value=3)
            output_directory.mkdir()

            items = discover_work_items(input_root, output_directory)

            self.assertEqual(len(items), 2)
            by_source = {item.source_paths[0]: item for item in items}
            self.assertEqual(set(by_source), {first.resolve(), second.resolve()})
            self.assertEqual(
                {item.output_stem for item in items},
                {"already_named_Ch2", "already_named_Ch3"},
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

    def test_ambiguous_or_malformed_dataset_layouts_are_rejected(self) -> None:
        cases = {
            "one plain TIFF": ("scan_2DT", ["only.tif"]),
            "three plain TIFFs": (
                "scan_2DT",
                ["first.tif", "second.tif", "third.tif"],
            ),
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


if __name__ == "__main__":
    unittest.main()
