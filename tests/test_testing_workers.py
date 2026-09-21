"""Focused tests for inference worker configuration and stack conversion."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


def load_testing_module():
    """Load test_collection with lightweight stand-ins for torch dependencies."""
    torch_module = types.ModuleType("torch")
    torch_utils_module = types.ModuleType("torch.utils")
    torch_data_module = types.ModuleType("torch.utils.data")
    torch_data_module.DataLoader = object
    torch_utils_module.data = torch_data_module
    torch_module.utils = torch_utils_module

    support_module = types.ModuleType("support.SUPPORT")
    support_module.SUPPORT = object
    dataset_module = types.ModuleType("utils.dataset")
    dataset_module.DatasetSUPPORT_test_stitch = object

    module_path = Path(__file__).resolve().parents[1] / "utils" / "test_collection.py"
    spec = importlib.util.spec_from_file_location(
        "_support_testing_workers_under_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "torch": torch_module,
        "torch.utils": torch_utils_module,
        "torch.utils.data": torch_data_module,
        "support.SUPPORT": support_module,
        "utils.dataset": dataset_module,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, prior_module in previous.items():
            if prior_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior_module
    return module


TESTING_MODULE = load_testing_module()
TestingClass = TESTING_MODULE.TestingClass


class TestingWorkerConfigurationTests(unittest.TestCase):
    @staticmethod
    def _tester() -> TestingClass:
        tester = TestingClass.__new__(TestingClass)
        tester.batch_size = 32
        tester.n_cpu = 4
        tester.cuda = True
        tester.pin_memory = True
        tester.prefetch_factor = 3
        return tester

    def test_command_line_inference_uses_configured_workers_and_prefetch(self) -> None:
        tester = self._tester()

        with mock.patch.object(tester, "_in_ipykernel", return_value=False):
            kwargs = tester._dataloader_kwargs()

        self.assertEqual(kwargs["num_workers"], 4)
        self.assertEqual(kwargs["prefetch_factor"], 3)
        self.assertEqual(kwargs["batch_size"], 32)
        self.assertTrue(kwargs["pin_memory"])

    def test_notebook_inference_uses_main_process_without_prefetch(self) -> None:
        tester = self._tester()

        with mock.patch.object(tester, "_in_ipykernel", return_value=True):
            kwargs = tester._dataloader_kwargs()

        self.assertEqual(kwargs["num_workers"], 0)
        self.assertNotIn("prefetch_factor", kwargs)

    def test_zero_configured_workers_omits_prefetch(self) -> None:
        tester = self._tester()
        tester.n_cpu = 0

        with mock.patch.object(tester, "_in_ipykernel", return_value=False):
            kwargs = tester._dataloader_kwargs()

        self.assertEqual(kwargs["num_workers"], 0)
        self.assertNotIn("prefetch_factor", kwargs)


class TestingStackConversionTests(unittest.TestCase):
    @staticmethod
    def _tester(n_cpu: int = 3) -> TestingClass:
        tester = TestingClass.__new__(TestingClass)
        tester.n_cpu = n_cpu
        return tester

    def test_parallel_conversion_preserves_uint16_stack(self) -> None:
        source = np.arange(7 * 2 * 3 * 4, dtype=np.uint16).reshape(7, 2, 3, 4)
        real_executor = TESTING_MODULE.ThreadPoolExecutor

        with mock.patch.object(
            TESTING_MODULE,
            "ThreadPoolExecutor",
            wraps=real_executor,
        ) as executor:
            converted = TESTING_MODULE._as_float32_parallel(source, max_workers=3)

        executor.assert_called_once_with(max_workers=3)
        self.assertEqual(converted.dtype, np.float32)
        self.assertEqual(converted.shape, source.shape)
        np.testing.assert_array_equal(converted, source.astype(np.float32))

    def test_float32_conversion_does_not_copy_an_ndarray(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        converted = TESTING_MODULE._as_float32_parallel(source, max_workers=8)

        self.assertIs(converted, source)

    def test_zero_conversion_workers_runs_in_calling_thread(self) -> None:
        source = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

        with mock.patch.object(
            TESTING_MODULE,
            "ThreadPoolExecutor",
            side_effect=AssertionError("executor should not be created"),
        ):
            converted = TESTING_MODULE._as_float32_parallel(source, max_workers=0)

        np.testing.assert_array_equal(converted, source.astype(np.float32))

    def test_denormalization_is_in_place(self) -> None:
        normalized = np.array([[[-1.0, 0.0], [1.0, 2.0]]], dtype=np.float32)
        expected = normalized.copy()
        expected *= np.float32(2.5)
        expected += np.float32(-1.25)

        result = TESTING_MODULE._denormalize_in_place(
            normalized,
            np.float32(2.5),
            np.float32(-1.25),
        )

        self.assertIs(result, normalized)
        np.testing.assert_array_equal(result, expected)

    def test_read_stack_prefers_memory_mapping(self) -> None:
        source = np.arange(5 * 3 * 4, dtype=np.uint16).reshape(5, 3, 4)
        tester = self._tester(n_cpu=2)

        with (
            mock.patch.object(
                TESTING_MODULE.tifffile,
                "memmap",
                return_value=source,
            ) as memmap,
            mock.patch.object(
                TESTING_MODULE.tifffile,
                "TiffFile",
                side_effect=AssertionError("TiffFile fallback should not be used"),
            ),
        ):
            converted = tester._read_stack("input.tif")

        memmap.assert_called_once()
        self.assertEqual(memmap.call_args.args[0], "input.tif")
        self.assertEqual(memmap.call_args.kwargs["mode"], "c")
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_array_equal(converted, source.astype(np.float32))

    def test_float32_memmap_is_writable_without_modifying_input(self) -> None:
        source = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)
        tester = self._tester(n_cpu=2)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "float32.tif"
            TESTING_MODULE.tifffile.imwrite(
                path,
                source,
                photometric="minisblack",
                metadata={"axes": "TYX"},
            )

            converted = tester._read_stack(path)

            self.assertIs(type(converted), np.ndarray)
            self.assertIsInstance(converted.base, np.memmap)
            self.assertTrue(converted.flags.writeable)
            np.testing.assert_array_equal(converted, source)
            converted[0, 0, 0] = -1.0
            np.testing.assert_array_equal(
                TESTING_MODULE.tifffile.imread(path),
                source,
            )

    def test_read_stack_falls_back_when_tiff_cannot_be_memory_mapped(self) -> None:
        source = np.arange(5 * 3 * 4, dtype=np.uint16).reshape(5, 3, 4)
        tester = self._tester(n_cpu=2)
        tiff_context = mock.MagicMock()
        tiff_context.__enter__.return_value.asarray.return_value = source

        with (
            mock.patch.object(
                TESTING_MODULE.tifffile,
                "memmap",
                side_effect=ValueError("image data are not memory-mappable"),
            ),
            mock.patch.object(
                TESTING_MODULE.tifffile,
                "TiffFile",
                return_value=tiff_context,
            ) as tiff_file,
        ):
            converted = tester._read_stack("compressed.tif")

        tiff_file.assert_called_once_with("compressed.tif")
        tiff_context.__enter__.return_value.asarray.assert_called_once_with()
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_array_equal(converted, source.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
