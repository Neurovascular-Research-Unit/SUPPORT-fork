"""Focused tests for inference DataLoader worker configuration."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


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


if __name__ == "__main__":
    unittest.main()
