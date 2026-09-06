"""Focused tests for noncritical TrainingClass lifecycle callbacks."""

from __future__ import annotations

import importlib.util
import io
import sys
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def load_training_module():
    """Load train_collection with lightweight stand-ins for torch dependencies."""
    torch_module = types.ModuleType("torch")
    torch_utils_module = types.ModuleType("torch.utils")
    torch_data_module = types.ModuleType("torch.utils.data")
    torch_data_module.DataLoader = object
    torch_utils_module.data = torch_data_module
    torch_module.utils = torch_utils_module

    support_module = types.ModuleType("support.SUPPORT")
    support_module.SUPPORT = object
    dataset_module = types.ModuleType("utils.dataset")
    dataset_module.random_transform = mock.Mock()
    dataset_module.DatasetSUPPORT = object

    module_path = Path(__file__).resolve().parents[1] / "utils" / "train_collection.py"
    spec = importlib.util.spec_from_file_location(
        "_support_training_lifecycle_under_test",
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


TRAINING_MODULE = load_training_module()
TrainingClass = TRAINING_MODULE.TrainingClass


class RecordingDataset:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def precompute_indices(self) -> None:
        self.events.append("precompute")


class RecordingLog:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def append_loss(self, *args, **kwargs) -> None:
        self.events.append("loss_log")


class TrainingLifecycleTests(unittest.TestCase):
    def test_epoch_callbacks_surround_training_validation_checkpoint_and_log(self) -> None:
        events: list[str] = []
        validation_metrics = {
            "loss_total": 0.25,
            "loss_l1": 0.2,
            "loss_l2": 0.3,
        }
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.epoch = 0
        trainer.n_epochs = 1
        trainer.dataloader = SimpleNamespace(dataset=RecordingDataset(events))
        trainer.checkpoints = True
        trainer.train_gen = True
        trainer.save_every_epoch = True
        trainer._checkpoint_controls_configured = True
        trainer.log = RecordingLog(events)

        def train() -> None:
            events.append("train")
            trainer.lh_L = [0.5]
            trainer.lh_L1 = [0.4]
            trainer.lh_L2 = [0.6]

        def validate(active_trainer, epoch, training_metrics):
            self.assertIs(active_trainer, trainer)
            self.assertEqual(epoch, 0)
            self.assertEqual(training_metrics["loss_total"], 0.5)
            events.append("validation")
            return validation_metrics

        def epoch_completed(
            active_trainer,
            epoch,
            training_metrics,
            received_validation_metrics,
        ) -> None:
            self.assertIs(active_trainer, trainer)
            self.assertEqual(epoch, 0)
            self.assertEqual(training_metrics["loss_total"], 0.5)
            self.assertIs(received_validation_metrics, validation_metrics)
            events.append("complete")

        trainer.train = train
        trainer.save_checkpoint = lambda epoch: events.append("checkpoint")
        trainer.epoch_start_callback = lambda active_trainer, epoch: events.append(
            "start"
        )
        trainer.epoch_end_callback = validate
        trainer.epoch_complete_callback = epoch_completed

        with mock.patch.object(TRAINING_MODULE, "plotASCII"):
            trainer.test()

        self.assertEqual(
            events,
            [
                "start",
                "precompute",
                "train",
                "validation",
                "checkpoint",
                "loss_log",
                "complete",
            ],
        )

    def test_optional_callback_exception_warns_without_leaking_message(self) -> None:
        trainer = TrainingClass.__new__(TrainingClass)
        secret_error_text = (
            "request failed at "
            "https://hooks.slack.com/services/TEST/ONLY/SENSITIVE_TOKEN"
        )

        def failing_callback(*args) -> None:
            raise RuntimeError(secret_error_text)

        trainer.epoch_start_callback = failing_callback
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            result = trainer._run_optional_callback(
                "epoch_start_callback",
                trainer,
                0,
            )

        warning = stderr.getvalue()
        self.assertIsNone(result)
        self.assertIn("epoch_start_callback failed (RuntimeError)", warning)
        self.assertIn("training will continue", warning)
        self.assertNotIn(secret_error_text, warning)
        self.assertNotIn("SENSITIVE_TOKEN", warning)


if __name__ == "__main__":
    unittest.main()
