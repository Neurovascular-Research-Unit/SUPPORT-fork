"""Focused tests for TrainingClass checkpoint and callback lifecycle."""

from __future__ import annotations

import importlib.util
import io
import math
import sys
import tempfile
import types
import unittest
import warnings
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


class RecordingStateTarget:
    def __init__(self) -> None:
        self.loaded = []

    def load_state_dict(self, state) -> None:
        self.loaded.append(state)


class RecordingStateSource:
    def state_dict(self):
        return {"weight": 1}


class TrainingLifecycleTests(unittest.TestCase):
    def test_loss_aggregates_retain_mean_and_ignore_nans_in_nanmean(self) -> None:
        mean, nanmean = TRAINING_MODULE._mean_and_nanmean(
            [1.0, float("nan"), 3.0]
        )

        self.assertTrue(math.isnan(mean))
        self.assertEqual(nanmean, 2.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            all_nan_mean, all_nan_nanmean = TRAINING_MODULE._mean_and_nanmean(
                [float("nan"), float("nan")]
            )

        self.assertTrue(math.isnan(all_nan_mean))
        self.assertTrue(math.isnan(all_nan_nanmean))
        self.assertEqual(caught, [])

    def test_tqdm_uses_nanmean_while_batch_log_retains_both_means(self) -> None:
        class FakeLoss:
            def __init__(self, value) -> None:
                self.value = float(value)

            def __rmul__(self, coefficient):
                return FakeLoss(coefficient * self.value)

            def __add__(self, other):
                return FakeLoss(self.value + other.value)

            def backward(self) -> None:
                pass

            def item(self):
                return self.value

        class SequenceLoss:
            def __init__(self) -> None:
                self.values = iter([1.0, float("nan"), 3.0])

            def __call__(self, prediction, target):
                return FakeLoss(next(self.values))

        class FakeImage:
            shape = (1, 3, 2, 2)

            def to(self, device):
                return self

            def __getitem__(self, key):
                return self

        class FakeModel:
            def train(self) -> None:
                pass

            def __call__(self, image):
                return image

        class FakeOptimizer:
            def zero_grad(self) -> None:
                pass

        class FakeScaler:
            def scale(self, loss):
                return loss

            def step(self, optimizer) -> None:
                pass

            def update(self) -> None:
                pass

        class FakeAutocast:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        class FakeProgress:
            def __init__(self, values) -> None:
                self.values = values
                self.postfixes = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def __iter__(self):
                return iter(self.values)

            def set_postfix(self, values) -> None:
                self.postfixes.append(dict(values))

        trainer = TrainingClass.__new__(TrainingClass)
        trainer.bs_size = [3, 3]
        trainer.local_model = FakeModel()
        trainer.dataloader = [(FakeImage(), None, None)] * 3
        trainer.device = "cpu"
        trainer.use_amp = False
        trainer.rng = object()
        trainer.optimizer = FakeOptimizer()
        trainer.scaler = FakeScaler()
        trainer.loss_coef = [0.5, 0.5]
        trainer.epoch = 0
        trainer.train_gen = True
        trainer.latest_checkpoint_epoch = None
        trainer.latest_checkpoint_batch = None
        trainer._scheduled_batch_checkpoint_batches = lambda total: set()
        batch_logs = []
        trainer.log = SimpleNamespace(
            append_batch_loss=lambda *args, **values: batch_logs.append(values)
        )
        progress = FakeProgress(trainer.dataloader)
        fake_nn = SimpleNamespace(
            L1Loss=SequenceLoss,
            MSELoss=SequenceLoss,
        )
        fake_amp = SimpleNamespace(
            autocast=lambda *args, **kwargs: FakeAutocast()
        )

        with (
            mock.patch.object(TRAINING_MODULE, "tqdm", return_value=progress),
            mock.patch.object(
                TRAINING_MODULE,
                "random_transform",
                side_effect=lambda input, target, **kwargs: (input, target),
            ),
            mock.patch.object(TRAINING_MODULE.torch, "nn", fake_nn, create=True),
            mock.patch.object(TRAINING_MODULE.torch, "amp", fake_amp, create=True),
            mock.patch.object(
                TRAINING_MODULE.torch,
                "unsqueeze",
                side_effect=lambda image, dim: image,
                create=True,
            ),
        ):
            trainer.train()

        self.assertEqual(
            progress.postfixes[-1],
            {"L": "2.0000", "L1": "2.0000", "L2": "2.0000"},
        )
        self.assertTrue(math.isnan(batch_logs[-1]["L_mean"]))
        self.assertTrue(math.isnan(batch_logs[-1]["L1_mean"]))
        self.assertTrue(math.isnan(batch_logs[-1]["L2_mean"]))
        self.assertEqual(batch_logs[-1]["L_nanmean"], 2.0)
        self.assertEqual(batch_logs[-1]["L1_nanmean"], 2.0)
        self.assertEqual(batch_logs[-1]["L2_nanmean"], 2.0)

    def test_legacy_pending_metrics_log_nanmean_fallbacks(self) -> None:
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.log = mock.Mock()
        trainer.epoch_complete_callback = None
        training_metrics = {
            "loss_total": 0.5,
            "loss_l1": 0.4,
            "loss_l2": 0.6,
        }

        trainer._record_epoch_completion(0, training_metrics, None)

        trainer.log.append_loss.assert_called_once_with(
            0,
            loss_total=0.5,
            loss_l1=0.4,
            loss_l2=0.6,
            loss_total_nanmean=0.5,
            loss_l1_nanmean=0.4,
            loss_l2_nanmean=0.6,
            extra=None,
        )

    def test_epoch_checkpoint_precedes_validation_and_completion(self) -> None:
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
        trainer.latest_checkpoint_epoch = None
        trainer.latest_checkpoint_batch = None

        def train() -> None:
            events.append("train")
            trainer.lh_L = [0.5]
            trainer.lh_L1 = [0.4]
            trainer.lh_L2 = [0.6]

        def validate(active_trainer, epoch, training_metrics):
            self.assertIs(active_trainer, trainer)
            self.assertEqual(epoch, 0)
            self.assertEqual(training_metrics["loss_total"], 0.5)
            self.assertTrue(Path(trainer.model_path).is_file())
            self.assertTrue(trainer._pending_validation_path().is_file())
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

        def save_checkpoint(epoch) -> None:
            events.append("checkpoint")
            model_path = Path(trainer.pth_dir) / f"model_{epoch}.pth"
            model_path.write_bytes(b"saved model")
            trainer.model_path = str(model_path)

        trainer.save_checkpoint = save_checkpoint
        trainer.epoch_start_callback = lambda active_trainer, epoch: events.append(
            "start"
        )
        trainer.epoch_end_callback = validate
        trainer.epoch_complete_callback = epoch_completed

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer.pth_dir = temporary_directory
            with mock.patch.object(
                TRAINING_MODULE,
                "plotASCII",
                side_effect=lambda history: events.append("plot"),
            ):
                trainer.test()
            self.assertFalse(trainer._pending_validation_path().exists())

        self.assertEqual(
            events,
            [
                "start",
                "precompute",
                "train",
                "checkpoint",
                "plot",
                "validation",
                "loss_log",
                "complete",
            ],
        )

    def test_validation_failure_leaves_saved_model_and_pending_state(self) -> None:
        events: list[str] = []
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.epoch = 0
        trainer.n_epochs = 1
        trainer.dataloader = SimpleNamespace(dataset=RecordingDataset(events))
        trainer.checkpoints = True
        trainer.train_gen = True
        trainer.save_every_epoch = True
        trainer._checkpoint_controls_configured = True
        trainer.log = RecordingLog(events)
        trainer.latest_checkpoint_epoch = None
        trainer.latest_checkpoint_batch = None
        trainer.epoch_start_callback = lambda active_trainer, epoch: events.append(
            "start"
        )
        trainer.epoch_complete_callback = lambda *args: events.append("complete")

        def train() -> None:
            events.append("train")
            trainer.lh_L = [0.5]
            trainer.lh_L1 = [0.4]
            trainer.lh_L2 = [0.6]

        def save_checkpoint(epoch) -> None:
            events.append("checkpoint")
            model_path = Path(trainer.pth_dir) / f"model_{epoch}.pth"
            model_path.write_bytes(b"saved model")
            trainer.model_path = str(model_path)
            trainer.latest_checkpoint_epoch = epoch
            trainer.latest_checkpoint_batch = None

        def fail_validation(active_trainer, epoch, training_metrics):
            events.append("validation")
            raise RuntimeError("validation failed")

        trainer.train = train
        trainer.save_checkpoint = save_checkpoint
        trainer.epoch_end_callback = fail_validation

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer.pth_dir = temporary_directory
            with mock.patch.object(
                TRAINING_MODULE,
                "plotASCII",
                side_effect=lambda history: events.append("plot"),
            ):
                with self.assertRaisesRegex(RuntimeError, "validation failed"):
                    trainer.test()

            self.assertTrue(Path(trainer.model_path).is_file())
            pending = trainer._read_pending_validation()
            self.assertEqual(pending["epoch"], 0)
            self.assertEqual(pending["training_metrics"]["loss_total"], 0.5)

        self.assertEqual(
            events,
            [
                "start",
                "precompute",
                "train",
                "checkpoint",
                "plot",
                "validation",
            ],
        )

    def test_resume_retries_pending_validation_without_retraining(self) -> None:
        events: list[str] = []
        validation_metrics = {
            "loss_total": 0.25,
            "loss_l1": 0.2,
            "loss_l2": 0.3,
        }
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.epoch = 1
        trainer.n_epochs = 1
        trainer.latest_checkpoint_epoch = 0
        trainer.latest_checkpoint_batch = None
        trainer.log = RecordingLog(events)
        trainer.epoch_start_callback = lambda *args: events.append("start")
        trainer.epoch_complete_callback = lambda *args: events.append("complete")
        trainer.train = lambda: events.append("train")
        expected_training_metrics = {
            "loss_total": 0.5,
            "loss_l1": 0.4,
            "loss_l2": 0.6,
            "loss_total_nanmean": 0.45,
            "loss_l1_nanmean": 0.35,
            "loss_l2_nanmean": 0.55,
        }

        def validate(active_trainer, epoch, training_metrics):
            self.assertIs(active_trainer, trainer)
            self.assertEqual(active_trainer.epoch, 0)
            self.assertEqual(epoch, 0)
            self.assertEqual(training_metrics, expected_training_metrics)
            events.append("validation")
            return validation_metrics

        trainer.epoch_end_callback = validate

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer.pth_dir = temporary_directory
            trainer._write_pending_validation(
                0,
                expected_training_metrics,
            )
            trainer.test()
            self.assertFalse(trainer._pending_validation_path().exists())

        self.assertEqual(events, ["start", "validation", "loss_log", "complete"])
        self.assertEqual(trainer.epoch, 1)
        self.assertIs(trainer.last_validation_metrics, validation_metrics)

    def test_resume_accepts_completed_final_batch_checkpoint(self) -> None:
        events: list[str] = []
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.epoch = 1
        trainer.n_epochs = 1
        trainer.latest_checkpoint_epoch = 0
        trainer.latest_checkpoint_batch = 3
        trainer.save_every_epoch = False
        trainer.dataloader = [None, None, None]
        trainer.log = RecordingLog(events)
        trainer.epoch_start_callback = lambda *args: events.append("start")
        trainer.epoch_complete_callback = lambda *args: events.append("complete")
        trainer.epoch_end_callback = lambda *args: (
            events.append("validation")
            or {"loss_total": 0.25, "loss_l1": 0.2, "loss_l2": 0.3}
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer.pth_dir = temporary_directory
            trainer._write_pending_validation(
                0,
                {"loss_total": 0.5, "loss_l1": 0.4, "loss_l2": 0.6},
            )
            trainer.test()
            self.assertFalse(trainer._pending_validation_path().exists())

        self.assertEqual(events, ["start", "validation", "loss_log", "complete"])

    def test_final_batch_checkpoint_is_promoted_before_resumed_validation(self) -> None:
        events: list[str] = []
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.epoch = 1
        trainer.n_epochs = 1
        trainer.latest_checkpoint_epoch = 0
        trainer.latest_checkpoint_batch = 4
        trainer.save_every_epoch = True
        trainer.dataloader = [None, None, None, None]
        trainer.log = RecordingLog(events)
        trainer.epoch_start_callback = lambda *args: events.append("start")
        trainer.epoch_complete_callback = lambda *args: events.append("complete")

        def save_checkpoint(epoch) -> None:
            events.append("epoch_checkpoint")
            trainer.latest_checkpoint_epoch = epoch
            trainer.latest_checkpoint_batch = None
            trainer.model_path = str(Path(trainer.pth_dir) / f"model_{epoch}.pth")

        def validate(*args):
            events.append("validation")
            return {"loss_total": 0.25, "loss_l1": 0.2, "loss_l2": 0.3}

        trainer.save_checkpoint = save_checkpoint
        trainer.epoch_end_callback = validate

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer.pth_dir = temporary_directory
            trainer._write_pending_validation(
                0,
                {"loss_total": 0.5, "loss_l1": 0.4, "loss_l2": 0.6},
            )
            trainer.test()

        self.assertEqual(
            events,
            ["epoch_checkpoint", "start", "validation", "loss_log", "complete"],
        )

    def test_quarter_checkpoint_batches_cover_four_epoch_quartiles(self) -> None:
        self.assertEqual(
            TrainingClass._quarter_checkpoint_batches(8536),
            (2134, 4268, 6402, 8536),
        )
        self.assertEqual(TrainingClass._quarter_checkpoint_batches(3), (1, 2, 3))

        trainer = TrainingClass.__new__(TrainingClass)
        trainer.train_gen = True
        trainer.checkpoints = True
        trainer.save_every_epoch = True
        self.assertEqual(
            trainer._scheduled_batch_checkpoint_batches(8536),
            {2134, 4268, 6402, 8536},
        )

    def test_rolling_batch_cleanup_preserves_epoch_checkpoints(self) -> None:
        trainer = TrainingClass.__new__(TrainingClass)
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer.pth_dir = temporary_directory
            checkpoint_dir = Path(temporary_directory)
            names = {
                "model_0.pth",
                "optimizer_0.pth",
                "model_1.pth",
                "optimizer_1.pth",
                "model_1_batch_25.pth",
                "optimizer_1_batch_25.pth",
                "scaler_1_batch_25.pth",
                ".checkpoint_1_batch_25.incomplete",
                "model_1_batch_50.pth",
                "optimizer_1_batch_50.pth",
                "scaler_1_batch_50.pth",
                ".pending_validation.json",
                "unrelated.txt",
            }
            for name in names:
                (checkpoint_dir / name).touch()

            trainer._remove_batch_checkpoints(keep_epoch=1, keep_batch=50)
            remaining = {path.name for path in checkpoint_dir.iterdir()}

        self.assertIn("model_0.pth", remaining)
        self.assertIn("optimizer_0.pth", remaining)
        self.assertIn("model_1.pth", remaining)
        self.assertIn("optimizer_1.pth", remaining)
        self.assertIn("model_1_batch_50.pth", remaining)
        self.assertIn("optimizer_1_batch_50.pth", remaining)
        self.assertIn("scaler_1_batch_50.pth", remaining)
        self.assertIn(".pending_validation.json", remaining)
        self.assertIn("unrelated.txt", remaining)
        self.assertNotIn("model_1_batch_25.pth", remaining)
        self.assertNotIn("optimizer_1_batch_25.pth", remaining)
        self.assertNotIn("scaler_1_batch_25.pth", remaining)
        self.assertNotIn(".checkpoint_1_batch_25.incomplete", remaining)

    def test_batch_save_rolls_forward_then_epoch_save_supersedes_it(self) -> None:
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.local_model = RecordingStateSource()
        trainer.optimizer = RecordingStateSource()
        trainer.use_amp = False
        trainer.rm_intermediaries = False

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer.pth_dir = temporary_directory
            checkpoint_dir = Path(temporary_directory)
            for name in (
                "model_0.pth",
                "optimizer_0.pth",
                "model_1_batch_25.pth",
                "optimizer_1_batch_25.pth",
            ):
                (checkpoint_dir / name).touch()

            trainer._atomic_torch_save = (
                lambda state, destination: Path(destination).write_bytes(b"saved")
            )
            trainer.save_checkpoint(1, batch=50, remove_previous=True)
            after_batch = {path.name for path in checkpoint_dir.iterdir()}
            trainer.save_checkpoint(1)
            after_epoch = {path.name for path in checkpoint_dir.iterdir()}

        self.assertIn("model_0.pth", after_batch)
        self.assertIn("optimizer_0.pth", after_batch)
        self.assertIn("model_1_batch_50.pth", after_batch)
        self.assertIn("optimizer_1_batch_50.pth", after_batch)
        self.assertNotIn("model_1_batch_25.pth", after_batch)
        self.assertNotIn("optimizer_1_batch_25.pth", after_batch)
        self.assertIn("model_0.pth", after_epoch)
        self.assertIn("optimizer_0.pth", after_epoch)
        self.assertIn("model_1.pth", after_epoch)
        self.assertIn("optimizer_1.pth", after_epoch)
        self.assertFalse(any("_batch_" in name for name in after_epoch))

    def test_resume_prefers_newer_epoch_batch_checkpoint(self) -> None:
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.resume = True
        trainer.train_gen = True
        trainer.save_every_epoch = True
        trainer.use_amp = False
        trainer.device = "cpu"
        trainer.dataloader = [None] * 100
        trainer.local_model = RecordingStateTarget()
        trainer.optimizer = RecordingStateTarget()

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer.pth_dir = temporary_directory
            checkpoint_dir = Path(temporary_directory)
            for name in (
                "model_0.pth",
                "optimizer_0.pth",
                "model_1_batch_25.pth",
                "optimizer_1_batch_25.pth",
            ):
                (checkpoint_dir / name).touch()
            trainer._load_state_dict = lambda path: Path(path).name
            trainer.load_checkpoint()

        self.assertEqual(trainer.epoch, 1)
        self.assertEqual(trainer.latest_checkpoint_epoch, 1)
        self.assertEqual(trainer.latest_checkpoint_batch, 25)
        self.assertEqual(trainer.local_model.loaded, ["model_1_batch_25.pth"])
        self.assertEqual(trainer.optimizer.loaded, ["optimizer_1_batch_25.pth"])

    def test_resume_refuses_unrelated_pending_validation(self) -> None:
        trainer = TrainingClass.__new__(TrainingClass)
        trainer.epoch = 2
        trainer.n_epochs = 2
        trainer.latest_checkpoint_epoch = 1
        trainer.latest_checkpoint_batch = None
        trainer.dataloader = []
        trainer.epoch_end_callback = lambda *args: self.fail(
            "mismatched validation must not run"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer.pth_dir = temporary_directory
            trainer._write_pending_validation(
                0,
                {"loss_total": 0.5, "loss_l1": 0.4, "loss_l2": 0.6},
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "does not match the loaded checkpoint",
            ):
                trainer.test()
            self.assertTrue(trainer._pending_validation_path().exists())

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
