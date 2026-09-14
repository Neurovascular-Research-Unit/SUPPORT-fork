"""Focused tests for releasing training tensors during validation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class FakeTensor:
    """Only supplies a type for ``isinstance(..., torch.Tensor)`` checks."""


class FakeDataLoader:
    def __init__(self, dataset, **kwargs) -> None:
        self.dataset = dataset
        self.kwargs = kwargs

    def __len__(self) -> int:
        return self.dataset.batch_count


def load_training_module():
    """Load train_collection with small stand-ins for its torch dependencies."""
    torch_module = types.ModuleType("torch")
    torch_utils_module = types.ModuleType("torch.utils")
    torch_data_module = types.ModuleType("torch.utils.data")
    torch_data_module.DataLoader = FakeDataLoader
    torch_utils_module.data = torch_data_module
    torch_module.utils = torch_utils_module
    torch_module.Tensor = FakeTensor
    torch_module.from_numpy = lambda array: np.asarray(array)

    support_module = types.ModuleType("support.SUPPORT")
    support_module.SUPPORT = object
    dataset_module = types.ModuleType("utils.dataset")
    dataset_module.random_transform = mock.Mock()
    dataset_module.DatasetSUPPORT = object

    module_path = Path(__file__).resolve().parents[1] / "utils" / "train_collection.py"
    spec = importlib.util.spec_from_file_location(
        "_support_training_offload_under_test",
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


class TinyDataset:
    def __init__(self, arrays, batch_count: int = 2) -> None:
        self.noisy_imgs = [np.array(array, dtype=np.float32, copy=True) for array in arrays]
        self.batch_count = batch_count
        self.precomputed_indices = [(0, 0, 0, 0)]
        self.patch_rng = np.random.default_rng(7)

    def precompute_indices(self) -> None:
        if not self.noisy_imgs:
            raise AssertionError("training data was not restored before the epoch")
        self.precomputed_indices = [(0, 0, 0, 0)]


class TrainingDataOffloadTests(unittest.TestCase):
    @staticmethod
    def _arrays():
        return [
            np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            np.array([[[-2.5, 0.25], [3.5, 9.0]]], dtype=np.float32),
        ]

    @staticmethod
    def _configure_offload(trainer, dataset, cache_directory: Path) -> None:
        trainer.load_to_memory = True
        trainer.offload_training_data_during_validation = True
        trainer.training_cache_dir = str(cache_directory)
        trainer._training_cache_root = None
        trainer._training_cache_files = ()
        trainer._training_cache_owned = False
        trainer._suspended_training_dataset = None
        trainer._training_dataloader_batch_count = None
        trainer._suspended_uses_training_cache = False
        trainer.epoch_end_callback = lambda *args: None
        trainer.batch_size = 2
        trainer.n_cpu = 0
        trainer.cuda = False
        trainer.pin_memory = False
        trainer.prefetch_factor = 1
        trainer.dataloader = FakeDataLoader(dataset)

    @staticmethod
    def _configure_lifecycle(trainer, dataset, cache_directory: Path, pth_dir: Path) -> None:
        TrainingDataOffloadTests._configure_offload(
            trainer,
            dataset,
            cache_directory,
        )
        trainer.epoch = 0
        trainer.n_epochs = 2
        trainer.checkpoints = True
        trainer.train_gen = True
        trainer.save_every_epoch = True
        trainer.rm_intermediaries = False
        trainer._checkpoint_controls_configured = True
        trainer.latest_checkpoint_epoch = None
        trainer.latest_checkpoint_batch = None
        trainer.model_path = None
        trainer.pth_dir = str(pth_dir)
        trainer.epoch_start_callback = None
        trainer.epoch_complete_callback = None

    def test_npy_cache_round_trip_is_exact_and_reused(self) -> None:
        arrays = self._arrays()
        expected = [array.copy() for array in arrays]
        dataset = TinyDataset(arrays)
        trainer = TrainingClass.__new__(TrainingClass)

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_base = Path(temporary_directory)
            self._configure_offload(trainer, dataset, cache_base)

            with mock.patch.object(
                TRAINING_MODULE,
                "tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ):
                self.assertTrue(trainer._ensure_training_cache(dataset))
                cache_root = Path(trainer._training_cache_root)
                cache_paths = tuple(
                    Path(metadata["path"])
                    for metadata in trainer._training_cache_files
                )
                cache_timestamps = tuple(path.stat().st_mtime_ns for path in cache_paths)

                for path, original in zip(cache_paths, expected):
                    cached = np.load(path, allow_pickle=False)
                    np.testing.assert_array_equal(cached, original)
                    self.assertEqual(cached.dtype, np.dtype(np.float32))

                self.assertTrue(trainer._ensure_training_cache(dataset))
                self.assertEqual(
                    cache_timestamps,
                    tuple(path.stat().st_mtime_ns for path in cache_paths),
                )
                patch_rng_state = dataset.patch_rng.bit_generator.state
                self.assertTrue(trainer._suspend_training_data_for_validation(True))
                self.assertFalse(hasattr(trainer, "dataloader"))
                self.assertEqual(dataset.noisy_imgs, [])
                self.assertTrue(trainer._restore_training_data_after_validation())

            for restored, original in zip(dataset.noisy_imgs, expected):
                np.testing.assert_array_equal(restored, original)
                self.assertEqual(restored.dtype, np.dtype(np.float32))
                self.assertTrue(restored.flags.c_contiguous)
            self.assertEqual(dataset.patch_rng.bit_generator.state, patch_rng_state)
            self.assertEqual(
                cache_timestamps,
                tuple(path.stat().st_mtime_ns for path in cache_paths),
            )

            trainer._remove_training_cache()
            self.assertFalse(cache_root.exists())
            self.assertTrue(cache_base.exists())

    def test_releases_before_validation_reloads_before_next_epoch_only(self) -> None:
        events = []
        dataset = TinyDataset(self._arrays())
        trainer = TrainingClass.__new__(TrainingClass)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._configure_lifecycle(trainer, dataset, root / "cache", root / "pth")
            (root / "pth").mkdir()

            def precompute_indices() -> None:
                if not dataset.noisy_imgs:
                    raise AssertionError("training data was not restored")
                events.append(f"precompute-{trainer.epoch}")

            dataset.precompute_indices = precompute_indices

            def train() -> None:
                if not dataset.noisy_imgs:
                    raise AssertionError("training ran without resident data")
                events.append(f"train-{trainer.epoch}")
                trainer.lh_L = [0.5]
                trainer.lh_L1 = [0.4]
                trainer.lh_L2 = [0.6]

            def save_checkpoint(epoch) -> None:
                events.append(f"checkpoint-{epoch}")
                model_path = Path(trainer.pth_dir) / f"model_{epoch}.pth"
                model_path.write_bytes(b"model")
                trainer.model_path = str(model_path)
                trainer.latest_checkpoint_epoch = epoch
                trainer.latest_checkpoint_batch = None

            def validate(active_trainer, epoch, training_metrics):
                self.assertIs(active_trainer, trainer)
                self.assertFalse(hasattr(trainer, "dataloader"))
                self.assertEqual(dataset.noisy_imgs, [])
                events.append(f"validation-{epoch}")
                return {"loss_total": 0.25, "loss_l1": 0.2, "loss_l2": 0.3}

            original_suspend = trainer._suspend_training_data_for_validation
            original_restore = trainer._restore_training_data_after_validation

            def suspend(reload_after_validation) -> bool:
                events.append(f"suspend-{reload_after_validation}")
                return original_suspend(reload_after_validation)

            def restore() -> bool:
                events.append("restore")
                return original_restore()

            trainer.train = train
            trainer.save_checkpoint = save_checkpoint
            trainer.epoch_end_callback = validate
            trainer._suspend_training_data_for_validation = suspend
            trainer._restore_training_data_after_validation = restore

            with mock.patch.object(TRAINING_MODULE, "plotASCII"), mock.patch.object(
                TRAINING_MODULE,
                "tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ):
                trainer.test()

        self.assertEqual(
            events,
            [
                "precompute-0",
                "train-0",
                "checkpoint-0",
                "suspend-True",
                "validation-0",
                "restore",
                "precompute-1",
                "train-1",
                "checkpoint-1",
                "suspend-False",
                "validation-1",
            ],
        )
        self.assertFalse(hasattr(trainer, "dataloader"))
        self.assertEqual(dataset.noisy_imgs, [])
        self.assertIs(trainer._suspended_training_dataset, dataset)

    def test_pending_validation_offloads_then_restores_for_next_epoch(self) -> None:
        events = []
        dataset = TinyDataset(self._arrays(), batch_count=3)
        trainer = TrainingClass.__new__(TrainingClass)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pth_dir = root / "pth"
            pth_dir.mkdir()
            self._configure_offload(trainer, dataset, root / "cache")
            trainer.pth_dir = str(pth_dir)
            trainer.epoch = 1
            trainer.n_epochs = 2
            trainer.latest_checkpoint_epoch = 0
            trainer.latest_checkpoint_batch = None
            trainer.epoch_start_callback = None
            trainer.epoch_complete_callback = None

            def validate(active_trainer, epoch, training_metrics):
                self.assertIs(active_trainer, trainer)
                self.assertEqual(epoch, 0)
                self.assertFalse(hasattr(trainer, "dataloader"))
                self.assertEqual(dataset.noisy_imgs, [])
                events.append("validation")
                return {"loss_total": 0.25, "loss_l1": 0.2, "loss_l2": 0.3}

            trainer.epoch_end_callback = validate
            original_suspend = trainer._suspend_training_data_for_validation
            original_restore = trainer._restore_training_data_after_validation
            trainer._suspend_training_data_for_validation = lambda reload: (
                events.append(f"suspend-{reload}") or original_suspend(reload)
            )
            trainer._restore_training_data_after_validation = lambda: (
                events.append("restore") or original_restore()
            )
            trainer._write_pending_validation(
                0,
                {"loss_total": 0.5, "loss_l1": 0.4, "loss_l2": 0.6},
            )

            with mock.patch.object(
                TRAINING_MODULE,
                "tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ):
                trainer._resume_pending_validation()

            self.assertFalse(trainer._pending_validation_path().exists())

        self.assertEqual(events, ["suspend-True", "validation", "restore"])
        self.assertEqual(trainer.epoch, 1)
        self.assertTrue(hasattr(trainer, "dataloader"))
        self.assertEqual(len(dataset.noisy_imgs), len(self._arrays()))

    def test_validation_failure_does_not_reload_and_retains_pending_state(self) -> None:
        events = []
        dataset = TinyDataset(self._arrays())
        trainer = TrainingClass.__new__(TrainingClass)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pth_dir = root / "pth"
            pth_dir.mkdir()
            self._configure_lifecycle(trainer, dataset, root / "cache", pth_dir)

            def train() -> None:
                trainer.lh_L = [0.5]
                trainer.lh_L1 = [0.4]
                trainer.lh_L2 = [0.6]

            def save_checkpoint(epoch) -> None:
                model_path = pth_dir / f"model_{epoch}.pth"
                model_path.write_bytes(b"model")
                trainer.model_path = str(model_path)
                trainer.latest_checkpoint_epoch = epoch
                trainer.latest_checkpoint_batch = None

            def fail_validation(*args):
                self.assertFalse(hasattr(trainer, "dataloader"))
                self.assertEqual(dataset.noisy_imgs, [])
                events.append("validation")
                raise RuntimeError("validation failed")

            original_restore = trainer._restore_training_data_after_validation

            def unexpected_restore() -> bool:
                events.append("restore")
                return original_restore()

            trainer.train = train
            trainer.save_checkpoint = save_checkpoint
            trainer.epoch_end_callback = fail_validation
            trainer._restore_training_data_after_validation = unexpected_restore

            with mock.patch.object(TRAINING_MODULE, "plotASCII"), mock.patch.object(
                TRAINING_MODULE,
                "tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ):
                with self.assertRaisesRegex(RuntimeError, "validation failed"):
                    trainer.test()

            pending = trainer._read_pending_validation()
            self.assertEqual(pending["epoch"], 0)
            self.assertTrue(Path(trainer.model_path).is_file())

        self.assertEqual(events, ["validation"])
        self.assertFalse(hasattr(trainer, "dataloader"))
        self.assertEqual(dataset.noisy_imgs, [])

    def test_load_to_memory_false_is_a_no_op(self) -> None:
        dataset = TinyDataset(self._arrays())
        trainer = TrainingClass.__new__(TrainingClass)

        with tempfile.TemporaryDirectory() as temporary_directory:
            self._configure_offload(trainer, dataset, Path(temporary_directory))
            trainer.load_to_memory = False
            original_dataloader = trainer.dataloader
            original_arrays = tuple(dataset.noisy_imgs)

            self.assertFalse(trainer._suspend_training_data_for_validation(True))
            self.assertIs(trainer.dataloader, original_dataloader)
            self.assertEqual(tuple(dataset.noisy_imgs), original_arrays)
            self.assertFalse(trainer._restore_training_data_after_validation())
            self.assertEqual(trainer._training_cache_files, ())
            self.assertIsNone(trainer._training_cache_root)

    def test_cache_write_failure_keeps_resident_data_and_removes_partial_cache(self) -> None:
        dataset = TinyDataset(self._arrays())
        expected = [array.copy() for array in dataset.noisy_imgs]
        trainer = TrainingClass.__new__(TrainingClass)

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_base = Path(temporary_directory)
            self._configure_offload(trainer, dataset, cache_base)
            with mock.patch.object(
                TRAINING_MODULE,
                "tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ), mock.patch.object(
                TRAINING_MODULE.np,
                "save",
                side_effect=OSError("scratch write failed"),
            ):
                with self.assertRaisesRegex(OSError, "scratch write failed"):
                    trainer._ensure_training_cache(dataset)

            self.assertEqual(list(cache_base.iterdir()), [])
            self.assertIsNone(trainer._training_cache_root)
            self.assertEqual(trainer._training_cache_files, ())
            for resident, original in zip(dataset.noisy_imgs, expected):
                np.testing.assert_array_equal(resident, original)

    def test_missing_cache_file_is_rebuilt_before_resident_data_is_released(self) -> None:
        dataset = TinyDataset(self._arrays())
        expected = [array.copy() for array in dataset.noisy_imgs]
        trainer = TrainingClass.__new__(TrainingClass)

        with tempfile.TemporaryDirectory() as temporary_directory:
            self._configure_offload(trainer, dataset, Path(temporary_directory))
            with mock.patch.object(
                TRAINING_MODULE,
                "tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ):
                trainer._ensure_training_cache(dataset)
                original_root = Path(trainer._training_cache_root)
                Path(trainer._training_cache_files[0]["path"]).unlink()
                trainer._suspend_training_data_for_validation(True)
                rebuilt_root = Path(trainer._training_cache_root)
                self.assertNotEqual(original_root, rebuilt_root)
                self.assertFalse(original_root.exists())
                trainer._restore_training_data_after_validation()

            for restored, original in zip(dataset.noisy_imgs, expected):
                np.testing.assert_array_equal(restored, original)

    def test_same_size_cache_corruption_is_rejected_by_checksum(self) -> None:
        dataset = TinyDataset(self._arrays())
        trainer = TrainingClass.__new__(TrainingClass)

        with tempfile.TemporaryDirectory() as temporary_directory:
            self._configure_offload(trainer, dataset, Path(temporary_directory))
            with mock.patch.object(
                TRAINING_MODULE,
                "tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ):
                trainer._ensure_training_cache(dataset)
                cache_path = Path(trainer._training_cache_files[0]["path"])
                with cache_path.open("r+b") as handle:
                    handle.seek(-1, 2)
                    original_byte = handle.read(1)
                    handle.seek(-1, 2)
                    handle.write(bytes([original_byte[0] ^ 0xFF]))
                trainer._suspend_training_data_for_validation(True)
                with self.assertRaisesRegex(RuntimeError, "checksum failed"):
                    trainer._restore_training_data_after_validation()

            self.assertEqual(dataset.noisy_imgs, [])


if __name__ == "__main__":
    unittest.main()
