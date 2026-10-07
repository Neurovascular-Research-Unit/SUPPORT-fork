"""Tests for simultaneous ordinary-mean and nanmean loss logging."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.log_methods import LOSS_CSV_FIELDS, LogManager


def load_validation_module():
    """Load validation without importing the optional PyTorch dependency."""
    test_collection_module = types.ModuleType("utils.test_collection")
    test_collection_module.TestingClass = object
    module_path = Path(__file__).resolve().parents[1] / "utils" / "validation.py"
    spec = importlib.util.spec_from_file_location(
        "_support_validation_loss_logging_under_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("utils.test_collection")
    sys.modules["utils.test_collection"] = test_collection_module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("utils.test_collection", None)
        else:
            sys.modules["utils.test_collection"] = previous
    return module


ValidationTracker = load_validation_module().ValidationTracker


class LossLoggingTests(unittest.TestCase):
    @staticmethod
    def _close_logger(manager: LogManager) -> None:
        for handler in list(manager.logger.handlers):
            handler.close()
            manager.logger.removeHandler(handler)

    def test_epoch_and_batch_logs_contain_mean_and_nanmean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = LogManager(
                SimpleNamespace(out_dir=temporary_directory),
                out_dir=temporary_directory,
                filename_prefix="experiment",
            )
            try:
                manager.append_batch_loss(
                    epoch=0,
                    batch=2,
                    total_batches=4,
                    L1_mean=float("nan"),
                    L2_mean=float("nan"),
                    L_mean=float("nan"),
                    L1_nanmean=1.0,
                    L2_nanmean=2.0,
                    L_nanmean=1.5,
                )
                manager.append_loss(
                    epoch=0,
                    loss_total=float("nan"),
                    loss_l1=float("nan"),
                    loss_l2=float("nan"),
                    loss_total_nanmean=1.5,
                    loss_l1_nanmean=1.0,
                    loss_l2_nanmean=2.0,
                    extra={"validation": {"loss_total": 0.25}},
                )

                with manager.json_path.open("r", encoding="utf-8") as file:
                    record = json.load(file)["losses"][-1]
                with manager.loss_csv.open("r", newline="", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    rows = list(reader)
                    fieldnames = reader.fieldnames
                log_text = manager.log_path.read_text(encoding="utf-8")
            finally:
                self._close_logger(manager)

        self.assertEqual(fieldnames, list(LOSS_CSV_FIELDS))
        self.assertEqual(record["loss_total_nanmean"], 1.5)
        self.assertEqual(rows[-1]["loss_total_nanmean"], "1.5")
        self.assertIn("L_mean=nan", log_text)
        self.assertIn("L_nanmean=1.5", log_text)
        self.assertIn("mean(loss_total=nan", log_text)
        self.assertIn("nanmean(loss_total=1.5", log_text)

    def test_append_loss_upgrades_legacy_csv_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = LogManager(
                SimpleNamespace(out_dir=temporary_directory),
                out_dir=temporary_directory,
                filename_prefix="experiment",
            )
            manager.loss_csv.write_text(
                "epoch,time,loss_total,loss_l1,loss_l2\n"
                "0,earlier,0.5,0.4,0.6\n",
                encoding="utf-8",
            )
            try:
                manager.append_loss(
                    epoch=1,
                    loss_total=float("nan"),
                    loss_l1=float("nan"),
                    loss_l2=float("nan"),
                    loss_total_nanmean=0.3,
                    loss_l1_nanmean=0.2,
                    loss_l2_nanmean=0.4,
                )
                with manager.loss_csv.open("r", newline="", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    rows = list(reader)
                    fieldnames = reader.fieldnames
            finally:
                self._close_logger(manager)

        self.assertEqual(fieldnames, list(LOSS_CSV_FIELDS))
        self.assertEqual(rows[0]["loss_total_nanmean"], "0.5")
        self.assertEqual(rows[1]["loss_total_nanmean"], "0.3")

    def test_append_loss_preserves_legacy_positional_extra_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = LogManager(
                SimpleNamespace(out_dir=temporary_directory),
                out_dir=temporary_directory,
                filename_prefix="experiment",
            )
            try:
                manager.append_loss(
                    0,
                    0.5,
                    0.4,
                    0.6,
                    {"source": "legacy positional call"},
                )
                with manager.json_path.open("r", encoding="utf-8") as file:
                    record = json.load(file)["losses"][-1]
            finally:
                self._close_logger(manager)

        self.assertEqual(record["extra"], {"source": "legacy positional call"})
        self.assertIsNone(record["loss_total_nanmean"])

    def test_validation_history_upgrades_legacy_training_loss_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            history_path = output_directory / "experiment_train_validation_losses.csv"
            history_path.write_text(
                "epoch,training_loss,training_l1,training_l2,"
                "validation_loss,validation_l1,validation_l2,"
                "validation_elements\n"
                "0,0.5,0.4,0.6,0.25,0.2,0.3,100\n",
                encoding="utf-8",
            )
            tracker = ValidationTracker.__new__(ValidationTracker)
            trainer = SimpleNamespace(
                out_dir=temporary_directory,
                exp_name="experiment",
            )

            tracker._record_history(
                trainer,
                epoch=1,
                training_metrics={
                    "loss_total": float("nan"),
                    "loss_l1": float("nan"),
                    "loss_l2": float("nan"),
                    "loss_total_nanmean": 0.3,
                    "loss_l1_nanmean": 0.2,
                    "loss_l2_nanmean": 0.4,
                },
                validation_metrics={
                    "loss_total": 0.2,
                    "loss_l1": 0.1,
                    "loss_l2": 0.3,
                    "element_count": 100,
                    "files": [],
                },
            )

            with history_path.open("r", newline="", encoding="utf-8") as file:
                fieldnames = csv.DictReader(file).fieldnames

        self.assertEqual(fieldnames, list(ValidationTracker.history_fields))
        self.assertEqual(tracker.history[0]["training_loss_nanmean"], 0.5)
        self.assertEqual(tracker.history[1]["training_loss_nanmean"], 0.3)


if __name__ == "__main__":
    unittest.main()
