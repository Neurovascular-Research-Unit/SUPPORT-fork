import csv
import json
import os
import logging
import stat
import tempfile
from datetime import datetime
from pathlib import Path


LOSS_CSV_FIELDS = (
    "epoch",
    "time",
    "loss_total",
    "loss_l1",
    "loss_l2",
    "loss_total_nanmean",
    "loss_l1_nanmean",
    "loss_l2_nanmean",
)
LEGACY_LOSS_CSV_FIELDS = LOSS_CSV_FIELDS[:5]

def _safe_serialize(v):
    try:
        json.dumps(v)
        return v
    except Exception:
        try:
            return repr(v)
        except Exception:
            return str(type(v))

class LogManager:
    def __init__(self, obj, mode="train", out_dir=None, filename_prefix="run"):
        """
        obj: the TrainingClass or TestingClass instance
        mode: "train" or "test"
        """
        self.obj = obj
        self.mode = mode
        self.out_dir = Path(out_dir) if out_dir is not None else (Path(obj.out_dir) if getattr(obj, "out_dir", None) else Path.cwd() / "outputs")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = filename_prefix
        self.log_path = self.out_dir / f"{self.prefix}_{self.mode}.log"
        self.json_path = self.out_dir / f"{self.prefix}_{self.mode}.json"
        self.loss_csv = self.out_dir / f"{self.prefix}_{self.mode}_losses.csv"
        self._setup_logger()
        self._init_json()

    def _setup_logger(self):
        resolved_log_path = self.log_path.resolve()
        self.logger = logging.getLogger(
            f"support_{self.mode}:{resolved_log_path}"
        )
        self.logger.setLevel(logging.INFO)
        if not any(
            isinstance(h, logging.FileHandler)
            and h.baseFilename == str(resolved_log_path)
            for h in self.logger.handlers
        ):
            fh = logging.FileHandler(resolved_log_path, mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self.logger.addHandler(fh)

    def _init_json(self):
        if self.json_path.exists():
            try:
                with open(self.json_path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            self.data = {}
        if "params" not in self.data:
            self.data["params"] = {}
        if "losses" not in self.data:
            self.data["losses"] = []

    def log_params(self):
        # capture timestamp and all public attributes of the object (non-callable)
        params = {}
        for k, v in vars(self.obj).items():
            if k.startswith("_"):
                continue
            if callable(v):
                continue
            params[k] = _safe_serialize(v)
        params["_logged_at"] = datetime.now().isoformat()
        self.data["params"] = params
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)
        self.logger.info("Logged params")
        # also write a human-readable header into the .log
        self.logger.info("=== RUN PARAMETERS ===")
        for k, v in params.items():
            self.logger.info(f"{k}: {v}")

    def _append_loss_csv(self, rec):
        """Append a loss row, upgrading the legacy five-column CSV if needed."""
        existing_rows = []
        if self.loss_csv.is_file():
            with open(self.loss_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames == list(LOSS_CSV_FIELDS):
                    with open(self.loss_csv, "a", newline="") as output:
                        csv.DictWriter(
                            output,
                            fieldnames=LOSS_CSV_FIELDS,
                            extrasaction="ignore",
                        ).writerow(rec)
                    return
                if reader.fieldnames != list(LEGACY_LOSS_CSV_FIELDS):
                    raise ValueError(
                        f"Unexpected training loss CSV schema in {self.loss_csv}"
                    )
                existing_rows = list(reader)

        if self.loss_csv.exists():
            output_mode = stat.S_IMODE(self.loss_csv.stat().st_mode)
        else:
            current_umask = os.umask(0)
            os.umask(current_umask)
            output_mode = 0o666 & ~current_umask
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.loss_csv.name}.",
            suffix=".tmp",
            dir=self.loss_csv.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            with temporary_path.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=LOSS_CSV_FIELDS,
                    extrasaction="ignore",
                )
                writer.writeheader()
                for row in existing_rows:
                    # Historical nanmeans cannot be recovered from a NaN mean.
                    # Copying the legacy value is exact whenever it was finite
                    # and remains conservatively NaN otherwise.
                    row["loss_total_nanmean"] = row["loss_total"]
                    row["loss_l1_nanmean"] = row["loss_l1"]
                    row["loss_l2_nanmean"] = row["loss_l2"]
                    writer.writerow(row)
                writer.writerow(rec)
            os.chmod(temporary_path, output_mode)
            os.replace(temporary_path, self.loss_csv)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def append_loss(
        self,
        epoch,
        loss_total=None,
        loss_l1=None,
        loss_l2=None,
        extra=None,
        *,
        loss_total_nanmean=None,
        loss_l1_nanmean=None,
        loss_l2_nanmean=None,
    ):
        rec = {
            "epoch": int(epoch),
            "time": datetime.now().isoformat(),
            "loss_total": _safe_serialize(loss_total),
            "loss_l1": _safe_serialize(loss_l1),
            "loss_l2": _safe_serialize(loss_l2),
            "loss_total_nanmean": _safe_serialize(loss_total_nanmean),
            "loss_l1_nanmean": _safe_serialize(loss_l1_nanmean),
            "loss_l2_nanmean": _safe_serialize(loss_l2_nanmean),
        }
        if extra is not None:
            rec["extra"] = _safe_serialize(extra)
        self.data.setdefault("losses", []).append(rec)
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)
        # append to CSV for quick plotting
        self._append_loss_csv(rec)
        self.logger.info(
            f"Epoch {epoch}: "
            f"mean(loss_total={loss_total} l1={loss_l1} l2={loss_l2}) "
            f"nanmean(loss_total={loss_total_nanmean} "
            f"l1={loss_l1_nanmean} l2={loss_l2_nanmean})"
        )

    def append_batch_loss(
        self,
        epoch,
        batch,
        total_batches,
        L1_mean,
        L2_mean,
        L_mean,
        L1_nanmean=None,
        L2_nanmean=None,
        L_nanmean=None,
    ):
        """Write both generalized-training running loss means to the log."""
        self.logger.info(
            f"Epoch {epoch} Batch {batch}/{total_batches}: "
            f"L1_mean={L1_mean} L2_mean={L2_mean} L_mean={L_mean} "
            f"L1_nanmean={L1_nanmean} L2_nanmean={L2_nanmean} "
            f"L_nanmean={L_nanmean}"
        )

    def log_diagnostic(self, text):
        self.logger.info(text)
        # also keep a diagnostics array
        self.data.setdefault("diagnostics", []).append({"time": datetime.now().isoformat(), "text": _safe_serialize(text)})
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)
