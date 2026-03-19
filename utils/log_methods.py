import json
import os
import logging
from datetime import datetime
from pathlib import Path

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
        self.logger = logging.getLogger(f"support_{self.mode}")
        self.logger.setLevel(logging.INFO)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(self.log_path) for h in self.logger.handlers):
            fh = logging.FileHandler(self.log_path, mode="a")
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

    def append_loss(self, epoch, loss_total=None, loss_l1=None, loss_l2=None, extra=None):
        rec = {
            "epoch": int(epoch),
            "time": datetime.now().isoformat(),
            "loss_total": _safe_serialize(loss_total),
            "loss_l1": _safe_serialize(loss_l1),
            "loss_l2": _safe_serialize(loss_l2),
        }
        if extra is not None:
            rec["extra"] = _safe_serialize(extra)
        self.data.setdefault("losses", []).append(rec)
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)
        # append to CSV for quick plotting
        header = not self.loss_csv.exists()
        with open(self.loss_csv, "a") as f:
            if header:
                f.write("epoch,time,loss_total,loss_l1,loss_l2\n")
            f.write(f'{rec["epoch"]},{rec["time"]},{rec["loss_total"]},{rec["loss_l1"]},{rec["loss_l2"]}\n')
        self.logger.info(f"Epoch {epoch}: loss_total={loss_total} l1={loss_l1} l2={loss_l2}")

    def log_diagnostic(self, text):
        self.logger.info(text)
        # also keep a diagnostics array
        self.data.setdefault("diagnostics", []).append({"time": datetime.now().isoformat(), "text": _safe_serialize(text)})
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)