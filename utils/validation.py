import csv
import os
import stat
import tempfile
from pathlib import Path

try:
    from .test_collection import TestingClass
except ImportError:  # Support direct execution from the repository root.
    from utils.test_collection import TestingClass


TIFF_SUFFIXES = {".tif", ".tiff"}


def discover_tiffs(test_dir):
    """Return validation TIFFs from a file or recursive directory search."""
    test_path = Path(test_dir).expanduser()
    if test_path.is_file():
        if test_path.suffix.lower() not in TIFF_SUFFIXES:
            raise ValueError(f"Validation input is not a TIFF file: {test_path}")
        return [test_path]
    if not test_path.is_dir():
        raise FileNotFoundError(f"Validation path does not exist: {test_path}")

    paths = sorted(
        path
        for path in test_path.rglob("*")
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(
            f"No TIFF files were found recursively in {test_path}."
        )
    return paths


def _as_bool(value, name):
    if type(value) is bool:
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise TypeError(f"{name} must be True or False")


class ValidationTracker:
    """Run held-out inference and record the self-supervised loss each epoch."""

    history_fields = (
        "epoch",
        "training_loss",
        "training_l1",
        "training_l2",
        "training_loss_nanmean",
        "training_l1_nanmean",
        "training_l2_nanmean",
        "validation_loss",
        "validation_l1",
        "validation_l2",
        "validation_elements",
    )
    per_file_fields = (
        "epoch",
        "validation_file",
        "output_file",
        "validation_loss",
        "validation_l1",
        "validation_l2",
        "validation_elements",
    )

    def __init__(
        self, test_dir, test_params, output_dir=None, validation_paths=None
    ):
        self.test_root = Path(test_dir).expanduser().resolve()
        if validation_paths is None:
            self.validation_paths = discover_tiffs(self.test_root)
        else:
            self.validation_paths = [
                Path(path).expanduser().resolve() for path in validation_paths
            ]
            if not self.validation_paths:
                raise ValueError("At least one frozen validation TIFF is required.")
            for path in self.validation_paths:
                if not path.is_file() or path.suffix.lower() not in TIFF_SUFFIXES:
                    raise ValueError(f"Frozen validation input is not a TIFF file: {path}")
        self.test_params = dict(test_params)
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else None
        )
        self._output_relative_paths = self._build_output_relative_paths()
        self.history = []
        self.history_csv = None
        self.per_file_csv = None

    def _relative_path(self, path):
        if self.test_root.is_dir():
            return path.relative_to(self.test_root)
        return Path(path.name)

    @staticmethod
    def _default_output_name(path):
        stem = path.stem
        if stem.lower().endswith("_sprt"):
            stem = stem[:-5]
        return f"{stem}_SPRT.tif"

    def _build_output_relative_paths(self):
        """Disambiguate inputs that would otherwise share one output file."""
        candidate_groups = {}
        for path in self.validation_paths:
            relative_path = self._relative_path(path)
            candidate = relative_path.with_name(
                self._default_output_name(relative_path)
            )
            candidate_groups.setdefault(candidate.as_posix().casefold(), []).append(
                path
            )

        output_paths = {}
        reserved = set()
        for paths in candidate_groups.values():
            if len(paths) != 1:
                continue
            path = paths[0]
            relative_path = self._relative_path(path)
            output_path = relative_path.with_name(
                self._default_output_name(relative_path)
            )
            output_paths[path] = output_path
            reserved.add(output_path.as_posix().casefold())

        for paths in candidate_groups.values():
            if len(paths) == 1:
                continue
            for path in paths:
                relative_path = self._relative_path(path)
                suffix_label = path.suffix.lower().lstrip(".") or "tiff"
                base_name = f"{path.stem}_{suffix_label}_SPRT"
                output_path = relative_path.with_name(f"{base_name}.tif")
                counter = 2
                while output_path.as_posix().casefold() in reserved:
                    output_path = relative_path.with_name(
                        f"{base_name}_{counter}.tif"
                    )
                    counter += 1
                output_paths[path] = output_path
                reserved.add(output_path.as_posix().casefold())
        return output_paths

    def _validation_output_root(self, trainer):
        if self.output_dir is None:
            self.output_dir = Path(trainer.out_dir).resolve() / "validation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def _tester_params(self, trainer, path, output_path):
        params = {
            "in_path": str(path),
            "exp_name": path.stem,
            "out_dir": str(output_path.parent),
            "output_path": str(output_path),
            "dimensionality": trainer.dimensionality,
            "cuda_device": trainer.cuda_device,
            "patch_size": trainer.patch_size,
            "depth": trainer.depth,
            "bc_channels": trainer.bc_channels,
            "obo_channels": trainer.obo_channels,
            "ll_channels": trainer.ll_channels,
            "unet_channels": trainer.unet_channels,
            "bs_size": trainer.bs_size,
            "bp": trainer.bp,
            "loss_coef": trainer.loss_coef,
            "use_cpu": trainer.use_cpu,
            "track_loss": True,
            "normalization_mean": getattr(trainer, "normalization_mean", None),
            "normalization_std": getattr(trainer, "normalization_std", None),
        }
        for name in (
            "patch_interval",
            "batch_size",
            "pin_memory",
            "n_cpu",
            "prefetch_factor",
        ):
            if name in self.test_params:
                params[name] = self.test_params[name]
        if "pad" in self.test_params:
            params["pad"] = _as_bool(self.test_params["pad"], "pad")
        return params

    @staticmethod
    def _upsert_csv(
        path,
        fieldnames,
        rows,
        key_fields,
        legacy_fieldnames=None,
        legacy_field_sources=None,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = list(rows)
        if path.exists():
            output_mode = stat.S_IMODE(path.stat().st_mode)
        else:
            current_umask = os.umask(0)
            os.umask(current_umask)
            output_mode = 0o666 & ~current_umask
        new_keys = {
            tuple(str(row[name]) for name in key_fields)
            for row in rows
        }
        retained_rows = []
        if path.is_file():
            with path.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != list(fieldnames):
                    if reader.fieldnames != list(legacy_fieldnames or ()):
                        raise ValueError(
                            f"Unexpected validation CSV schema in {path}"
                        )
                    existing_rows = list(reader)
                    for row in existing_rows:
                        for destination, source in (
                            legacy_field_sources or {}
                        ).items():
                            row[destination] = row[source]
                else:
                    existing_rows = list(reader)
                retained_rows = [
                    row
                    for row in existing_rows
                    if tuple(row[name] for name in key_fields) not in new_keys
                ]

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            output_rows = retained_rows + rows
            if "epoch" in fieldnames:
                output_rows.sort(
                    key=lambda row: (
                        int(row["epoch"]),
                        *(str(row[name]) for name in key_fields if name != "epoch"),
                    )
                )
            with temporary_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in output_rows:
                    writer.writerow({name: row[name] for name in fieldnames})
            os.chmod(temporary_path, output_mode)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _record_history(self, trainer, epoch, training_metrics, validation_metrics):
        history_root = Path(trainer.out_dir).resolve()
        self.history_csv = history_root / (
            f"{trainer.exp_name}_train_validation_losses.csv"
        )
        self.per_file_csv = history_root / (
            f"{trainer.exp_name}_validation_files.csv"
        )
        record = {
            "epoch": int(epoch),
            "training_loss": training_metrics["loss_total"],
            "training_l1": training_metrics["loss_l1"],
            "training_l2": training_metrics["loss_l2"],
            "training_loss_nanmean": training_metrics.get(
                "loss_total_nanmean", training_metrics["loss_total"]
            ),
            "training_l1_nanmean": training_metrics.get(
                "loss_l1_nanmean", training_metrics["loss_l1"]
            ),
            "training_l2_nanmean": training_metrics.get(
                "loss_l2_nanmean", training_metrics["loss_l2"]
            ),
            "validation_loss": validation_metrics["loss_total"],
            "validation_l1": validation_metrics["loss_l1"],
            "validation_l2": validation_metrics["loss_l2"],
            "validation_elements": validation_metrics["element_count"],
        }
        self._upsert_csv(
            self.history_csv,
            self.history_fields,
            [record],
            key_fields=("epoch",),
            legacy_fieldnames=tuple(
                name
                for name in self.history_fields
                if not name.endswith("_nanmean")
            ),
            legacy_field_sources={
                "training_loss_nanmean": "training_loss",
                "training_l1_nanmean": "training_l1",
                "training_l2_nanmean": "training_l2",
            },
        )

        file_records = []
        for file_metrics in validation_metrics["files"]:
            file_records.append(
                {
                    "epoch": int(epoch),
                    "validation_file": file_metrics["path"],
                    "output_file": file_metrics["output_path"],
                    "validation_loss": file_metrics["loss_total"],
                    "validation_l1": file_metrics["loss_l1"],
                    "validation_l2": file_metrics["loss_l2"],
                    "validation_elements": file_metrics["element_count"],
                }
            )
        self._upsert_csv(
            self.per_file_csv,
            self.per_file_fields,
            file_records,
            key_fields=("epoch",),
        )
        with self.history_csv.open("r", newline="", encoding="utf-8") as file:
            self.history = []
            for row in csv.DictReader(file):
                self.history.append(
                    {
                        "epoch": int(row["epoch"]),
                        "training_loss": float(row["training_loss"]),
                        "training_l1": float(row["training_l1"]),
                        "training_l2": float(row["training_l2"]),
                        "training_loss_nanmean": float(
                            row["training_loss_nanmean"]
                        ),
                        "training_l1_nanmean": float(
                            row["training_l1_nanmean"]
                        ),
                        "training_l2_nanmean": float(
                            row["training_l2_nanmean"]
                        ),
                        "validation_loss": float(row["validation_loss"]),
                        "validation_l1": float(row["validation_l1"]),
                        "validation_l2": float(row["validation_l2"]),
                        "validation_elements": int(row["validation_elements"]),
                    }
                )

    def __call__(self, trainer, epoch, training_metrics):
        """Validate an epoch and return JSON-serializable metrics."""
        output_root = self._validation_output_root(trainer)
        file_metrics = []
        loss_l1_sum = 0.0
        loss_l2_sum = 0.0
        element_count = 0

        for path in self.validation_paths:
            relative_path = self._relative_path(path)
            output_path = output_root / self._output_relative_paths[path]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tester = TestingClass(
                self._tester_params(trainer, path, output_path)
            )
            try:
                metrics = tester.run(model=trainer.local_model)
                output_path = str(Path(tester.out_path).resolve())
            finally:
                tester.release_resources()

            loss_l1_sum += metrics["loss_l1_sum"]
            loss_l2_sum += metrics["loss_l2_sum"]
            element_count += metrics["element_count"]
            file_metrics.append(
                {
                    "path": relative_path.as_posix(),
                    "output_path": output_path,
                    "loss_total": metrics["loss_total"],
                    "loss_l1": metrics["loss_l1"],
                    "loss_l2": metrics["loss_l2"],
                    "element_count": metrics["element_count"],
                }
            )

        loss_l1 = loss_l1_sum / element_count
        loss_l2 = loss_l2_sum / element_count
        validation_metrics = {
            "loss_total": (
                trainer.loss_coef[0] * loss_l1
                + trainer.loss_coef[1] * loss_l2
            ),
            "loss_l1": loss_l1,
            "loss_l2": loss_l2,
            "element_count": element_count,
            "files": file_metrics,
        }
        self._record_history(
            trainer,
            epoch,
            training_metrics,
            validation_metrics,
        )
        print(
            f"Epoch {epoch}: training loss "
            f"mean={training_metrics['loss_total']:.6g}, "
            "nanmean="
            f"{training_metrics.get('loss_total_nanmean', training_metrics['loss_total']):.6g}, "
            f"validation loss={validation_metrics['loss_total']:.6g}"
        )
        return validation_metrics
