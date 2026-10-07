import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import numpy as np
import tifffile
import torch
from support.SUPPORT import SUPPORT

import time
from tqdm import tqdm
import random
from utils.log_methods import LogManager
from utils.dataset import random_transform, DatasetSUPPORT
from torch.utils.data import DataLoader
from utils.loss_plot import plotASCII
from IPython.display import clear_output


def _mean_and_nanmean(values):
    """Return the ordinary mean and the NaN-ignoring mean of loss values."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(array))
    nanmean = (
        float("nan")
        if np.isnan(array).all()
        else float(np.nanmean(array))
    )
    return mean, nanmean


class TrainingClass():
    """
    Class implementing training process.
    """
    
    def __init__(self, params_dict):
        """
        Constructor class for training process

        Args:
           params_dict: dict
               The collection of training params set by users
        Returns:
           self
        """
        # EXPERIMENT
        self.in_path = ""
        self.out_dir = None
        self.exp_name = None
        self.dimensionality = None
        self.rm_dir = True
        self.n_epochs = 500
        self.epoch = 0
        self.checkpoints = False
        self.resume = False
        self.rm_intermediaries = False
        self.train_gen = False
        # Permanent epoch snapshots; generalized quartile checkpoints roll
        # independently whenever checkpointing is enabled.
        self.save_every_epoch = False
        self.epoch_end_callback = None
        # Optional, noncritical lifecycle hooks (for example Slack updates).
        # Validation remains on epoch_end_callback, but a resumable checkpoint
        # is committed before that callback runs.
        self.epoch_start_callback = None
        self.epoch_complete_callback = None
        self.rnd_seed = 0
        self.cuda_device = [0]
        
        # DATASET
        self.patch_size = [61, 128, 128]
        self.patch_interval = [1, 64, 64]
        self.batch_size = 16
        self.load_to_memory = True
        self.pin_memory = True
        # Generalized validation can need enough host RAM for another full TIFF.
        # Retain an exact normalized copy on job-local scratch and drop the
        # resident training tensors while the validation callback runs.
        self.offload_training_data_during_validation = True
        self.training_cache_dir = os.environ.get("SUPPORT_TRAINING_CACHE_DIR")
        
        # MODEL
        self.depth = 5
        self.bc_channels = 64
        self.obo_channels = [32, 16]
        self.ll_channels = [64, 32, 16]
        self.bs_size = [3, 3]
        self.bp = False
        self.unet_channels = [64, 128, 256, 512, 1024]
        
        # TRAINING
        self.lr = 5E-4
        self.loss_coef = [0.5, 0.5]
        self.use_amp = False
        self.use_cpu = False
        self.n_cpu = 8
        self.prefetch_factor = 2
        
        # AUTOMATED
        self.cuda = False
        self.device = "cpu"
        self.is_dir = False
        self.model_path = None
        self.latest_checkpoint_epoch = None
        self.latest_checkpoint_batch = None
        self.normalization_scope = "per_stack"
        self.normalization_mean = None
        self.normalization_std = None
        self.normalization_voxel_count = None
        self._training_cache_root = None
        self._training_cache_files = ()
        self._training_cache_owned = False
        self._suspended_training_dataset = None
        self._training_dataloader_batch_count = None
        self._checkpoint_controls_configured = any(
            key in params_dict
            for key in ("checkpoints", "resume", "rm_intermediaries", "save_every_epoch")
        )
        self.set_params(params_dict)
        for option in (
            "checkpoints",
            "resume",
            "rm_intermediaries",
            "train_gen",
            "save_every_epoch",
            "pin_memory",
            "offload_training_data_during_validation",
        ):
            if type(getattr(self, option)) is not bool:
                raise TypeError(f"{option} must be True or False")
        if self.save_every_epoch:
            self.checkpoints = True
        if self.epoch_end_callback is not None and not callable(self.epoch_end_callback):
            raise TypeError("epoch_end_callback must be callable or None")
        for callback_name in ("epoch_start_callback", "epoch_complete_callback"):
            callback = getattr(self, callback_name)
            if callback is not None and not callable(callback):
                raise TypeError(f"{callback_name} must be callable or None")
        if self.train_gen and not self.is_dir:
            raise ValueError("train_gen requires in_path to be a directory")
        self.normalization_scope = "dataset" if self.train_gen else "per_stack"
        self.cuda = torch.cuda.is_available() and not self.use_cpu
        self.device = "cuda" if self.cuda else "cpu"

    def _in_ipykernel(self):
        try:
            from IPython import get_ipython
            shell = get_ipython()
            return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
        except Exception:
            return False

    def _dataloader_kwargs(self, shuffle):
        # Notebook kernels are more prone to worker crashes; fall back to the main process there.
        num_workers = 0 if self._in_ipykernel() else self.n_cpu
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": self.cuda and self.pin_memory,
            # Exhausted epoch iterators shut workers/prefetch queues down before
            # the validation boundary; a fresh iterator is created next epoch.
            "persistent_workers": False,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _run_optional_callback(self, callback_name, *args):
        """Run a noncritical lifecycle callback without risking training."""
        callback = getattr(self, callback_name, None)
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:
            # Do not print exception text: notification exceptions can contain
            # a secret webhook URL.
            print(
                f"WARNING: {callback_name} failed ({type(exc).__name__}); "
                "training will continue.",
                file=sys.stderr,
            )
        
    def vars_dict(self): self.prepare_file()

    def release_resources(self):
        # Release the largest host allocation first so teardown itself does not
        # run out of RAM while moving the comparatively small model off the GPU.
        if hasattr(self, "dataloader"):
            self._shutdown_dataloader_workers(self.dataloader)
            del self.dataloader
        suspended_dataset = getattr(self, "_suspended_training_dataset", None)
        if suspended_dataset is not None:
            suspended_dataset.noisy_imgs = []
            suspended_dataset.precomputed_indices = None
            self._suspended_training_dataset = None
        gc.collect()
        if hasattr(self, "optimizer"):
            self.optimizer.state.clear()
            del self.optimizer
        if hasattr(self, "local_model"):
            self.local_model.to("cpu")
            del self.local_model
        if hasattr(self, "scaler"):
            del self.scaler
        if hasattr(self, "Tensor"):
            del self.Tensor
        if hasattr(self, "rng"):
            del self.rng
        self._remove_training_cache()
        gc.collect()
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
    def run(self):
        """
        General function for training SUPPORT network.
        """
        self.prepare_file()
        self.generate_dataloader()
        self.initialize_network()
        self.initialize_optimizer()
        self.load_checkpoint()
        try:
            self._prepare_training_cache_for_run()
            self.test()
        except BaseException:
            # Integrations also call release_resources(), but keep direct class
            # callers from orphaning a large job-owned scratch tree on failure.
            self._remove_training_cache()
            raise

    def set_params(self, params_dict):
        # This is where I would implement overlap factor
        print('\033[1;31mTraining parameters -----> \033[0m')
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            print("{:30s} {:s} ".format(key, str(value)))
        if Path(self.in_path).is_dir(): self.is_dir = True

    def prepare_file(self):
        if getattr(self, "_files_prepared", False):
            return
        if self.exp_name is None: self.exp_name = Path(self.in_path).stem
        self.cwd = str(Path("./").absolute())
        if self.out_dir is None: self.out_dir = self.cwd + '/outputs'
        if self.rm_dir:
            self.tmp_dir = self.cwd + f'/tmp/{self.exp_name}'
            if not Path(self.tmp_dir).exists(): Path(self.tmp_dir).mkdir(parents=True)
            self.pth_dir = self.tmp_dir + '/pth'
            self.onnx_dir = self.tmp_dir + '/onnx'
        else:
            self.out_dir += f'/{self.exp_name}'
            self.pth_dir = self.out_dir + '/pth'
            self.onnx_dir = self.out_dir + '/onnx'
        if not Path(self.out_dir).exists(): Path(self.out_dir).mkdir(parents=True)
        if not Path(self.pth_dir).exists(): Path(self.pth_dir).mkdir(parents=True)
        if not Path(self.onnx_dir).exists(): Path(self.onnx_dir).mkdir(parents=True)
        try:
            self.log = LogManager(self, mode="train", out_dir=self.out_dir, filename_prefix=self.exp_name)
            self.log.log_params()
        except Exception:
            if self.train_gen:
                raise
        self._files_prepared = True

    def _read_stack(self, path):
        if not self.load_to_memory:
            try:
                return tifffile.memmap(path)
            except Exception:
                pass
        noisy_img = tifffile.TiffFile(path).asarray()
        if self.load_to_memory:
            return np.asarray(noisy_img, dtype=np.float32)
        return np.asarray(noisy_img)

    def _prepare_training_stack(self, noisy_img):
        if self.load_to_memory:
            return torch.from_numpy(np.ascontiguousarray(noisy_img, dtype=np.float32)).type(torch.FloatTensor)
        return noisy_img

    def _describe_stack(self, noisy_shape):
        if self.dimensionality == "3+1D":
            if len(noisy_shape) == 4:
                T, Z, Y, X = noisy_shape
                return f"(X, Y, Z, T) = ({X}, {Y}, {Z}, {T})"
            if len(noisy_shape) == 3:
                T, Y, X = noisy_shape
                return f"(X, Y, T) = ({X}, {Y}, {T})"
            raise ValueError(f"Expected a 3D or 4D TIFF for 3+1D input, received shape {noisy_shape}.")

        if self.dimensionality == "2+1D":
            if len(noisy_shape) != 3:
                raise ValueError(f"Expected a 3D TIFF for 2+1D input, received shape {noisy_shape}.")
            T, Y, X = noisy_shape
            return f"(X, Y, T) = ({X}, {Y}, {T})"

        if self.dimensionality == "3D":
            if len(noisy_shape) != 3:
                raise ValueError(f"Expected a 3D TIFF for 3D input, received shape {noisy_shape}.")
            Z, Y, X = noisy_shape
            return f"(X, Y, Z) = ({X}, {Y}, {Z})"

        return f"Shape: {noisy_shape}"

    def _prepare_training_stacks(self, noisy_img):
        if self.dimensionality == "3+1D":
            if noisy_img.ndim == 4:
                return [
                    self._prepare_training_stack(noisy_img[:, z_idx, :, :])
                    for z_idx in range(noisy_img.shape[1])
                ]
            if noisy_img.ndim == 3:
                return [self._prepare_training_stack(noisy_img)]
            raise ValueError(f"Expected a 3D or 4D TIFF for 3+1D input, received shape {noisy_img.shape}.")

        if self.dimensionality in ("2+1D", "3D"):
            if noisy_img.ndim != 3:
                raise ValueError(
                    f"Expected a 3D TIFF for {self.dimensionality} input, received shape {noisy_img.shape}."
                )
            return [self._prepare_training_stack(noisy_img)]

        if noisy_img.ndim != 3:
            raise ValueError(
                f"Expected a 3D TIFF for {self.dimensionality} input, received shape {noisy_img.shape}."
            )
        return [self._prepare_training_stack(noisy_img)]

    def generate_dataloader(self):
        """
        
        """
        noisy_imgs = []
        normalization_group_ids = []
        uses_4d_source_normalization = False
        print("\033[1;31mLoaded Files -----> \033[0m")
        frozen_input_paths = getattr(self, "_frozen_input_paths", None)
        if frozen_input_paths is not None:
            noisy_paths = [Path(path) for path in frozen_input_paths]
        elif self.is_dir:
            path_iterator = (
                Path(self.in_path).rglob('*')
                if self.train_gen
                else Path(self.in_path).iterdir()
            )
            noisy_paths = [
                path for path in path_iterator
                if path.is_file() and path.suffix.lower() in ('.tif', '.tiff')
            ]
        else:
            noisy_paths = [Path(self.in_path)]
        noisy_paths = sorted(noisy_paths)
        if len(noisy_paths) == 0:
            raise FileNotFoundError(f'No TIFF files were found in {self.in_path}.')
        if frozen_input_paths is None and self.train_gen:
            # A fallback reload after validation must use the exact original
            # file list rather than discovering additions made mid-run.
            self._frozen_input_paths = tuple(
                str(path.absolute()) for path in noisy_paths
            )
        noisy_stems = [str(path.stem) for path in noisy_paths]
        slen = max((len(s) for s in noisy_stems), default=0)

        for source_index, (path, noisy_stem) in enumerate(zip(noisy_paths, noisy_stems)):
            noisy_img = self._read_stack(path)
            print(f"\033[1m{noisy_stem:<{slen}}\033[0m  {self._describe_stack(noisy_img.shape)}")
            prepared_stacks = self._prepare_training_stacks(noisy_img)
            noisy_imgs.extend(prepared_stacks)
            if not self.train_gen:
                normalization_group_ids.extend(
                    [source_index] * len(prepared_stacks)
                )
                uses_4d_source_normalization = (
                    uses_4d_source_normalization
                    or (self.dimensionality == "3+1D" and noisy_img.ndim == 4)
                )

        dataset = DatasetSUPPORT(
            noisy_imgs=noisy_imgs,
            patch_size=self.patch_size,
            patch_interval=self.patch_interval,
            transform=None,
            random_patch=True,
            load_to_memory=self.load_to_memory,
            normalize_across_images=self.train_gen,
            normalization_group_ids=(
                normalization_group_ids
                if uses_4d_source_normalization
                else None
            ),
        )
        if self.train_gen:
            self.normalization_scope = "dataset"
            self.normalization_mean = float(dataset.dataset_mean.item())
            self.normalization_std = float(dataset.dataset_std.item())
            self.normalization_voxel_count = int(dataset.dataset_count)
            diagnostic = (
                "Global training normalization: "
                f"voxels={self.normalization_voxel_count} "
                f"mean={self.normalization_mean} "
                f"std={self.normalization_std} correction=1"
            )
            print(diagnostic)
            if hasattr(self, "log"):
                self.log.log_diagnostic(diagnostic)
        elif uses_4d_source_normalization:
            diagnostic = (
                "4-D training normalization: one mean/std per source TIFF "
                "across all T, Z, Y, and X voxels."
            )
            print(diagnostic)
            if hasattr(self, "log"):
                self.log.log_diagnostic(diagnostic)
        
        self.dataloader = DataLoader(
            dataset=dataset,
            **self._dataloader_kwargs(shuffle=True),
        )

    @staticmethod
    def _format_byte_count(byte_count):
        value = float(byte_count)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024 or unit == "TiB":
                return f"{value:.2f} {unit}"
            value /= 1024

    @staticmethod
    def _drop_file_cache(file_handle):
        """Best-effort eviction of scratch I/O pages from the Linux page cache."""
        posix_fadvise = getattr(os, "posix_fadvise", None)
        dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
        if posix_fadvise is None or dontneed is None:
            return
        try:
            posix_fadvise(file_handle.fileno(), 0, 0, dontneed)
        except OSError:
            # Correctness does not depend on this performance/memory hint.
            pass

    @staticmethod
    def _array_sha256(array):
        """Hash a contiguous array without allocating another full-size copy."""
        byte_view = memoryview(array).cast("B")
        digest = hashlib.sha256()
        chunk_size = 16 * 1024 * 1024
        for offset in range(0, byte_view.nbytes, chunk_size):
            digest.update(byte_view[offset:offset + chunk_size])
        return digest.hexdigest()

    @staticmethod
    def _cache_metadata_matches_array(metadata, array):
        return (
            tuple(array.shape) == tuple(metadata["shape"])
            and array.dtype.str == metadata["dtype"]
            and int(array.nbytes) == int(metadata["nbytes"])
        )

    def _training_cache_is_valid(self, cache_files, resident_stacks):
        """Cheaply validate scratch files before releasing resident tensors."""
        if len(cache_files) != len(resident_stacks):
            return False
        try:
            for metadata, resident_stack in zip(cache_files, resident_stacks):
                path = Path(metadata["path"])
                path_stat = path.lstat()
                if (
                    path.is_symlink()
                    or not stat.S_ISREG(path_stat.st_mode)
                    or path_stat.st_size != int(metadata["file_size"])
                ):
                    return False
                cached_array = np.load(path, mmap_mode="r", allow_pickle=False)
                try:
                    if not self._cache_metadata_matches_array(
                        metadata, cached_array
                    ):
                        return False
                    if isinstance(resident_stack, torch.Tensor):
                        resident_shape = tuple(resident_stack.shape)
                        resident_dtype = resident_stack.detach().numpy().dtype.str
                        resident_nbytes = (
                            resident_stack.numel() * resident_stack.element_size()
                        )
                    else:
                        resident_array = np.asarray(resident_stack)
                        resident_shape = tuple(resident_array.shape)
                        resident_dtype = resident_array.dtype.str
                        resident_nbytes = resident_array.nbytes
                    if (
                        resident_shape != tuple(metadata["shape"])
                        or resident_dtype != metadata["dtype"]
                        or int(resident_nbytes) != int(metadata["nbytes"])
                    ):
                        return False
                finally:
                    mapped_file = getattr(cached_array, "_mmap", None)
                    if mapped_file is not None:
                        mapped_file.close()
        except (KeyError, OSError, TypeError, ValueError):
            return False
        return True

    def _training_data_offload_enabled(self):
        return (
            bool(getattr(self, "offload_training_data_during_validation", True))
            and bool(getattr(self, "load_to_memory", False))
            and getattr(self, "epoch_end_callback", None) is not None
        )

    def _prepare_training_cache_for_run(self):
        """Cache early when this job will reload data after a validation."""
        if not self._training_data_offload_enabled():
            return False
        pending_validation = self._read_pending_validation()
        if pending_validation is not None:
            needs_reload = self.epoch < self.n_epochs
        else:
            needs_reload = self.epoch + 1 < self.n_epochs
        if not needs_reload:
            return False
        return self._ensure_training_cache(self.dataloader.dataset)

    def _resolve_training_cache_base(self):
        configured = getattr(self, "training_cache_dir", None)
        if not configured:
            configured = os.environ.get("SUPPORT_TRAINING_CACHE_DIR")
        if not configured:
            job_id = os.environ.get("SLURM_JOB_ID")
            if job_id:
                allocated_scratch = Path("/lscratch") / job_id
                if allocated_scratch.is_dir():
                    configured = allocated_scratch / "support-training"
        if not configured:
            return None

        cache_base = Path(configured).expanduser()
        cache_base.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not cache_base.is_dir():
            raise RuntimeError(
                f"Training cache path is not a directory: {cache_base}"
            )
        if not os.access(cache_base, os.W_OK | os.X_OK):
            raise PermissionError(
                f"Training cache directory is not writable: {cache_base}"
            )
        return cache_base.resolve()

    def _ensure_training_cache(self, dataset):
        """Write the normalized in-memory corpus to owned local scratch once."""
        cache_files = tuple(getattr(self, "_training_cache_files", ()))
        if cache_files:
            if self._training_cache_is_valid(cache_files, dataset.noisy_imgs):
                return True
            print(
                "WARNING: rebuilding an incomplete or invalid training scratch "
                "cache before releasing resident data.",
                file=sys.stderr,
            )
            self._remove_training_cache()

        cache_base = self._resolve_training_cache_base()
        if cache_base is None:
            if not getattr(self, "_training_cache_warning_printed", False):
                print(
                    "WARNING: no training scratch cache is configured; training "
                    "data will be reread and renormalized after each validation. "
                    "Set SUPPORT_TRAINING_CACHE_DIR to enable exact scratch reloads.",
                    file=sys.stderr,
                )
                self._training_cache_warning_printed = True
            return False

        arrays = []
        required_bytes = 0
        for stack in dataset.noisy_imgs:
            if isinstance(stack, torch.Tensor):
                if stack.device.type != "cpu":
                    raise RuntimeError(
                        "Training scratch caching requires CPU-resident tensors"
                    )
                array = stack.detach().numpy()
            else:
                array = np.asarray(stack)
            if array.dtype != np.float32 or not array.flags.c_contiguous:
                raise RuntimeError(
                    "Training scratch caching requires contiguous float32 stacks"
                )
            arrays.append(array)
            required_bytes += int(array.nbytes)
        if not arrays:
            raise RuntimeError("Cannot cache an empty training dataset")

        # NPY headers are small, but reserve 1% plus 1 MiB per stack so a cache
        # that only barely fits fails before any large write begins.
        headroom = max(
            int(required_bytes * 0.01),
            len(arrays) * 1024 * 1024,
        )
        free_bytes = shutil.disk_usage(cache_base).free
        if free_bytes < required_bytes + headroom:
            raise RuntimeError(
                "Insufficient training scratch space: need approximately "
                f"{self._format_byte_count(required_bytes + headroom)}, but "
                f"{self._format_byte_count(free_bytes)} is free in {cache_base}"
            )

        cache_root = Path(
            tempfile.mkdtemp(prefix="support-train-", dir=cache_base)
        )
        cache_root.chmod(0o700)
        metadata = []
        print(
            "Caching normalized training data for validation offload: "
            f"{len(arrays)} stack(s), {self._format_byte_count(required_bytes)} "
            f"in {cache_root}"
        )
        try:
            for index, array in enumerate(
                tqdm(arrays, desc="Caching training data to local scratch")
            ):
                # Keep large 4D corpora from placing thousands of plane caches
                # in one directory on the node-local filesystem.
                stack_directory = cache_root / f"{index // 100:04d}"
                stack_directory.mkdir(mode=0o700, exist_ok=True)
                destination = stack_directory / f"stack_{index:06d}.npy"
                file_descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=stack_directory,
                )
                temporary_path = Path(temporary_name)
                try:
                    with os.fdopen(file_descriptor, "wb") as handle:
                        file_descriptor = None
                        np.save(handle, array, allow_pickle=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                        self._drop_file_cache(handle)
                    os.replace(temporary_path, destination)
                finally:
                    if file_descriptor is not None:
                        os.close(file_descriptor)
                    if temporary_path.exists():
                        temporary_path.unlink()
                metadata.append(
                    {
                        "path": str(destination),
                        "shape": tuple(int(size) for size in array.shape),
                        "dtype": array.dtype.str,
                        "nbytes": int(array.nbytes),
                        "file_size": destination.stat().st_size,
                        "sha256": self._array_sha256(array),
                    }
                )
        except BaseException:
            shutil.rmtree(cache_root, ignore_errors=True)
            raise

        self._training_cache_root = str(cache_root)
        self._training_cache_files = tuple(metadata)
        self._training_cache_owned = True
        print("Training scratch cache is complete and will be reused each epoch.")
        return True

    @staticmethod
    def _shutdown_dataloader_workers(dataloader):
        iterator = getattr(dataloader, "_iterator", None)
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()

    def _suspend_training_data_for_validation(self, reload_after_validation):
        """Release the DataLoader and resident training tensors before validation."""
        if not self._training_data_offload_enabled():
            return False
        if getattr(self, "_suspended_training_dataset", None) is not None:
            raise RuntimeError("Training data is already suspended")

        dataloader = self.dataloader
        dataset = dataloader.dataset
        uses_cache = False
        if reload_after_validation:
            uses_cache = self._ensure_training_cache(dataset)

        self._training_dataloader_batch_count = len(dataloader)
        self._shutdown_dataloader_workers(dataloader)
        del self.dataloader
        dataset.precomputed_indices = None
        self._suspended_training_dataset = dataset
        self._suspended_patch_rng_state = dataset.patch_rng.bit_generator.state
        self._suspended_uses_training_cache = uses_cache
        resident_stack_count = len(dataset.noisy_imgs)
        dataset.noisy_imgs = []
        gc.collect()
        if not reload_after_validation:
            reload_detail = "no reload is needed after final validation"
        elif uses_cache:
            reload_detail = f"reload source: local scratch at {self._training_cache_root}"
        else:
            reload_detail = "reload source: the original TIFF files"
        print(
            f"Released {resident_stack_count} resident training stack(s) before "
            f"validation; {reload_detail}."
        )
        return True

    def _restore_training_data_after_validation(self):
        """Restore training tensors and a fresh DataLoader after validation."""
        dataset = getattr(self, "_suspended_training_dataset", None)
        if dataset is None:
            return False

        # Validation has returned, so collect its large temporary arrays before
        # allocating the resident training corpus again.
        gc.collect()
        if getattr(self, "_suspended_uses_training_cache", False):
            restored_stacks = []
            for metadata in tqdm(
                self._training_cache_files,
                desc="Reloading training data from local scratch",
            ):
                with open(metadata["path"], "rb") as handle:
                    array = np.load(handle, allow_pickle=False)
                    self._drop_file_cache(handle)
                if not self._cache_metadata_matches_array(metadata, array):
                    raise RuntimeError(
                        "Training scratch cache changed unexpectedly: "
                        f"{metadata['path']}"
                    )
                if self._array_sha256(array) != metadata["sha256"]:
                    raise RuntimeError(
                        "Training scratch cache checksum failed: "
                        f"{metadata['path']}"
                    )
                restored_stacks.append(torch.from_numpy(array))
            dataset.noisy_imgs = restored_stacks
            dataset.patch_rng.bit_generator.state = self._suspended_patch_rng_state
            self.dataloader = DataLoader(
                dataset=dataset,
                **self._dataloader_kwargs(shuffle=True),
            )
        else:
            # This path is intentionally slower but still prevents validation
            # from competing with the training corpus for RAM when local
            # scratch was not configured.
            patch_rng_state = self._suspended_patch_rng_state
            self._suspended_training_dataset = None
            self.generate_dataloader()
            self.dataloader.dataset.patch_rng.bit_generator.state = patch_rng_state

        restored_stack_count = len(self.dataloader.dataset.noisy_imgs)
        self._suspended_training_dataset = None
        self._training_dataloader_batch_count = len(self.dataloader)
        self._suspended_uses_training_cache = False
        gc.collect()
        print(
            f"Reloaded {restored_stack_count} training stack(s) after validation."
        )
        return True

    def _remove_training_cache(self):
        cache_root = getattr(self, "_training_cache_root", None)
        if cache_root and getattr(self, "_training_cache_owned", False):
            shutil.rmtree(cache_root, ignore_errors=True)
        self._training_cache_root = None
        self._training_cache_files = ()
        self._training_cache_owned = False

    def initialize_network(self):
        """
        Initialize U-Net 3D network, which is the main network architecture of DeepCAD

        Important Fields:
            self.local_model: the denoise network

        """
        model = SUPPORT(
            in_channels=self.patch_size[0],
            mid_channels=self.unet_channels,
            depth=self.depth,
            blind_conv_channels=self.bc_channels,
            one_by_one_channels=self.obo_channels,
            last_layer_channels=self.ll_channels,
            bs_size=self.bs_size,
            bp=self.bp,
        )
        self.local_model = model

    def initialize_optimizer(self):
        """
        
        """
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.lr)
        if self.cuda: self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        else: self.scaler = torch.amp.GradScaler('cpu', enabled=self.use_amp)
        self.local_model = self.local_model.to(self.device)
        random.seed(0)
        torch.manual_seed(self.rnd_seed)
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        self.rng = np.random.default_rng(self.rnd_seed)

    def _checkpoint_label(self, epoch, batch=None):
        if batch is None:
            return str(epoch)
        return f"{epoch}_batch_{batch}"

    def _checkpoint_paths(self, epoch, batch=None):
        checkpoint_dir = Path(self.pth_dir)
        label = self._checkpoint_label(epoch, batch)
        return {
            "model": checkpoint_dir / f"model_{label}.pth",
            "optimizer": checkpoint_dir / f"optimizer_{label}.pth",
            "scaler": checkpoint_dir / f"scaler_{label}.pth",
        }

    def _checkpoint_incomplete_path(self, epoch, batch=None):
        label = self._checkpoint_label(epoch, batch)
        return Path(self.pth_dir) / f".checkpoint_{label}.incomplete"

    def _required_checkpoint_paths(self, epoch, batch=None):
        paths = self._checkpoint_paths(epoch, batch)
        required = [paths["model"], paths["optimizer"]]
        if self.use_amp:
            required.append(paths["scaler"])
        return paths, required

    def _find_latest_complete_checkpoint(self):
        """Return the newest complete model/optimizer/(optional) scaler bundle."""
        checkpoint_epochs = []
        model_pattern = re.compile(r"^model_(\d+)\.pth$")
        for model_path in Path(self.pth_dir).iterdir():
            match = model_pattern.fullmatch(model_path.name)
            if match is None:
                continue
            epoch = int(match.group(1))
            if self._checkpoint_incomplete_path(epoch).is_file():
                continue
            _, required = self._required_checkpoint_paths(epoch)
            if all(path.is_file() for path in required):
                checkpoint_epochs.append(epoch)
        return max(checkpoint_epochs) if checkpoint_epochs else None

    def _find_latest_complete_generalization_checkpoint(self):
        """Return the latest complete ``(epoch, batch)`` checkpoint bundle."""
        checkpoints = []
        model_pattern = re.compile(r"^model_(\d+)_batch_(\d+)\.pth$")
        for model_path in Path(self.pth_dir).iterdir():
            match = model_pattern.fullmatch(model_path.name)
            if match is None:
                continue
            epoch, batch = (int(value) for value in match.groups())
            if self._checkpoint_incomplete_path(epoch, batch).is_file():
                continue
            _, required = self._required_checkpoint_paths(epoch, batch)
            if all(path.is_file() for path in required):
                checkpoints.append((epoch, batch))
        return max(checkpoints) if checkpoints else None

    def _find_latest_legacy_checkpoint(self):
        """Find the newest one-based, model-only checkpoint from older notebooks."""
        latest = None
        model_pattern = re.compile(r"^model(\d+)\.pth$")
        for model_path in Path(self.pth_dir).iterdir():
            match = model_pattern.fullmatch(model_path.name)
            if match is None:
                continue
            completed_epochs = int(match.group(1))
            if completed_epochs < 1:
                continue
            epoch = completed_epochs - 1
            if latest is None or epoch > latest[0]:
                latest = (epoch, model_path)
        return latest

    def _load_state_dict(self, path):
        """Load a state dict on the selected device across supported torch versions."""
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def load_checkpoint(self):
        """Restore an automatic or explicitly selected checkpoint.

        Automatic resume selects the newest complete state bundle. The explicit
        ``epoch=N`` path retains the original SUPPORT contract and loads N-1.
        """
        checkpoint_batch = None
        if self.resume and self.train_gen:
            generalization_checkpoint = (
                self._find_latest_complete_generalization_checkpoint()
            )
            epoch_checkpoint = (
                self._find_latest_complete_checkpoint()
                if self.save_every_epoch
                else None
            )
            use_generalization_checkpoint = generalization_checkpoint is not None
            if generalization_checkpoint is not None and epoch_checkpoint is not None:
                generalized_epoch, _ = generalization_checkpoint
                # A batch checkpoint in a later epoch contains newer learned
                # weights than the preceding epoch boundary. At the same epoch,
                # prefer the permanent epoch snapshot.
                use_generalization_checkpoint = generalized_epoch > epoch_checkpoint
            if use_generalization_checkpoint:
                checkpoint_epoch, checkpoint_batch = generalization_checkpoint
                total_batches = len(self.dataloader)
                if checkpoint_batch == total_batches:
                    self.epoch = checkpoint_epoch + 1
                else:
                    # A mid-epoch checkpoint repeats that epoch. This can repeat
                    # some work, but never skips the unfinished portion.
                    self.epoch = checkpoint_epoch

                paths, required = self._required_checkpoint_paths(
                    checkpoint_epoch, checkpoint_batch
                )
                incomplete_path = self._checkpoint_incomplete_path(
                    checkpoint_epoch, checkpoint_batch
                )
                if incomplete_path.is_file():
                    raise RuntimeError(
                        f"Cannot resume from epoch {checkpoint_epoch}, batch "
                        f"{checkpoint_batch}; its checkpoint bundle is marked "
                        "incomplete"
                    )
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        "Cannot resume from generalized checkpoint; missing "
                        f"checkpoint file(s): {', '.join(missing)}"
                    )
                self.local_model.load_state_dict(
                    self._load_state_dict(paths["model"])
                )
                self.optimizer.load_state_dict(
                    self._load_state_dict(paths["optimizer"])
                )
                if self.use_amp:
                    self.scaler.load_state_dict(
                        self._load_state_dict(paths["scaler"])
                    )
                self.latest_checkpoint_epoch = checkpoint_epoch
                self.latest_checkpoint_batch = checkpoint_batch
                self.model_path = str(paths["model"])
                print(
                    "Loaded generalized checkpoint from epoch "
                    f"{checkpoint_epoch}, batch {checkpoint_batch}/{total_batches}; "
                    f"training will continue at epoch {self.epoch}"
                )
                return

        if self.resume:
            checkpoint_epoch = self._find_latest_complete_checkpoint()
            legacy_checkpoint = self._find_latest_legacy_checkpoint()
            if (
                legacy_checkpoint is not None
                and (checkpoint_epoch is None or legacy_checkpoint[0] > checkpoint_epoch)
            ):
                checkpoint_epoch, legacy_model_path = legacy_checkpoint
                self.local_model.load_state_dict(
                    self._load_state_dict(legacy_model_path)
                )
                self.epoch = checkpoint_epoch + 1
                self.latest_checkpoint_epoch = checkpoint_epoch
                self.latest_checkpoint_batch = None
                self.model_path = str(legacy_model_path)
                print(
                    f"Loaded legacy model-only checkpoint {legacy_model_path.name}; "
                    f"training will continue at epoch {self.epoch} with a newly "
                    "initialized optimizer and AMP scaler"
                )
                return
            if checkpoint_epoch is None:
                self.epoch = 0
                print("No complete checkpoint found; starting training from scratch")
                return
            self.epoch = checkpoint_epoch + 1
        else:
            if self.epoch == 0:
                return
            checkpoint_epoch = self.epoch - 1

        paths, required = self._required_checkpoint_paths(checkpoint_epoch)
        incomplete_path = self._checkpoint_incomplete_path(checkpoint_epoch)
        if incomplete_path.is_file():
            raise RuntimeError(
                f"Cannot resume from epoch {checkpoint_epoch}; its checkpoint "
                "bundle is marked incomplete"
            )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cannot resume at epoch "
                f"{self.epoch}; missing checkpoint file(s): {', '.join(missing)}"
            )

        self.local_model.load_state_dict(self._load_state_dict(paths["model"]))
        self.optimizer.load_state_dict(self._load_state_dict(paths["optimizer"]))
        if self.use_amp:
            self.scaler.load_state_dict(self._load_state_dict(paths["scaler"]))
        self.latest_checkpoint_epoch = checkpoint_epoch
        self.latest_checkpoint_batch = None
        self.model_path = str(paths["model"])
        print(
            f"Loaded checkpoint from epoch {checkpoint_epoch}; "
            f"training will continue at epoch {self.epoch}"
        )

    def _atomic_torch_save(self, state, destination):
        destination = Path(destination)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(state, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _pending_validation_path(self):
        return Path(self.pth_dir) / ".pending_validation.json"

    def _write_pending_validation(self, epoch, training_metrics):
        """Atomically remember validation work paired with a saved epoch."""
        serialized_metrics = {
            "loss_total": float(training_metrics["loss_total"]),
            "loss_l1": float(training_metrics["loss_l1"]),
            "loss_l2": float(training_metrics["loss_l2"]),
        }
        for name in (
            "loss_total_nanmean",
            "loss_l1_nanmean",
            "loss_l2_nanmean",
        ):
            if name in training_metrics:
                serialized_metrics[name] = float(training_metrics[name])
        payload = {
            "version": 1,
            "epoch": int(epoch),
            "training_metrics": serialized_metrics,
        }
        destination = self._pending_validation_path()
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            file = os.fdopen(file_descriptor, "w", encoding="utf-8")
            file_descriptor = None
            with file:
                json.dump(payload, file, sort_keys=True)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, destination)
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            if temporary_path.exists():
                temporary_path.unlink()

    def _read_pending_validation(self):
        path = self._pending_validation_path()
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                raise ValueError("state must be a JSON object")
            if payload.get("version") != 1:
                raise ValueError("unsupported version")
            epoch = payload["epoch"]
            if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
                raise ValueError("invalid epoch")
            raw_metrics = payload["training_metrics"]
            if not isinstance(raw_metrics, dict):
                raise ValueError("invalid training metrics")
            training_metrics = {
                name: float(raw_metrics[name])
                for name in ("loss_total", "loss_l1", "loss_l2")
            }
            nanmean_names = (
                "loss_total_nanmean",
                "loss_l1_nanmean",
                "loss_l2_nanmean",
            )
            present_nanmeans = [
                name for name in nanmean_names if name in raw_metrics
            ]
            if present_nanmeans and len(present_nanmeans) != len(nanmean_names):
                raise ValueError("incomplete nanmean training metrics")
            training_metrics.update(
                {
                    name: float(raw_metrics[name])
                    for name in present_nanmeans
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid pending-validation state in {path}"
            ) from exc
        return {
            "epoch": int(epoch),
            "training_metrics": training_metrics,
        }

    def _clear_pending_validation(self, epoch):
        pending = self._read_pending_validation()
        if pending is None:
            return
        if pending["epoch"] != int(epoch):
            raise RuntimeError(
                "Pending-validation epoch changed while validation was running"
            )
        self._pending_validation_path().unlink()

    def _record_epoch_completion(
        self,
        epoch,
        training_metrics,
        validation_metrics,
    ):
        if hasattr(self, "log"):
            try:
                extra = (
                    {"validation": validation_metrics}
                    if validation_metrics is not None
                    else None
                )
                self.log.append_loss(
                    epoch,
                    loss_total=training_metrics["loss_total"],
                    loss_l1=training_metrics["loss_l1"],
                    loss_l2=training_metrics["loss_l2"],
                    loss_total_nanmean=training_metrics.get(
                        "loss_total_nanmean",
                        training_metrics["loss_total"],
                    ),
                    loss_l1_nanmean=training_metrics.get(
                        "loss_l1_nanmean",
                        training_metrics["loss_l1"],
                    ),
                    loss_l2_nanmean=training_metrics.get(
                        "loss_l2_nanmean",
                        training_metrics["loss_l2"],
                    ),
                    extra=extra,
                )
            except Exception:
                pass
        self._run_optional_callback(
            "epoch_complete_callback",
            self,
            epoch,
            training_metrics,
            validation_metrics,
        )

    def _promote_final_batch_checkpoint(self):
        """Keep a permanent epoch snapshot after a stop at the fourth quartile."""
        if (
            not getattr(self, "save_every_epoch", False)
            or getattr(self, "latest_checkpoint_epoch", None) is None
            or (
                getattr(self, "latest_checkpoint_batch", None)
                != len(self.dataloader)
            )
            or self.epoch != self.latest_checkpoint_epoch + 1
        ):
            return
        completed_epoch = self.latest_checkpoint_epoch
        self.save_checkpoint(completed_epoch)
        print(
            f"Promoted epoch {completed_epoch}'s final batch checkpoint to: "
            f"{self.model_path}"
        )

    def _resume_pending_validation(self):
        """Finish validation for the loaded checkpoint without retraining it."""
        pending = self._read_pending_validation()
        if pending is None:
            return
        if self.epoch_end_callback is None:
            raise RuntimeError(
                "A saved checkpoint still requires validation, but no "
                "epoch_end_callback is configured"
            )

        pending_epoch = pending["epoch"]
        loaded_epoch_boundary = (
            self.latest_checkpoint_epoch == pending_epoch
            and (
                self.latest_checkpoint_batch is None
                or self.latest_checkpoint_batch == len(self.dataloader)
            )
        )
        if not loaded_epoch_boundary and pending_epoch == self.epoch:
            print(
                f"Discarding pending validation for epoch {pending_epoch}: "
                "its complete checkpoint was not saved, so the epoch will be "
                "retrained."
            )
            self._clear_pending_validation(pending_epoch)
            return
        if not loaded_epoch_boundary:
            raise RuntimeError(
                f"Pending validation for epoch {pending_epoch} does not match "
                "the loaded checkpoint; refusing to skip required validation"
            )

        next_epoch = self.epoch
        self.epoch = pending_epoch
        training_metrics = pending["training_metrics"]
        print(
            f"Resuming validation for saved epoch {pending_epoch} before "
            f"training epoch {next_epoch}."
        )
        self._run_optional_callback(
            "epoch_start_callback",
            self,
            pending_epoch,
        )
        reload_after_validation = next_epoch < self.n_epochs
        self._suspend_training_data_for_validation(reload_after_validation)
        validation_metrics = self.epoch_end_callback(
            self,
            pending_epoch,
            training_metrics,
        )
        self.last_validation_metrics = validation_metrics
        # The validation callback records its durable CSV state before it
        # returns. Once it succeeds, a retry is no longer required.
        self._clear_pending_validation(pending_epoch)
        self._record_epoch_completion(
            pending_epoch,
            training_metrics,
            validation_metrics,
        )
        self.epoch = next_epoch
        if reload_after_validation:
            self._restore_training_data_after_validation()
        else:
            self._remove_training_cache()

    def _remove_intermediary_checkpoints(self, keep_epoch, keep_batch=None):
        checkpoint_pattern = re.compile(r"^(model|optimizer|scaler)_(\d+)\.pth$")
        generalization_pattern = re.compile(
            r"^(model|optimizer|scaler)_(\d+)_batch_(\d+)\.pth$"
        )
        legacy_model_pattern = re.compile(r"^model\d+\.pth$")
        incomplete_pattern = re.compile(r"^\.checkpoint_(\d+)\.incomplete$")
        generalization_incomplete_pattern = re.compile(
            r"^\.checkpoint_(\d+)_batch_(\d+)\.incomplete$"
        )
        for path in Path(self.pth_dir).iterdir():
            match = checkpoint_pattern.fullmatch(path.name)
            remove = match is not None and (
                keep_batch is not None or int(match.group(2)) != keep_epoch
            )
            generalization_match = generalization_pattern.fullmatch(path.name)
            remove = remove or (
                generalization_match is not None
                and (
                    int(generalization_match.group(2)),
                    int(generalization_match.group(3)),
                )
                != (keep_epoch, keep_batch)
            )
            remove = remove or legacy_model_pattern.fullmatch(path.name) is not None
            incomplete_match = incomplete_pattern.fullmatch(path.name)
            remove = remove or (
                incomplete_match is not None
                and (
                    keep_batch is not None
                    or int(incomplete_match.group(1)) != keep_epoch
                )
            )
            generalization_incomplete_match = (
                generalization_incomplete_pattern.fullmatch(path.name)
            )
            remove = remove or (
                generalization_incomplete_match is not None
                and (
                    int(generalization_incomplete_match.group(1)),
                    int(generalization_incomplete_match.group(2)),
                )
                != (keep_epoch, keep_batch)
            )
            if remove:
                path.unlink()

    def _remove_batch_checkpoints(self, keep_epoch=None, keep_batch=None):
        """Remove rolling batch bundles without touching epoch snapshots."""
        checkpoint_pattern = re.compile(
            r"^(model|optimizer|scaler)_(\d+)_batch_(\d+)\.pth$"
        )
        incomplete_pattern = re.compile(
            r"^\.checkpoint_(\d+)_batch_(\d+)\.incomplete$"
        )
        keep = (
            (int(keep_epoch), int(keep_batch))
            if keep_epoch is not None and keep_batch is not None
            else None
        )
        for path in Path(self.pth_dir).iterdir():
            checkpoint_match = checkpoint_pattern.fullmatch(path.name)
            incomplete_match = incomplete_pattern.fullmatch(path.name)
            if checkpoint_match is not None:
                checkpoint = (
                    int(checkpoint_match.group(2)),
                    int(checkpoint_match.group(3)),
                )
            elif incomplete_match is not None:
                checkpoint = (
                    int(incomplete_match.group(1)),
                    int(incomplete_match.group(2)),
                )
            else:
                continue
            if checkpoint != keep:
                path.unlink()

    def save_checkpoint(self, epoch, batch=None, remove_previous=False):
        """Save a complete epoch-level or generalized batch-level state bundle."""
        paths = self._checkpoint_paths(epoch, batch)
        incomplete_path = self._checkpoint_incomplete_path(epoch, batch)
        incomplete_path.touch(exist_ok=True)
        self._atomic_torch_save(self.local_model.state_dict(), paths["model"])
        self._atomic_torch_save(self.optimizer.state_dict(), paths["optimizer"])
        if self.use_amp:
            self._atomic_torch_save(self.scaler.state_dict(), paths["scaler"])
        elif paths["scaler"].is_file():
            paths["scaler"].unlink()
        incomplete_path.unlink()

        self.latest_checkpoint_epoch = epoch
        self.latest_checkpoint_batch = batch
        self.model_path = str(paths["model"])
        if batch is not None:
            if remove_previous or self.rm_intermediaries:
                # Batch checkpoints are one rolling recovery bundle. Keep
                # permanent epoch snapshots until a newer epoch commits.
                self._remove_batch_checkpoints(
                    keep_epoch=epoch,
                    keep_batch=batch,
                )
        else:
            # The permanent epoch bundle supersedes the rolling batch bundle.
            self._remove_batch_checkpoints()
        if batch is None and (remove_previous or self.rm_intermediaries):
            self._remove_intermediary_checkpoints(
                keep_epoch=epoch, keep_batch=batch
            )

    @staticmethod
    def _quarter_checkpoint_batches(total_batches):
        """Return distinct one-based batch numbers at the four quartile ends."""
        if total_batches < 1:
            return ()
        return tuple(sorted({
            (total_batches * quarter + 3) // 4
            for quarter in range(1, 5)
        }))

    def _scheduled_batch_checkpoint_batches(self, total_batches):
        if not self.train_gen or not self.checkpoints:
            return set()
        return set(self._quarter_checkpoint_batches(total_batches))
        

    def train(self):
        """
        Train a model for a single epoch

        Arguments:
            train_dataloader: (Pytorch DataLoader)
            model: (Pytorch nn.Module)
            optimizer: (Pytorch optimzer)
            rng: numpy random number generator
            writer: (Tensorboard writer)
            epoch: epoch of training (int)
            opt: argparse dictionary

        Returns:
            loss_list: list of total loss of each batch ([float])
            loss_list_l1: list of L1 loss of each batch ([float])
            loss_list_l2: list of L2 loss of each batch ([float])
            corr_list: list of correlation of each batch ([float])
        """
        self.rotate = True if self.bs_size[0] == self.bs_size[1] else False
        self.local_model.train()
        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()
        lh_L = []
        lh_L1 = []
        lh_L2 = []
        total_batches = len(self.dataloader)
        checkpoint_batches = self._scheduled_batch_checkpoint_batches(
            total_batches
        )
        if (
            self.latest_checkpoint_epoch == self.epoch
            and self.latest_checkpoint_batch is not None
        ):
            # A resumed incomplete epoch is retrained from its beginning, but an
            # earlier quartile must never replace the newer bundle we loaded.
            checkpoint_batches = {
                batch
                for batch in checkpoint_batches
                if batch > self.latest_checkpoint_batch
            }

        with tqdm(self.dataloader, unit='batches', desc=f'Epoch {self.epoch}') as pbar:
            for i, data in enumerate(pbar):
                (noisy_img, _, ds_idx) = data
                B, T, X, Y = noisy_img.shape
                noisy_img = noisy_img.to(self.device)
                noisy_img, _ = random_transform(
                    input=noisy_img, 
                    target=None, 
                    rng=self.rng, 
                    is_rotate=self.rotate
                )
                noisy_img_target = torch.unsqueeze(noisy_img[:,int(T/2),:,:], dim=1)
                self.optimizer.zero_grad()
                
                # Forward pass wrapped in autocast for AMP
                with torch.amp.autocast(self.device, enabled=self.use_amp):
                    denoised_img = self.local_model(noisy_img)
                    loss_L1_pixelwise = L1_pixelwise(denoised_img, noisy_img_target)
                    loss_L2_pixelwise = L2_pixelwise(denoised_img, noisy_img_target)
                    loss_sum = self.loss_coef[0] * loss_L1_pixelwise + self.loss_coef[1] * loss_L2_pixelwise
                    
                # Backward pass with GradScaler if AMP is enabled
                self.scaler.scale(loss_sum).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                lh_L.append(loss_sum.item())
                lh_L1.append(loss_L1_pixelwise.item())
                lh_L2.append(loss_L2_pixelwise.item())
                L_mean, L_nanmean = _mean_and_nanmean(lh_L)
                L1_mean, L1_nanmean = _mean_and_nanmean(lh_L1)
                L2_mean, L2_nanmean = _mean_and_nanmean(lh_L2)
                pbar.set_postfix(
                    {
                        'L': L_nanmean,
                        'L1': L1_nanmean,
                        'L2': L2_nanmean,
                    }
                )
                batch = i + 1
                if self.train_gen:
                    if not hasattr(self, "log"):
                        raise RuntimeError(
                            "Generalized training requires an initialized log"
                        )
                    self.log.append_batch_loss(
                        self.epoch,
                        batch,
                        total_batches,
                        L1_mean=L1_mean,
                        L2_mean=L2_mean,
                        L_mean=L_mean,
                        L1_nanmean=L1_nanmean,
                        L2_nanmean=L2_nanmean,
                        L_nanmean=L_nanmean,
                    )
                if batch in checkpoint_batches:
                    # When an interrupted epoch restarts exactly on its latest
                    # saved quartile, retain that complete bundle until the next
                    # distinct checkpoint has been written.
                    is_current_checkpoint = (
                        self.latest_checkpoint_epoch == self.epoch
                        and self.latest_checkpoint_batch == batch
                        and self.model_path is not None
                        and Path(self.model_path).is_file()
                    )
                    if not is_current_checkpoint:
                        if (
                            batch == total_batches
                            and self.epoch_end_callback is not None
                        ):
                            self._write_pending_validation(
                                self.epoch,
                                {
                                    "loss_total": L_mean,
                                    "loss_l1": L1_mean,
                                    "loss_l2": L2_mean,
                                    "loss_total_nanmean": L_nanmean,
                                    "loss_l1_nanmean": L1_nanmean,
                                    "loss_l2_nanmean": L2_nanmean,
                                },
                            )
                        self.save_checkpoint(
                            self.epoch,
                            batch=batch,
                            remove_previous=True,
                        )
                        tqdm.write(
                            "Saved rolling batch checkpoint at epoch "
                            f"{self.epoch}, batch {batch}/{total_batches}: "
                            f"{self.model_path}"
                        )
        self.lh_L = lh_L
        self.lh_L1 = lh_L1
        self.lh_L2 = lh_L2
    
    def test(self):
        """
        
        """
        self._promote_final_batch_checkpoint()
        self._resume_pending_validation()
        t_init = time.mktime(time.localtime())
        lh = []
        clear_epochs_interval = 10
        clear_epochs = np.arange(clear_epochs_interval, 100, clear_epochs_interval)
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.n_epochs):
            self.epoch = epoch
            self._run_optional_callback(
                "epoch_start_callback",
                self,
                epoch,
            )
            self.dataloader.dataset.precompute_indices()
            self.train()
            t_inter = int(time.mktime(time.localtime()) - t_init)
            L_mean, L_nanmean = _mean_and_nanmean(self.lh_L)
            L1_mean, L1_nanmean = _mean_and_nanmean(self.lh_L1)
            L2_mean, L2_nanmean = _mean_and_nanmean(self.lh_L2)
            lh.append(L_mean)
            training_metrics = {
                "loss_total": float(L_mean),
                "loss_l1": float(L1_mean),
                "loss_l2": float(L2_mean),
                "loss_total_nanmean": float(L_nanmean),
                "loss_l1_nanmean": float(L1_nanmean),
                "loss_l2_nanmean": float(L2_nanmean),
            }
            saves_epoch_checkpoint = (
                self.checkpoints
                and (not self.train_gen or self.save_every_epoch)
            ) or (
                not self.checkpoints
                and self._checkpoint_controls_configured
                and epoch == self.n_epochs - 1
            )
            if (
                self.epoch_end_callback is not None
                and saves_epoch_checkpoint
            ):
                # Persist this first so any complete checkpoint is known to
                # need validation if the process stops before validation ends.
                self._write_pending_validation(epoch, training_metrics)
            if self.checkpoints:
                if not self.train_gen or self.save_every_epoch:
                    self.save_checkpoint(epoch)
            elif self._checkpoint_controls_configured and epoch == self.n_epochs - 1:
                # New checkpoint controls save a complete final bundle for inference
                # and for later continuation, even when per-epoch saves are disabled.
                self.save_checkpoint(epoch)
            # Preserve the final-model convention used by legacy callers.
            elif epoch == self.n_epochs - 1:
                legacy_model_path = Path(self.pth_dir) / f'model{epoch+1:03d}.pth'
                self._atomic_torch_save(
                    self.local_model.state_dict(),
                    legacy_model_path,
                )
                self.model_path = str(legacy_model_path)
            has_resumable_epoch_checkpoint = saves_epoch_checkpoint or (
                self.checkpoints
                and self.train_gen
                and not self.save_every_epoch
                and self.latest_checkpoint_epoch == epoch
                and self.latest_checkpoint_batch == len(self.dataloader)
            )
            if has_resumable_epoch_checkpoint:
                timing = (
                    " before validation"
                    if self.epoch_end_callback is not None
                    else ""
                )
                print(
                    f"Saved epoch {epoch} checkpoint{timing}: "
                    f"{getattr(self, 'model_path', None)}"
                )
            plotASCII(lh)
            validation_metrics = None
            if self.epoch_end_callback is not None:
                reload_after_validation = epoch + 1 < self.n_epochs
                self._suspend_training_data_for_validation(
                    reload_after_validation
                )
                validation_metrics = self.epoch_end_callback(
                    self,
                    epoch,
                    training_metrics,
                )
                self.last_validation_metrics = validation_metrics
                if has_resumable_epoch_checkpoint:
                    self._clear_pending_validation(epoch)
            self._record_epoch_completion(
                epoch,
                training_metrics,
                validation_metrics,
            )
            if (
                self.epoch_end_callback is not None
                and epoch + 1 < self.n_epochs
            ):
                self._restore_training_data_after_validation()
            elif self.epoch_end_callback is not None:
                self._remove_training_cache()
            if epoch in clear_epochs:
                clear_output(wait=True)
