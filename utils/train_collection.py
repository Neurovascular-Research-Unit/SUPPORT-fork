import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import re
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
        # Backward-compatible alias for callers that used the earlier API.
        self.save_every_epoch = False
        self.rnd_seed = 0
        self.cuda_device = [0]
        
        # DATASET
        self.patch_size = [61, 128, 128]
        self.patch_interval = [1, 64, 64]
        self.batch_size = 16
        self.load_to_memory = True
        
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
        self._checkpoint_controls_configured = any(
            key in params_dict
            for key in ("checkpoints", "resume", "rm_intermediaries", "save_every_epoch")
        )
        checkpoints_was_set = "checkpoints" in params_dict
        self.set_params(params_dict)
        if not checkpoints_was_set and self.save_every_epoch:
            self.checkpoints = True
        for option in ("checkpoints", "resume", "rm_intermediaries"):
            if type(getattr(self, option)) is not bool:
                raise TypeError(f"{option} must be True or False")
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
            "pin_memory": self.cuda,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs
        
    def vars_dict(self): self.prepare_file()

    def release_resources(self):
        if hasattr(self, "local_model"):
            self.local_model.to("cpu")
            del self.local_model
        if hasattr(self, "optimizer"):
            self.optimizer.state.clear()
            del self.optimizer
        if hasattr(self, "scaler"):
            del self.scaler
        if hasattr(self, "dataloader"):
            del self.dataloader
        if hasattr(self, "Tensor"):
            del self.Tensor
        if hasattr(self, "rng"):
            del self.rng
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
        self.test()

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
            pass
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
        print("\033[1;31mLoaded Files -----> \033[0m")
        if self.is_dir:
            noisy_paths = [
                path for path in Path(self.in_path).iterdir()
                if path.is_file() and path.suffix.lower() in ('.tif', '.tiff')
            ]
        else:
            noisy_paths = [Path(self.in_path)]
        noisy_paths = sorted(noisy_paths)
        if len(noisy_paths) == 0:
            raise FileNotFoundError(f'No TIFF files were found in {self.in_path}.')
        noisy_stems = [str(path.stem) for path in noisy_paths]
        slen = max((len(s) for s in noisy_stems), default=0)

        for path, noisy_stem in zip(noisy_paths, noisy_stems):
            noisy_img = self._read_stack(path)
            print(f"\033[1m{noisy_stem:<{slen}}\033[0m  {self._describe_stack(noisy_img.shape)}")
            noisy_imgs.extend(self._prepare_training_stacks(noisy_img))

        dataset = DatasetSUPPORT(
            noisy_imgs=noisy_imgs,
            patch_size=self.patch_size,
            patch_interval=self.patch_interval,
            transform=None,
            random_patch=True,
            load_to_memory=self.load_to_memory
        )
        
        self.dataloader = DataLoader(
            dataset=dataset,
            **self._dataloader_kwargs(shuffle=True),
        )

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

    def _checkpoint_paths(self, epoch):
        checkpoint_dir = Path(self.pth_dir)
        return {
            "model": checkpoint_dir / f"model_{epoch}.pth",
            "optimizer": checkpoint_dir / f"optimizer_{epoch}.pth",
            "scaler": checkpoint_dir / f"scaler_{epoch}.pth",
        }

    def _checkpoint_incomplete_path(self, epoch):
        return Path(self.pth_dir) / f".checkpoint_{epoch}.incomplete"

    def _required_checkpoint_paths(self, epoch):
        paths = self._checkpoint_paths(epoch)
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

    def _remove_intermediary_checkpoints(self, keep_epoch):
        checkpoint_pattern = re.compile(r"^(model|optimizer|scaler)_(\d+)\.pth$")
        legacy_model_pattern = re.compile(r"^model\d+\.pth$")
        incomplete_pattern = re.compile(r"^\.checkpoint_(\d+)\.incomplete$")
        for path in Path(self.pth_dir).iterdir():
            match = checkpoint_pattern.fullmatch(path.name)
            remove = match is not None and int(match.group(2)) != keep_epoch
            remove = remove or legacy_model_pattern.fullmatch(path.name) is not None
            incomplete_match = incomplete_pattern.fullmatch(path.name)
            remove = remove or (
                incomplete_match is not None
                and int(incomplete_match.group(1)) != keep_epoch
            )
            if remove:
                path.unlink()

    def save_checkpoint(self, epoch):
        """Save model, optimizer, and optional AMP state for a completed epoch."""
        paths = self._checkpoint_paths(epoch)
        incomplete_path = self._checkpoint_incomplete_path(epoch)
        incomplete_path.touch(exist_ok=True)
        self._atomic_torch_save(self.local_model.state_dict(), paths["model"])
        self._atomic_torch_save(self.optimizer.state_dict(), paths["optimizer"])
        if self.use_amp:
            self._atomic_torch_save(self.scaler.state_dict(), paths["scaler"])
        elif paths["scaler"].is_file():
            paths["scaler"].unlink()
        incomplete_path.unlink()

        self.latest_checkpoint_epoch = epoch
        self.model_path = str(paths["model"])
        if self.rm_intermediaries:
            self._remove_intermediary_checkpoints(keep_epoch=epoch)
        

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
                pbar.set_postfix(
                    {
                        'L':np.mean(np.array(lh_L)), 
                        'L1': np.mean(np.array(lh_L1)), 
                        'L2': np.mean(np.array(lh_L2))
                    }
                )
        self.lh_L = lh_L
        self.lh_L1 = lh_L1
        self.lh_L2 = lh_L2
    
    def test(self):
        """
        
        """
        t_init = time.mktime(time.localtime())
        lh = []
        clear_epochs_interval = 10
        clear_epochs = np.arange(clear_epochs_interval, 100, clear_epochs_interval)
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.n_epochs):
            self.epoch = epoch
            self.dataloader.dataset.precompute_indices()
            self.train()
            t_inter = int(time.mktime(time.localtime()) - t_init)
            L_mean = np.mean(np.array(self.lh_L))
            L1_mean = np.mean(np.array(self.lh_L1))
            L2_mean = np.mean(np.array(self.lh_L2))
            lh.append(L_mean)
            plotASCII(lh)
            if hasattr(self, "log"):
                try:
                    self.log.append_loss(epoch, loss_total=L_mean, loss_l1=L1_mean, loss_l2=L2_mean)
                except Exception:
                    pass
            if self.checkpoints:
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
            if epoch in clear_epochs:
                clear_output(wait=True)
