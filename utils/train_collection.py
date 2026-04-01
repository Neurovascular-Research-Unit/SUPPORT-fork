import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        self.rnd_seed = 0
        self.cuda_device = [0]
        
        # DATASET
        self.patch_size = [61, 128, 128]
        self.patch_interval = [1, 64, 64]
        self.batch_size = 16
        
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
        self.cuda = torch.cuda.is_available() and not self.use_cpu
        self.is_dir = False 
        self.set_params(params_dict)
        
    def vars_dict(self): self.prepare_file()
        
    def run(self):
        """
        General function for training SUPPORT network.
        """
        self.prepare_file()
        self.generate_dataloader()
        self.initialize_network()
        self.initialize_optimizer()
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
        if not Path(self.out_dir).exists(): Path(self.out_dir).mkdir()
        if not Path(self.pth_dir).exists(): Path(self.pth_dir).mkdir()
        if not Path(self.onnx_dir).exists(): Path(self.onnx_dir).mkdir()
        try:
            self.log = LogManager(self, mode="train", out_dir=self.out_dir, filename_prefix=self.exp_name)
            self.log.log_params()
        except Exception:
            pass

    def _read_stack(self, path):
        noisy_img = tifffile.TiffFile(path).asarray()
        noisy_img = torch.from_numpy(noisy_img).to("cuda").to(torch.float32).cpu().numpy()
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
                    torch.from_numpy(np.ascontiguousarray(noisy_img[:, z_idx, :, :])).type(torch.FloatTensor)
                    for z_idx in range(noisy_img.shape[1])
                ]
            if noisy_img.ndim == 3:
                return [torch.from_numpy(noisy_img).type(torch.FloatTensor)]
            raise ValueError(f"Expected a 3D or 4D TIFF for 3+1D input, received shape {noisy_img.shape}.")

        if self.dimensionality in ("2+1D", "3D"):
            if noisy_img.ndim != 3:
                raise ValueError(
                    f"Expected a 3D TIFF for {self.dimensionality} input, received shape {noisy_img.shape}."
                )
            return [torch.from_numpy(noisy_img).type(torch.FloatTensor)]

        if noisy_img.ndim != 3:
            raise ValueError(
                f"Expected a 3D TIFF for {self.dimensionality} input, received shape {noisy_img.shape}."
            )
        return [torch.from_numpy(noisy_img).type(torch.FloatTensor)]

    def generate_dataloader(self):
        """
        
        """
        noisy_imgs = []
        print("\033[1;31mLoaded Files -----> \033[0m")
        noisy_paths = list(Path(self.in_path).glob('*.tif')) if self.is_dir else [Path(self.in_path)]
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
            load_to_memory=False
        )
        
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor
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
        if self.cuda: self.local_model = self.local_model.cuda()
        random.seed(0)
        torch.manual_seed(self.rnd_seed)
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        self.rng = np.random.default_rng(self.rnd_seed)
        

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
                noisy_img = noisy_img.cuda()
                noisy_img, _ = random_transform(
                    input=noisy_img, 
                    target=None, 
                    rng=self.rng, 
                    is_rotate=self.rotate
                )
                noisy_img_target = torch.unsqueeze(noisy_img[:,int(T/2),:,:], dim=1)
                self.optimizer.zero_grad()
                
                # Forward pass wrapped in autocast for AMP
                with torch.amp.autocast('cuda', enabled=self.use_amp):
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
        for epoch in range(self.n_epochs):
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
            if epoch == self.n_epochs or epoch == self.n_epochs-1: torch.save(self.local_model.state_dict(), self.pth_dir + f'/model{epoch+1:03d}.pth')
            if epoch in clear_epochs:
                clear_output(wait=True)
