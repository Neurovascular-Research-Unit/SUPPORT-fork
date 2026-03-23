import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path('./').resolve().parent.parent))
import torch
import numpy as np
from tqdm import tqdm
import tifffile
from utils.log_methods import LogManager
from utils.dataset import DatasetSUPPORT_test_stitch
from torch.utils.data import DataLoader
from support.SUPPORT import SUPPORT



class TestingClass():
    """
    
    """
    def __init__(self, params_dict):
        """
        Constructor class for testing process

        Args:
           params_dict: dict
               The collection of testing params set by users
        Returns:
           self
        """
        # EXPERIMENT
        self.in_path = ""
        self.model_path = ""
        self.out_dir = None
        self.exp_name = None
        self.dimensionality = None
        self.rm_dir = True
        self.cuda_device = [0]
        self.pad = True
        
        # DATASET
        self.patch_size = [61, 128, 128]
        self.patch_interval = [1, 11, 11]
        self.batch_size = 16
        
        # MODEL
        self.depth = 5
        self.bc_channels = 64
        self.obo_channels = [32, 16]
        self.ll_channels = [64, 32, 16]
        self.bs_size = [3, 3]
        self.bp = False
        self.unet_channels = [64, 128, 256, 512, 1024]
        
        # TESTING
        self.use_cpu = False
        self.n_cpu = 8
        self.prefetch_factor = 2
        
        # AUTOMATED
        self.cuda = torch.cuda.is_available() and not self.use_cpu
        self.is_dir = False
        self.set_params(params_dict)
        
    def run(self):
        """
        General function for training SUPPORT network.
        """
        self.initialize_network()
        self.run_test()
        
    def set_params(self, params_dict):
        # This is where I would implement overlap factor
        print('\033[1;31mTesting parameters -----> \033[0m')
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
        try:
            self.log = LogManager(self, mode="test", out_dir=self.out_dir, filename_prefix=self.exp_name)
            self.log.log_params()
        except Exception:
            pass
    
    def initialize_network(self):
        """
        Initialize U-Net 3D network, which is the main network architecture of SUPPORT

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
            bp=self.bp
        )
        self.local_model = model
        checkpoint = torch.load(self.model_path, weights_only=True)
        self.local_model.load_state_dict(checkpoint)
        self.local_model.to('cuda')
        
    def generate_dataloader(self, path):
        """"""
        noisy_img = tifffile.TiffFile(path).asarray()
        noisy_img = noisy_img.astype(np.float32, copy=False)
        if self.pad: noisy_img = np.pad(noisy_img, (((self.local_model.in_channels)//2, (self.local_model.in_channels)//2), (0,0), (0,0)), 'reflect')
        noisy_img = torch.from_numpy(noisy_img).type(torch.FloatTensor)
        test_set = DatasetSUPPORT_test_stitch(
            noisy_img=noisy_img, 
            patch_size=self.patch_size, 
            patch_interval=self.patch_interval,
            transform=None,
            random_patch=False,
            load_to_memory=True,
        )
        self.dataloader = DataLoader(
            dataset=test_set, 
            batch_size=self.batch_size, 
            num_workers=self.n_cpu,
        )
            
    def save_inference(self, out_path):
        if self.pad:
            crop_margin = (self.local_model.in_channels) // 2
            tifffile.imwrite(out_path, self.denoised_stack[crop_margin:-crop_margin, :, :], imagej=True, metadata={'axes': 'TYX'})
        else: tifffile.imwrite(out_path, self.denoised_stack, imagej=True, metadata={'axes': 'TYX'})
        
            
    def test(self):
        """
        Validate a model with a test data
        
        Arguments:
            test_dataloader: (Pytorch DataLoader)
                Should be DatasetFRECTAL_test_stitch!
            model: (Pytorch nn.Module)

        Returns:
            denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
        """
        with torch.no_grad():
            self.local_model.eval()
            denoised_stack = np.zeros(self.dataloader.dataset.noisy_img.shape, dtype=np.float32)
            for _, (noisy_img, _, single_coordinate) in enumerate(tqdm(self.dataloader, desc="validate")):
                noisy_img = noisy_img.cuda() #[b, z, y, x]
                noisy_img_denoised = self.local_model(noisy_img)
                T = noisy_img.size(1)
                for bi in range(noisy_img.size(0)): 
                    stack_start_w = int(single_coordinate['stack_start_w'][bi])
                    stack_end_w = int(single_coordinate['stack_end_w'][bi])
                    patch_start_w = int(single_coordinate['patch_start_w'][bi])
                    patch_end_w = int(single_coordinate['patch_end_w'][bi])

                    stack_start_h = int(single_coordinate['stack_start_h'][bi])
                    stack_end_h = int(single_coordinate['stack_end_h'][bi])
                    patch_start_h = int(single_coordinate['patch_start_h'][bi])
                    patch_end_h = int(single_coordinate['patch_end_h'][bi])

                    stack_start_s = int(single_coordinate['init_s'][bi])
                    
                    denoised_stack[stack_start_s+(T//2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = noisy_img_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

            # change nan values to 0 and denormalize
            self.denoised_stack = denoised_stack * self.dataloader.dataset.std_img.numpy() + self.dataloader.dataset.mean_img.numpy()
        
    def compile(self):
        """
        """
        denoised_paths = list(Path(self.out_dir).glob(f'{self.exp_name}*.tif'))
        denoised_paths = [str(path) for path in denoised_paths]
        denoised_paths.sort()
        denoised_img = []
        for path in denoised_paths:
            denoised_img.append(tifffile.TiffFile(path).asarray())
        if self.dimensionality == '3+1D':
            denoised_img = np.stack(denoised_img, axis=1).astype(np.float32)
        else:
            denoised_img = np.array(denoised_img)
        # try:
        #     for path in denoised_paths: Path(path).unlink()
        # except: pass
        out_path = self.out_dir + f'/{self.exp_name}_SPRT.tif'
        print(out_path)
        if self.dimensionality == '3+1D': 
            tifffile.imwrite(out_path, denoised_img.astype(np.float32), imagej=True, metadata={'axes': 'TZYX'})
        elif self.dimensionality == '2+1D':
            tifffile.imwrite(out_path, denoised_img.astype(np.float32), imagej=True, metadata={'axes': 'TYX'})
        
        
        
    def run_test(self):
        """
        """
        if self.is_dir:
            out_path = self.out_dir + f'/{self.exp_name}_SPRT.tif'
            print(out_path)
            noisy_paths = list(Path(self.in_path).glob('*.tif'))
            noisy_paths = [str(path) for path in noisy_paths]
            noisy_paths.sort()
            noisy_stems = [str(Path(path).stem) for path in noisy_paths]
            for i, path in enumerate(noisy_paths):
                out_path = self.out_dir+f'/{self.exp_name}_{i:03d}.tif'
                print("\033[1;31mRunning inference on\033[0m")
                print(f'\033[1m{noisy_stems[i]}\033[0m')
                self.generate_dataloader(path)
                self.test()
                print(f'Saving denoised \033[1m{noisy_stems[i]}\033[0m to {out_path}')
                self.save_inference(out_path)
            if self.dimensionality in ("3+1D", "2+1D"): self.compile()
        else:
            noisy_stem = str(Path(self.in_path).stem)
            out_path = self.out_dir+f'/{self.exp_name}.tif'
            print(f'Running inference on \033[1m{noisy_stem}\033[0m')
            self.generate_dataloader(self.in_path)
            self.test()
            print(f'Saving denoised \033[1m{noisy_stem}\033[0m to {out_path}')
            self.save_inference(out_path)
            