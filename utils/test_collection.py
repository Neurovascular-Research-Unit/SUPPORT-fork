import os
import shutil
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import numpy as np
from tqdm import tqdm
import tifffile
from utils.log_methods import LogManager
from utils.dataset import DatasetSUPPORT_test_stitch, dataset_stats
from torch.utils.data import DataLoader
from support.SUPPORT import SUPPORT


def _copy_axis0_chunk(source, destination, start, stop):
    """Copy one non-overlapping axis-0 chunk while converting to float32."""
    np.copyto(destination[start:stop], source[start:stop], casting="unsafe")


def _as_float32_parallel(array, max_workers):
    """Return an array as float32, parallelizing full-stack dtype conversion."""
    source = np.asarray(array)
    if source.dtype == np.float32 and source.flags.writeable:
        return source

    destination = np.empty(source.shape, dtype=np.float32)
    if source.size == 0:
        return destination
    if source.ndim == 0:
        np.copyto(destination, source, casting="unsafe")
        return destination

    try:
        requested_workers = int(max_workers)
    except (TypeError, ValueError):
        requested_workers = 1
    worker_count = min(max(1, requested_workers), source.shape[0])
    boundaries = np.linspace(0, source.shape[0], worker_count + 1, dtype=int)
    chunks = list(zip(boundaries[:-1], boundaries[1:]))

    if worker_count == 1:
        _copy_axis0_chunk(source, destination, *chunks[0])
        return destination

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _copy_axis0_chunk,
                source,
                destination,
                int(start),
                int(stop),
            )
            for start, stop in chunks
        ]
        for future in futures:
            future.result()
    return destination


def _denormalize_in_place(array, std, mean):
    """Undo normalization without allocating full-stack intermediates."""
    array *= std
    array += mean
    return array



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
        self.output_path = None
        self.exp_name = None
        self.dimensionality = None
        self.rm_dir = True
        self.cuda_device = [0]
        self.pad = True
        self.track_loss = False
        
        # DATASET
        self.patch_size = [61, 128, 128]
        self.patch_interval = [1, 11, 11]
        self.batch_size = 16
        self.pin_memory = True
        self.normalization_mean = None
        self.normalization_std = None
        
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
        self.loss_coef = [0.5, 0.5]
        
        # AUTOMATED
        self.cuda = False
        self.device = "cpu"
        self.is_dir = False
        self._owns_model = False
        self.set_params(params_dict)
        for option in ("pin_memory", "track_loss"):
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

    def _dataloader_kwargs(self):
        # Notebook kernels are more prone to worker crashes; command-line jobs use
        # the configured worker count supplied by n_cpu.
        num_workers = 0 if self._in_ipykernel() else self.n_cpu
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "pin_memory": self.cuda and self.pin_memory,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs
        
    def run(self, model=None):
        """
        General function for training SUPPORT network.
        """
        shared_model = model is not None
        previous_training_mode = model.training if shared_model else None
        if shared_model:
            self.local_model = model
            self._owns_model = False
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.initialize_network()
            self._owns_model = True

        self._reset_loss_tracking()
        try:
            self.run_test()
            return self._loss_metrics() if self.track_loss else None
        finally:
            if shared_model:
                self.local_model.train(previous_training_mode)

    def release_resources(self):
        if hasattr(self, "local_model"):
            if self._owns_model:
                self.local_model.to("cpu")
            del self.local_model
        if hasattr(self, "dataloader"):
            del self.dataloader
        if hasattr(self, "denoised_stack"):
            del self.denoised_stack
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
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
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        self.local_model.load_state_dict(checkpoint)
        self.local_model.to(self.device)
        
    def _read_stack(self, path):
        path_name = Path(path).name
        mapped_source = False
        try:
            # Copy-on-write keeps native float32 TIFFs lazy while allowing the
            # dataset to normalize them in place without modifying the file.
            noisy_img = tifffile.memmap(path, mode="c", series=0)
            mapped_source = True
            read_method = "memory-mapped"
        except (OSError, ValueError):
            # Compressed and non-contiguous TIFF data cannot be memory-mapped.
            with tifffile.TiffFile(path) as tif:
                noisy_img = tif.asarray()
            read_method = "loaded"

        if noisy_img.dtype == np.float32 and noisy_img.flags.writeable:
            print(
                f"Read {path_name} as {read_method} float32 data "
                f"with shape {noisy_img.shape}.",
                flush=True,
            )
            return np.asarray(noisy_img)

        try:
            requested_workers = int(self.n_cpu)
        except (TypeError, ValueError):
            requested_workers = 1
        slice_count = noisy_img.shape[0] if noisy_img.ndim else 1
        worker_count = min(max(1, requested_workers), max(1, slice_count))
        if noisy_img.dtype == np.float32:
            action = f"Copying {read_method} {path_name} into writable float32 memory"
        else:
            action = (
                f"Converting {read_method} {path_name} from {noisy_img.dtype} "
                "to float32"
            )
        print(f"{action} with {worker_count} worker(s)...", flush=True)
        try:
            converted_img = _as_float32_parallel(noisy_img, worker_count)
        finally:
            if mapped_source:
                mmap_handle = getattr(noisy_img, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
        print(f"Finished preparing {path_name} as float32.", flush=True)
        return converted_img

    def _describe_stack(self, noisy_shape):
        if self.dimensionality == "3+1D":
            if len(noisy_shape) == 4:
                T, Z, Y, X = noisy_shape
                return f"(X, Y, Z, T) = ({X}, {Y}, {Z}, {T})"
            if len(noisy_shape) == 3:
                T, Y, X = noisy_shape
                return f"(X, Y, T) = ({X}, {Y}, {T}) [legacy slice stack]"
            raise ValueError(f"Expected a 3D or 4D TIFF for 3+1D input, received shape {noisy_shape}.")

        if self.dimensionality == "2+1D":
            if len(noisy_shape) != 3:
                raise ValueError(f"Expected a 3D TIFF for 2+1D input, received shape {noisy_shape}.")
            T, Y, X = noisy_shape
            return f"(X, Y, T) = ({X}, {Y}, {T})"

        if len(noisy_shape) != 3:
            raise ValueError(f"Expected a 3D TIFF for {self.dimensionality} input, received shape {noisy_shape}.")
        Z, Y, X = noisy_shape
        return f"(X, Y, Z) = ({X}, {Y}, {Z})"

    def generate_dataloader(
        self,
        noisy_img,
        normalization_mean=None,
        normalization_std=None,
    ):
        """"""
        if self.pad: noisy_img = np.pad(noisy_img, (((self.local_model.in_channels)//2, (self.local_model.in_channels)//2), (0,0), (0,0)), 'reflect')
        noisy_img = torch.from_numpy(noisy_img).type(torch.FloatTensor)
        if normalization_mean is None and normalization_std is None:
            normalization_mean = self.normalization_mean
            normalization_std = self.normalization_std
        test_set = DatasetSUPPORT_test_stitch(
            noisy_img=noisy_img, 
            patch_size=self.patch_size, 
            patch_interval=self.patch_interval,
            transform=None,
            random_patch=False,
            load_to_memory=True,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )
        self.dataloader = DataLoader(
            dataset=test_set, 
            **self._dataloader_kwargs(),
        )

    def _crop_padding(self, denoised_stack):
        if not self.pad:
            return denoised_stack
        crop_margin = (self.local_model.in_channels) // 2
        if crop_margin == 0:
            return denoised_stack
        return denoised_stack[crop_margin:-crop_margin, :, :]

    def save_inference(self, out_path, denoised_img, axes):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            output_mode = stat.S_IMODE(out_path.stat().st_mode)
        else:
            current_umask = os.umask(0)
            os.umask(current_umask)
            output_mode = 0o666 & ~current_umask
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{out_path.name}.",
            suffix=".tmp.tif",
            dir=out_path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            tifffile.imwrite(
                temporary_path,
                np.asarray(denoised_img, dtype=np.float32),
                imagej=True,
                metadata={'axes': axes},
            )
            os.chmod(temporary_path, output_mode)
            os.replace(temporary_path, out_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _reset_loss_tracking(self):
        self._loss_l1_sum = 0.0
        self._loss_l2_sum = 0.0
        self._loss_element_count = 0

    def _record_validation_loss(self, denoised_img, noisy_img_target):
        difference = denoised_img.float() - noisy_img_target.float()
        self._loss_l1_sum += torch.abs(difference).sum().item()
        self._loss_l2_sum += torch.square(difference).sum().item()
        self._loss_element_count += noisy_img_target.numel()

    def _loss_metrics(self):
        if self._loss_element_count == 0:
            raise RuntimeError("Validation produced no output pixels")
        loss_l1 = self._loss_l1_sum / self._loss_element_count
        loss_l2 = self._loss_l2_sum / self._loss_element_count
        return {
            "loss_total": self.loss_coef[0] * loss_l1 + self.loss_coef[1] * loss_l2,
            "loss_l1": loss_l1,
            "loss_l2": loss_l2,
            "loss_l1_sum": self._loss_l1_sum,
            "loss_l2_sum": self._loss_l2_sum,
            "element_count": self._loss_element_count,
        }

    def _sprt_output_path(self, stem):
        if self.output_path is not None:
            return str(self.output_path)
        stem = str(stem)
        if stem.lower().endswith('_sprt'):
            stem = stem[:-5]
        return str(Path(self.out_dir) / f'{stem}_SPRT.tif')

    def _create_cache_dir(self, label):
        cache_root = Path(self.out_dir) / ".support_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"{self.exp_name}_{label}_", dir=cache_root))

    def _cleanup_cache_dir(self, cache_dir):
        if cache_dir is None:
            return
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_root = cache_dir.parent
        try:
            cache_root.rmdir()
        except OSError:
            pass

    def _run_inference_on_stack(
        self,
        noisy_img,
        normalization_mean=None,
        normalization_std=None,
    ):
        self.generate_dataloader(
            noisy_img,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )
        self.test()
        denoised_stack = self._crop_padding(self.denoised_stack)
        if hasattr(self, "dataloader"):
            del self.dataloader
        if hasattr(self, "denoised_stack"):
            del self.denoised_stack
        if self.cuda:
            torch.cuda.empty_cache()
        return denoised_stack

    def _run_inference_on_4d_stack(self, noisy_img):
        normalization_mean = self.normalization_mean
        normalization_std = self.normalization_std
        if normalization_mean is None and normalization_std is None:
            normalization_mean, normalization_std, _ = dataset_stats([noisy_img])
            normalization_mean = float(normalization_mean.item())
            normalization_std = float(normalization_std.item())

        cache_dir = self._create_cache_dir("4d_volume_cache")
        total_slices = noisy_img.shape[1]
        try:
            for z_idx in range(total_slices):
                print(f'  Slice {z_idx+1}/{total_slices}')
                denoised_slice = self._run_inference_on_stack(
                    np.ascontiguousarray(noisy_img[:, z_idx, :, :]),
                    normalization_mean=normalization_mean,
                    normalization_std=normalization_std,
                )
                np.save(cache_dir / f"slice_{z_idx:05d}.npy", denoised_slice.astype(np.float32))

            stitched_img = np.empty(
                (noisy_img.shape[0], total_slices, noisy_img.shape[2], noisy_img.shape[3]),
                dtype=np.float32,
            )
            for z_idx in range(total_slices):
                stitched_img[:, z_idx, :, :] = np.load(cache_dir / f"slice_{z_idx:05d}.npy", mmap_mode='r')
            return stitched_img
        finally:
            self._cleanup_cache_dir(cache_dir)

    def _single_stack_axes(self):
        if self.dimensionality == "3D":
            return 'ZYX'
        return 'TYX'
        
            
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
            validation_mask = (
                np.zeros(denoised_stack.shape, dtype=bool)
                if self.track_loss
                else None
            )
            for _, (noisy_img, _, single_coordinate) in enumerate(tqdm(self.dataloader, desc="validate")):
                noisy_img = noisy_img.to(self.device) #[b, z, y, x]
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

                    denoised_patch = noisy_img_denoised[bi].squeeze()[
                        patch_start_h:patch_end_h,
                        patch_start_w:patch_end_w,
                    ]
                    denoised_stack[stack_start_s+(T//2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = denoised_patch.cpu()
                    if self.track_loss:
                        validation_mask[
                            stack_start_s + (T // 2),
                            stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w,
                        ] = True
                del noisy_img_denoised
                del noisy_img

            if self.track_loss:
                noisy_stack = self.dataloader.dataset.noisy_img.numpy()
                # Compare only pixels owned by the final stitched result. This
                # avoids double-counting overlap or including unfilled borders.
                self._record_validation_loss(
                    torch.from_numpy(denoised_stack[validation_mask]),
                    torch.from_numpy(noisy_stack[validation_mask]),
                )
            # Denormalize without allocating full-size multiplication and
            # addition intermediates.
            print("Denormalizing inference output in place...", flush=True)
            self.denoised_stack = _denormalize_in_place(
                denoised_stack,
                self.dataloader.dataset.std_img.numpy(),
                self.dataloader.dataset.mean_img.numpy(),
            )
            print("Finished denormalizing inference output.", flush=True)

    def _infer_single_input(self, noisy_img):
        if self.dimensionality == "3+1D":
            if noisy_img.ndim == 4:
                return self._run_inference_on_4d_stack(noisy_img), 'TZYX'
            if noisy_img.ndim == 3:
                return self._run_inference_on_stack(noisy_img), self._single_stack_axes()
            raise ValueError(f"Expected a 3D or 4D TIFF for 3+1D input, received shape {noisy_img.shape}.")

        if noisy_img.ndim != 3:
            raise ValueError(
                f"Expected a 3D TIFF for {self.dimensionality} input, received shape {noisy_img.shape}."
            )
        return self._run_inference_on_stack(noisy_img), self._single_stack_axes()

    def run_test(self):
        """
        """
        if self.is_dir:
            noisy_paths = sorted(
                path for path in Path(self.in_path).iterdir()
                if path.is_file() and path.suffix.lower() in {'.tif', '.tiff'}
            )
            if len(noisy_paths) == 0:
                raise FileNotFoundError(f'No TIFF files were found in {self.in_path}.')

            if self.dimensionality == "3+1D":
                first_img = self._read_stack(noisy_paths[0])
                if first_img.ndim == 3:
                    print("\033[1;31mRunning inference on\033[0m")
                    print(f'\033[1m{self.exp_name}\033[0m  [legacy 3+1D slice directory]')
                    cache_dir = self._create_cache_dir("legacy_3p1d_cache")
                    try:
                        for i, path in enumerate(noisy_paths):
                            noisy_img = first_img if i == 0 else self._read_stack(path)
                            if noisy_img.ndim != 3:
                                raise ValueError(
                                    f"Mixed TIFF dimensionalities found in {self.in_path}. "
                                    "Legacy 3+1D slice directories must contain only 3D TIFFs."
                                )
                            noisy_stem = str(path.stem)
                            print(f'  Slice stack \033[1m{noisy_stem}\033[0m  {self._describe_stack(noisy_img.shape)}')
                            denoised_slice = self._run_inference_on_stack(noisy_img)
                            np.save(cache_dir / f"slice_{i:05d}.npy", denoised_slice.astype(np.float32))

                        denoised_img = np.empty(
                            (first_img.shape[0], len(noisy_paths), first_img.shape[1], first_img.shape[2]),
                            dtype=np.float32,
                        )
                        for i in range(len(noisy_paths)):
                            denoised_img[:, i, :, :] = np.load(cache_dir / f"slice_{i:05d}.npy", mmap_mode='r')
                    finally:
                        self._cleanup_cache_dir(cache_dir)
                    self.out_path = self._sprt_output_path(self.exp_name)
                    print(f'Saving denoised \033[1m{self.exp_name}\033[0m to {self.out_path}')
                    self.save_inference(self.out_path, denoised_img, 'TZYX')
                    return

                if first_img.ndim != 4:
                    raise ValueError(f"Expected 4D TIFF stacks for 3+1D directory input, received shape {first_img.shape}.")

                for i, path in enumerate(noisy_paths):
                    noisy_img = first_img if i == 0 else self._read_stack(path)
                    if noisy_img.ndim != 4:
                        raise ValueError(
                            f"Mixed TIFF dimensionalities found in {self.in_path}. "
                            "3+1D stack directories must contain only 4D TIFFs."
                        )
                    noisy_stem = str(path.stem)
                    self.out_path = self._sprt_output_path(noisy_stem)
                    print("\033[1;31mRunning inference on\033[0m")
                    print(f'\033[1m{noisy_stem}\033[0m  {self._describe_stack(noisy_img.shape)}')
                    denoised_img, axes = self._infer_single_input(noisy_img)
                    print(f'Saving denoised \033[1m{noisy_stem}\033[0m to {self.out_path}')
                    self.save_inference(self.out_path, denoised_img, axes)
                return

            for path in noisy_paths:
                noisy_stem = str(path.stem)
                noisy_img = self._read_stack(path)
                self.out_path = self._sprt_output_path(noisy_stem)
                print("\033[1;31mRunning inference on\033[0m")
                print(f'\033[1m{noisy_stem}\033[0m  {self._describe_stack(noisy_img.shape)}')
                denoised_img, axes = self._infer_single_input(noisy_img)
                print(f'Saving denoised \033[1m{noisy_stem}\033[0m to {self.out_path}')
                self.save_inference(self.out_path, denoised_img, axes)
        else:
            noisy_path = Path(self.in_path)
            noisy_stem = str(noisy_path.stem)
            self.out_path = self._sprt_output_path(self.exp_name)
            noisy_img = self._read_stack(noisy_path)
            print(f'Running inference on \033[1m{noisy_stem}\033[0m  {self._describe_stack(noisy_img.shape)}')
            denoised_img, axes = self._infer_single_input(noisy_img)
            print(f'Saving denoised \033[1m{noisy_stem}\033[0m to {self.out_path}')
            self.save_inference(self.out_path, denoised_img, axes)
