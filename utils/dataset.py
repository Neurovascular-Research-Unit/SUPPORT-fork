import torch
import numpy as np
import math
from torch.utils.data import Dataset
from tqdm import tqdm

def random_transform(input, target, rng, is_rotate=True):
    """
    Randomly rotate/flip the image

    Arguments:
        input: input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: targer image stack (Pytorch Tensor with dimension [b, T, X, Y]), can be None
        rng: numpy random number generator
    
    Returns:
        input: randomly rotated/flipped input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: randomly rotated/flipped target image stack (Pytorch Tensor with dimension [b, T, X, Y])
    """
    rand_num = rng.integers(0, 4) # random number for rotation
    rand_num_2 = rng.integers(0, 2) # random number for flip

    if is_rotate:
        if rand_num == 1:
            input = torch.rot90(input, k=1, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=1, dims=(2, 3))
        elif rand_num == 2:
            input = torch.rot90(input, k=2, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=2, dims=(2, 3))
        elif rand_num == 3:
            input = torch.rot90(input, k=3, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=3, dims=(2, 3))
    
    if rand_num_2 == 1:
        input = torch.flip(input, dims=[2])
        if target is not None:
            target = torch.flip(target, dims=[2])

    return input, target

def normalize(image):
    """
    Normalize the image to [mean/std]=[0/1]

    Arguments:
        image: image stack (Pytorch Tensor with dimension [T, X, Y])

    Returns:
        image: normalized image stack (Pytorch Tensor with dimension [T, X, Y])
        mean_image: mean of the image stack (np.float)
        std_image: standard deviation of the image stack (np.float)
    """
    mean_image = torch.mean(image)
    std_image = (
        torch.ones_like(mean_image)
        if image.numel() <= 1
        else torch.std(image)
    )
    if not torch.isfinite(std_image) or std_image == 0:
        std_image = torch.ones_like(std_image)

    image -= mean_image
    image /= std_image

    return image, mean_image, std_image


def image_stats(image):
    if hasattr(image, "attrs") and "mean" in image.attrs and "std" in image.attrs:
        mean_image = torch.tensor(image.attrs["mean"], dtype=torch.float32)
        std_image = torch.tensor(image.attrs["std"], dtype=torch.float32)
    elif isinstance(image, torch.Tensor):
        mean_image = torch.mean(image.float())
        std_image = (
            torch.ones_like(mean_image)
            if image.numel() <= 1
            else torch.std(image.float())
        )
    else:
        arr = np.asarray(image, dtype=np.float32)
        mean_image = torch.tensor(np.mean(arr), dtype=torch.float32)
        ddof = 1 if arr.size > 1 else 0
        std_image = torch.tensor(np.std(arr, ddof=ddof), dtype=torch.float32)

    if not torch.isfinite(std_image) or std_image == 0:
        std_image = torch.ones_like(std_image)
    return mean_image.float(), std_image.float()


_NORMALIZATION_CHUNK_ELEMENTS = 1_000_000


def _iter_stat_chunks(image, max_elements=_NORMALIZATION_CHUNK_ELEMENTS):
    """Yield bounded views for statistics without materializing a full memmap."""
    if max_elements < 1:
        raise ValueError("max_elements must be positive")

    def split(value):
        shape = tuple(int(size) for size in value.shape)
        element_count = math.prod(shape)
        if element_count == 0:
            return
        if element_count <= max_elements or len(shape) == 0:
            yield value
            return

        trailing_count = math.prod(shape[1:]) if len(shape) > 1 else 1
        step = max(1, max_elements // max(trailing_count, 1))
        if step > 1:
            for start in range(0, shape[0], step):
                yield value[start:start + step]
            return

        for index in range(shape[0]):
            yield from split(value[index])

    yield from split(image)


def _chunk_moments(chunk):
    """Return ``(count, mean, M2)`` for one chunk using float64 reductions."""
    if isinstance(chunk, torch.Tensor):
        # Training always consumes float32 patches. Match that representation
        # before using float64 for stable moment calculations.
        values = chunk.detach().to(dtype=torch.float32).to(dtype=torch.float64)
        count = values.numel()
        variance, mean = torch.var_mean(values, unbiased=False)
        mean = float(mean.item())
        m2 = float(variance.item()) * count
    else:
        # Training patches are converted to float32 in __getitem__, so compute
        # statistics from that same representation before reducing in float64.
        values = np.array(chunk, dtype=np.float32, copy=True)
        count = values.size
        mean = float(np.mean(values, dtype=np.float64))
        m2 = float(np.var(values, dtype=np.float64)) * count

    if not math.isfinite(mean) or not math.isfinite(m2):
        raise ValueError("Training data contains non-finite values")
    return count, mean, max(m2, 0.0)


def dataset_stats(images, max_chunk_elements=_NORMALIZATION_CHUNK_ELEMENTS):
    """Compute a voxel-weighted mean and sample std across all image stacks.

    Parallel Welford aggregation keeps the calculation numerically stable while
    bounded chunks avoid loading an entire memory-mapped training corpus at once.
    """
    total_count = 0
    total_mean = 0.0
    total_m2 = 0.0

    for image in images:
        for chunk in _iter_stat_chunks(image, max_chunk_elements):
            chunk_count, chunk_mean, chunk_m2 = _chunk_moments(chunk)
            if chunk_count == 0:
                continue
            combined_count = total_count + chunk_count
            delta = chunk_mean - total_mean
            total_m2 += (
                chunk_m2
                + delta * delta * total_count * chunk_count / combined_count
            )
            total_mean += delta * chunk_count / combined_count
            total_count = combined_count

    if total_count == 0:
        raise ValueError("Cannot normalize an empty training dataset")

    variance = total_m2 / (total_count - 1) if total_count > 1 else 0.0
    total_std = math.sqrt(max(variance, 0.0))
    if not math.isfinite(total_mean) or not math.isfinite(total_std):
        raise ValueError("Training data produced non-finite normalization statistics")
    if total_std == 0:
        total_std = 1.0

    return (
        torch.tensor(total_mean, dtype=torch.float64),
        torch.tensor(total_std, dtype=torch.float64),
        total_count,
    )


def get_coordinate(img_size, patch_size, patch_interval):
    """DeepCAD version of stitching
    https://github.com/cabooster/DeepCAD/blob/53a9b8491170e298aa7740a4656b4f679ded6f41/DeepCAD_pytorch/data_process.py#L374
    """
    #print(img_size, patch_size, patch_interval)
    whole_s, whole_h, whole_w = img_size
    img_s, img_h, img_w = patch_size
    gap_s, gap_h, gap_w = patch_interval

    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s - gap_s)/2

    # print(whole_s, whole_h, whole_w)
    # print(img_s, img_h, img_w)
    # print(gap_s, gap_h, gap_w)

    num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
    num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
    num_s = math.ceil((whole_s-img_s+gap_s)/gap_s)

    coordinate_list = []
    for x in range(0,num_h):
        for y in range(0,num_w):
            for z in range(0,num_s):
                single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                if x != (num_h-1):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                elif x == (num_h-1):
                    init_h = whole_h - img_h
                    end_h = whole_h

                if y != (num_w-1):
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                elif y == (num_w-1):
                    init_w = whole_w - img_w
                    end_w = whole_w

                if z != (num_s-1):
                    init_s = gap_s*z
                    end_s = gap_s*z + img_s
                elif z == (num_s-1):
                    init_s = whole_s - img_s
                    end_s = whole_s
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    if num_w > 1:
                        single_coordinate['stack_start_w'] = y*gap_w
                        single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w-cut_w
                    else:
                        single_coordinate['stack_start_w'] = 0
                        single_coordinate['stack_end_w'] = img_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w
                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_w-img_w+cut_w
                    single_coordinate['stack_end_w'] = whole_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w
                else:
                    single_coordinate['stack_start_w'] = y*gap_w+cut_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w-cut_w

                if x == 0:
                    if num_h > 1:
                        single_coordinate['stack_start_h'] = x*gap_h
                        single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h-cut_h
                    else:
                        single_coordinate['stack_start_h'] = 0
                        single_coordinate['stack_end_h'] = x*gap_h+img_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h
                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_h-img_h+cut_h
                    single_coordinate['stack_end_h'] = whole_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h
                else:
                    single_coordinate['stack_start_h'] = x*gap_h+cut_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h-cut_h

                if z == 0:
                    if num_s > 1:
                        single_coordinate['stack_start_s'] = z*gap_s
                        single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s-cut_s
                    else:
                        single_coordinate['stack_start_s'] = z*gap_s
                        single_coordinate['stack_end_s'] = z*gap_s+img_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s
                elif z == num_s-1:
                    single_coordinate['stack_start_s'] = whole_s-img_s+cut_s
                    single_coordinate['stack_end_s'] = whole_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s
                else:
                    single_coordinate['stack_start_s'] = z*gap_s+cut_s
                    single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s-cut_s

                coordinate_list.append(single_coordinate)
    return coordinate_list


class DatasetSUPPORT(Dataset):
    def __init__(self, noisy_imgs, patch_size=[61, 128, 128], patch_interval=[10, 64, 64], load_to_memory=True,\
        transform=None, random_patch=True, random_patch_seed=0, normalize_across_images=False,
        normalization_group_ids=None):
        """
        Arguments:
            noisy_imgs: list of noisy image stack ([Tensor with dimension [t, x, y]])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
            normalize_across_images: use one voxel-weighted mean/std for every
                image rather than normalizing each image separately (bool)
            normalization_group_ids: optional group identifier for each image.
                Images with the same identifier share one voxel-weighted
                mean/std. This lets all Z planes derived from one 4-D TIFF use
                that TIFF's statistics while keeping different TIFFs separate.
            algorithm: the algorithm of use (str)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")      
        if type(normalize_across_images) is not bool:
            raise TypeError("normalize_across_images must be True or False")
        if normalization_group_ids is not None:
            normalization_group_ids = list(normalization_group_ids)
            if len(normalization_group_ids) != len(noisy_imgs):
                raise ValueError(
                    "normalization_group_ids must contain one identifier per image"
                )
            if normalize_across_images:
                raise ValueError(
                    "normalization_group_ids cannot be combined with "
                    "normalize_across_images=True"
                )

        if normalize_across_images and load_to_memory:
            noisy_imgs = [
                noisy_img.detach().to(device="cpu", dtype=torch.float32)
                if isinstance(noisy_img, torch.Tensor)
                else torch.from_numpy(
                    np.array(noisy_img, dtype=np.float32, copy=True)
                )
                for noisy_img in noisy_imgs
            ]

        # initialize
        self.data_weight = []
        for noisy_img in noisy_imgs:
            if load_to_memory:
                self.data_weight.append(torch.numel(noisy_img))
            else:
                self.data_weight.append(np.prod(noisy_img.shape))

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.precomputed_indices = None
        self.load_to_memory = load_to_memory
        self.normalize_across_images = normalize_across_images
        self.normalization_group_ids = normalization_group_ids

        self.noisy_imgs = noisy_imgs
        self.dataset_mean = None
        self.dataset_std = None
        self.dataset_count = None
        if self.normalize_across_images:
            (
                self.dataset_mean,
                self.dataset_std,
                self.dataset_count,
            ) = dataset_stats(
                tqdm(noisy_imgs, desc="Computing shared normalization")
            )
            image_count = len(noisy_imgs)
            self.mean_images = self.dataset_mean.repeat(image_count)
            self.std_images = self.dataset_std.repeat(image_count)
            if load_to_memory:
                for noisy_img in tqdm(noisy_imgs):
                    for chunk in _iter_stat_chunks(noisy_img):
                        normalized_chunk = (
                            chunk.detach().to(dtype=torch.float32).to(dtype=torch.float64)
                        )
                        normalized_chunk.sub_(self.dataset_mean).div_(self.dataset_std)
                        chunk.copy_(normalized_chunk)
                print("Normalized all noisy images with one shared dataset mean/std.")
            else:
                print("Stored shared dataset mean/std for on-the-fly normalization.")
        elif self.normalization_group_ids is not None:
            grouped_indices = {}
            for image_index, group_id in enumerate(self.normalization_group_ids):
                try:
                    grouped_indices.setdefault(group_id, []).append(image_index)
                except TypeError as exc:
                    raise TypeError(
                        "normalization_group_ids must contain hashable values"
                    ) from exc

            grouped_stats = {}
            self.mean_images = [None] * len(noisy_imgs)
            self.std_images = [None] * len(noisy_imgs)
            for group_id, image_indices in grouped_indices.items():
                mean_image, std_image, voxel_count = dataset_stats(
                    noisy_imgs[index] for index in image_indices
                )
                # Upstream SUPPORT normalizes float32 input with float32
                # statistics. Keep that representation consistent for both
                # resident and lazy-loading paths.
                mean_image = mean_image.detach().cpu().float()
                std_image = std_image.detach().cpu().float()
                grouped_stats[group_id] = {
                    "mean": mean_image,
                    "std": std_image,
                    "voxel_count": voxel_count,
                }
                for index in image_indices:
                    self.mean_images[index] = mean_image
                    self.std_images[index] = std_image

            self.normalization_group_stats = grouped_stats
            self.mean_images = (
                torch.stack(self.mean_images) if self.mean_images else torch.empty(0)
            )
            self.std_images = (
                torch.stack(self.std_images) if self.std_images else torch.empty(0)
            )
            if load_to_memory:
                for index, noisy_img in enumerate(tqdm(noisy_imgs)):
                    mean_image = self.mean_images[index].to(
                        device=noisy_img.device,
                        dtype=noisy_img.dtype,
                    )
                    std_image = self.std_images[index].to(
                        device=noisy_img.device,
                        dtype=noisy_img.dtype,
                    )
                    noisy_img.sub_(mean_image).div_(std_image)
                print("Normalized noisy images with one mean/std per source TIFF.")
            else:
                print("Stored one mean/std per source TIFF for on-the-fly normalization.")
        else:
            self.mean_images = []
            self.std_images = []
            for idx, noisy_img in enumerate(tqdm(noisy_imgs)):
                if load_to_memory:
                    noisy_img, mean_image, std_image = normalize(noisy_img)
                    self.noisy_imgs[idx] = noisy_img
                else:
                    mean_image, std_image = image_stats(noisy_img)
                self.mean_images.append(mean_image.detach().cpu().float())
                self.std_images.append(std_image.detach().cpu().float())

            self.mean_images = torch.stack(self.mean_images) if self.mean_images else torch.empty(0)
            self.std_images = torch.stack(self.std_images) if self.std_images else torch.empty(0)
            if load_to_memory:
                print("Normalized noisy images and stored mean/std in memory.")
                #print(f"Mean: {self.mean_images}, Std: {self.std_images}")
            else:
                print("Stored mean/std for on-the-fly normalization.")

        # generate index
        self.indices_ds = []
        for noisy_img in self.noisy_imgs:
            indices = []
            tmp_size = noisy_img.shape
            if np.any(tmp_size < np.array(self.patch_size)):
                raise Exception("patch size is larger than data size")

            for k in range(3):
                z_range = list(range(0, tmp_size[k]-self.patch_size[k]+1, self.patch_interval[k]))
                if tmp_size[k] - self.patch_size[k] > z_range[-1]:
                    z_range.append(tmp_size[k]-self.patch_size[k])
                indices.append(z_range)
            self.indices_ds.append(indices)

    def precompute_indices(self):
        """
        Precompute random patch indices for each image using vectorized operations.
        This function accounts for images of different sizes by generating random
        indices within the valid range for each image.
        """
        precomputed_indices = []
        
        # Iterate over each image in the dataset
        for ds_idx, noisy_img in enumerate(self.noisy_imgs):
            # Get the shape of the image (T, H, W)
            shape = noisy_img.shape
            
            # Determine the number of patches available for this image.
            # Here, we use the precomputed indices list from __init__ (indices_ds)
            # which was generated based on patch_size and patch_interval.
            indices_lists = self.indices_ds[ds_idx]
            count_i = len(indices_lists[0]) * len(indices_lists[1]) * len(indices_lists[2])
            
            # Calculate the valid range for each dimension
            t_range = shape[0] - self.patch_size[0] + 1
            y_range = shape[1] - self.patch_size[1] + 1
            z_range = shape[2] - self.patch_size[2] + 1
            
            # Generate random indices in a vectorized way for the current image
            t_indices = self.patch_rng.integers(0, t_range, size=count_i)
            y_indices = self.patch_rng.integers(0, y_range, size=count_i)
            z_indices = self.patch_rng.integers(0, z_range, size=count_i)
            
            # Create a list of tuples (ds_idx, t_idx, y_idx, z_idx) for this image
            indices_for_image = [(ds_idx, int(t), int(y), int(z))
                                for t, y, z in zip(t_indices, y_indices, z_indices)]
            precomputed_indices.extend(indices_for_image)
        
        # Shuffle the complete list of precomputed indices to randomize order per epoch
        self.patch_rng.shuffle(precomputed_indices)
        self.precomputed_indices = precomputed_indices

    def __len__(self):
        total = 0
        for indices in self.indices_ds:
            total += len(indices[0]) * len(indices[1]) * len(indices[2])

        return total

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            ds_idx, t_idx, y_idx, z_idx = self.precomputed_indices[i]
        else:
            ds_idx = 0
            t_idx = self.indices_ds[ds_idx][0][i // (len(self.indices_ds[ds_idx][1]) * len(self.indices_ds[ds_idx][2]))]
            y_idx = self.indices_ds[ds_idx][1][(i % (len(self.indices_ds[ds_idx][1]) * len(self.indices_ds[ds_idx][2]))) // len(self.indices_ds[ds_idx][2])]
            z_idx = self.indices_ds[ds_idx][2][i % len(self.indices_ds[ds_idx][2])]

        # input dataset range
        t_range = slice(t_idx, t_idx + self.patch_size[0])
        y_range = slice(y_idx, y_idx + self.patch_size[1])
        z_range = slice(z_idx, z_idx + self.patch_size[2])
        
        if self.load_to_memory:
            noisy_img = self.noisy_imgs[ds_idx][t_range, y_range, z_range]
        else:
            img_obj = self.noisy_imgs[ds_idx]
            # if on-disk object with attrs (e.g., zarr/tiff), use slicing and convert
            if hasattr(img_obj, "attrs"):
                noisy_patch = img_obj[t_range, y_range, z_range]
                noisy_img = torch.from_numpy(np.array(noisy_patch, dtype=np.float32, copy=True))
            else:
                if isinstance(img_obj, torch.Tensor):
                    noisy_img = img_obj[t_range, y_range, z_range].to(torch.float32)
                else:
                    noisy_patch = img_obj[t_range, y_range, z_range]
                    noisy_img = torch.from_numpy(np.array(noisy_patch, dtype=np.float32, copy=True))
            noisy_img = noisy_img.to(torch.float64)
            noisy_img.sub_(self.mean_images[ds_idx]).div_(self.std_images[ds_idx])
            noisy_img = noisy_img.to(torch.float32)
        return noisy_img, torch.tensor([[t_idx, t_idx + self.patch_size[0]],\
            [y_idx, y_idx + self.patch_size[1]], [z_idx, z_idx + self.patch_size[2]]]), torch.tensor(ds_idx)

class DatasetSUPPORT_test_stitch(Dataset):
    def __init__(self, noisy_img, patch_size=[61, 128, 128], patch_interval=[10, 64, 64], load_to_memory=True,\
        transform=None, random_patch=False, random_patch_seed=0,
        normalization_mean=None, normalization_std=None):
        """
        Arguments:
            noisy_img: noisy image stack (Tensor with dimension [t, x, y])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
            normalization_mean: optional fixed scalar mean (float)
            normalization_std: optional fixed positive scalar std (float)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        if not isinstance(noisy_img, torch.Tensor):
            noisy_img = torch.from_numpy(np.asarray(noisy_img, dtype=np.float32)).type(torch.FloatTensor)
        else:
            noisy_img = noisy_img.to(torch.float32)
        self.noisy_img = noisy_img
        self.load_to_memory = load_to_memory
        if (normalization_mean is None) != (normalization_std is None):
            raise ValueError(
                "normalization_mean and normalization_std must be provided together"
            )
        if normalization_mean is None:
            self.noisy_img, self.mean_img, self.std_img = normalize(
                self.noisy_img
            )
        else:
            self.mean_img = torch.as_tensor(
                normalization_mean,
                dtype=torch.float32,
            )
            self.std_img = torch.as_tensor(
                normalization_std,
                dtype=torch.float32,
            )
            if self.mean_img.numel() != 1 or self.std_img.numel() != 1:
                raise ValueError("normalization mean and std must be scalars")
            if (
                not torch.isfinite(self.mean_img)
                or not torch.isfinite(self.std_img)
                or self.std_img <= 0
            ):
                raise ValueError("normalization mean/std must be finite with std > 0")
            self.noisy_img.sub_(self.mean_img).div_(self.std_img)

        # generate index
        self.indices = []
        tmp_size = self.noisy_img.size()
        if np.any(tmp_size < np.array(self.patch_size)):
            raise Exception("patch size is larger than data size")

        self.indices = get_coordinate(tmp_size, patch_size, patch_interval)            

    def __len__(self):
        return len(self.indices) # len(self.indices[0]) * len(self.indices[1]) * len(self.indices[2])

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            idx = self.patch_rng.integers(0, len(self.indices) - 1)
        else:
            idx = i
        single_coordinate = self.indices[idx]
        
        # input dataset range
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        # for stitching dataset range
        noisy_img = self.noisy_img[init_s:end_s,init_h:end_h,init_w:end_w]

        # transform
        if self.transform:
            rand_i = self.patch_rng.integers(0, self.transform.n_masks)
            rand_t = self.patch_rng.integers(0, 2)
            noisy_img = self.transform.mask(noisy_img, rand_i, rand_t)

        return noisy_img, torch.empty(1), single_coordinate
