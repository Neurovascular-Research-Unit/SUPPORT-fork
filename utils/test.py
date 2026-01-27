import numpy as np
import torch
import skimage.io as skio
import configparser

from tqdm import tqdm
from utils.dataset import DatasetSUPPORT_test_stitch
from support.SUPPORT import SUPPORT




def validate(test_dataloader, model):
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
        model.eval()
        # initialize denoised stack to NaN array.
        denoised_stack = np.zeros(test_dataloader.dataset.noisy_image.shape, dtype=np.float32)
        
        # stitching denoised stack
        # insert the results if the stack value was NaN
        # or, half of the output volume
        for _, (noisy_image, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="validate")):
            noisy_image = noisy_image.cuda() #[b, z, y, x]
            noisy_image_denoised = model(noisy_image)
            T = noisy_image.size(1)
            for bi in range(noisy_image.size(0)): 
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
                    = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

        # change nan values to 0 and denormalize
        denoised_stack = denoised_stack * test_dataloader.dataset.std_image.numpy() + test_dataloader.dataset.mean_image.numpy()

        return denoised_stack

# Function to load training parameters from training_params.txt
def load_training_params(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                key = key.strip()
                value = eval(value.strip())  # Convert string to appropriate type
                params[key] = value
    return params

if __name__ == '__main__':

    ########## TESTING PARAMETERS ##############
    import os
    
    # Model to test
    model_name = "4D_at_200um_Cdh5_M_3_months_1.5mg_TRITC_119"
    model_name = "Cdh5_109_M_FOV1_4Dadsjkasdhkja"
    #model_name = /data/garfinkeljb/CloudCode/SUPPORT_TEST/results/images/Cdh5_109_M_FOV1_4D
    model_name = '20251007_FOV1_2D_Ch2'
    model_name = 'GC8_221_ChronicCrani'
    model_name = 'Cdh5_40 M_20250113_FOV1_4D_Ch1_Z000'
    model_epoch = 19  # Which epoch to test (e.g., 9 for model_9.pth)
    
    # Input data file to denoise
    # For (2+1)D: Use 3D file with shape (T, Y, X)
    # For (3+1)D: Use 4D file with shape (T, Z, Y, X) OR flattened 3D (T*Z, Y, X)
    data_file = "/data/garfinkeljb/CloudCode/Datasets/4D_timepoints/"
    data_file = '/data/garfinkeljb/CloudCode/Datasets/Cdh5_109_M_FOV1_4D_z_planes/z_plane_000.tif'
    data_file = '/data/garfinkeljb/CloudCode/Datasets/20251007_FOV1_2D_Ch2.tif'
    data_file = '/data/garfinkeljb/CloudCode/Datasets/Ch2/GC8_221_ChronicCrani_2025_1110_BaselineImage2_FOV1_250Deep_2DT-001_Ch2.tif'
    data_file = '/data/garfinkeljb/CloudCode/Datasets/Ch1/Cdh5 40 M 20250113_FOV1 4D-137_Ch1_z_slice_000/Cdh5 40 M 20250113_FOV1 4D-137_Ch1_z_slice_000.tif'
    
    # Output file
    output_file = f'./results/images/{model_name}_epoch{model_epoch}_denoised.tif'
    
    # Load training parameters (patch_size, patch_interval, etc.)
    # These MUST match what was used during training!
    params_file = f"./results/training_params/{model_name}/training_params.txt"  # FIXED: Added f before the string
    training_params = load_training_params(params_file)
    
    # Extract parameters
    patch_size = training_params.get("patch_size", [61, 128, 128])
    patch_interval = training_params.get("patch_interval", [1, 61, 61])
    batch_size = training_params.get("batch_size", 16)
    bs_size = training_params.get("bs_size", 3)
    bp_mode = training_params.get("bp_mode", False)
    
    # Construct model file path
    model_file = f'./results/saved_models/{model_name}/model_{model_epoch}.pth'
    
    print("="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Epoch: {model_epoch}")
    print(f"Model file: {model_file}")
    print(f"Input data: {data_file}")
    print(f"Output file: {output_file}")
    print(f"Patch size: {patch_size}")
    print(f"Patch interval: {patch_interval}")
    print(f"Batch size: {batch_size}")
    print(f"Blind-spot size: {bs_size}")
    print(f"Blind-plane mode: {bp_mode}")
    print("="*60)
    
    ##################################################

    checkpoint = torch.load(model_file, weights_only=True)
    checkpoint_last_layer_channels = [checkpoint['last_layers.0.weight'].shape[0], checkpoint['last_layers.1.weight'].shape[0], checkpoint['last_layers.2.weight'].shape[0]]
    checkpoint_enc_layers = [checkpoint['enc_layers.0.weight'].shape[0], checkpoint['enc_layers.1.weight'].shape[0], checkpoint['enc_layers.2.weight'].shape[0], checkpoint['enc_layers.3.weight'].shape[0], checkpoint['enc_layers.4.weight'].shape[0]]
    checkpoint_dec_layers = [checkpoint['dec_layers.0.weight'].shape[0], checkpoint['dec_layers.1.weight'].shape[0], checkpoint['dec_layers.2.weight'].shape[0], checkpoint['dec_layers.3.weight'].shape[0]]
    
    print(f"Last layer channels: {checkpoint_last_layer_channels}")
    print(f"Encoder layers: {checkpoint_enc_layers}")
    print(f"Decoder layers: {checkpoint_dec_layers}")
    
    model = SUPPORT(in_channels=patch_size[0], mid_channels=checkpoint_enc_layers, depth=5,
            blind_conv_channels=64, one_by_one_channels=[32, 16], last_layer_channels=checkpoint_last_layer_channels, bs_size=bs_size, bp=bp_mode).cuda()

    model.load_state_dict(checkpoint)

    print(f"\nLoading test data from: {data_file}")
    demo_tif = torch.from_numpy(skio.imread(data_file).astype(np.float32)).type(torch.FloatTensor)
    print(f"Input data shape: {demo_tif.shape}")

    testset = DatasetSUPPORT_test_stitch(demo_tif, patch_size=patch_size, patch_interval=patch_interval)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    
    print("\nRunning denoising...")
    denoised_stack = validate(testloader, model)

    crop_margin = (model.in_channels - 1) // 2
    denoised_stack_cropped = denoised_stack[crop_margin:-crop_margin, :, :]
    
    print(f"\nDenoised stack shape: {denoised_stack_cropped.shape}")
    print(f"Saving to: {output_file}")
    skio.imsave(output_file, denoised_stack_cropped, metadata={'axes': 'TYX'})
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
