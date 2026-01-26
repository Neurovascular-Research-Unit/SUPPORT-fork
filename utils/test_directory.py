import numpy as np
import torch
import skimage.io as skio
import os
import os.path

from tqdm import tqdm
from utils.dataset import DatasetSUPPORT_test_stitch
from model.SUPPORT import SUPPORT


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
                value = eval(value.strip())
                params[key] = value
    return params


if __name__ == '__main__':
    
    ########## TESTING PARAMETERS ##############
    
    # Model to test
    model_name = "4D_at_200um_Cdh5_M_3_months_1.5mg_TRITC_119_v2"
    model_name = 'Cdh5_109_M_FOV1_4D'
    model_epoch = 6  # Which epoch to test
    
    # Input directory containing .tif files to denoise
    foldername = '/data/garfinkeljb/CloudCode/Datasets/Cdh5_109_M_FOV1_4D_z_planes/'
    
    # Output directory for denoised files
    saveheader = f"./results/images/{model_name}_epoch{model_epoch}"
    
    # Handle first/last frames (optional)
    # Options: None, "repeat", "mirror"
    include_first_and_last = None
    
    # Load training parameters automatically
    params_file = f"./results/training_params/{model_name}/training_params.txt"
    training_params = load_training_params(params_file)
    
    # Extract parameters
    patch_size = training_params.get("patch_size", [61, 128, 128])
    patch_interval = training_params.get("patch_interval", [1, 61, 61])
    patch_interval = [1, 48, 48]
    batch_size = training_params.get("batch_size", 16)
    bs_size = training_params.get("bs_size", 3)
    bp_mode = training_params.get("bp_mode", False)
    
    # Construct model file path
    model_file = f'./results/saved_models/{model_name}/model_{model_epoch}.pth'
    
    print("="*60)
    print("TESTING DIRECTORY CONFIGURATION")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Epoch: {model_epoch}")
    print(f"Model file: {model_file}")
    print(f"Input directory: {foldername}")
    print(f"Output directory: {saveheader}")
    print(f"Patch size: {patch_size}")
    print(f"Patch interval: {patch_interval}")
    print(f"Batch size: {batch_size}")
    print(f"Blind-spot size: {bs_size}")
    print(f"Blind-plane mode: {bp_mode}")
    print(f"First/last frame handling: {include_first_and_last}")
    print("="*60)
    
    ##################################################

    # Create output directory
    if not os.path.exists(saveheader):
        print('Save directory created.')
        os.makedirs(saveheader, exist_ok=True)

    # Find all .tif files in input directory
    data_files = []
    output_files = []
    for dirpath, dirnames, filenames in os.walk(foldername):
        for filename in sorted([f for f in filenames if f.endswith(".tif")]):
            print(f'Data file: {filename}')
            data_files.append(os.path.join(dirpath, filename))
            output_files.append(f"{saveheader}/denoised_{filename}")
    
    print(f"\nFound {len(data_files)} TIFF files to process\n")

    # Load model architecture from checkpoint
    checkpoint = torch.load(model_file, weights_only=True)
    checkpoint_last_layer_channels = [
        checkpoint['last_layers.0.weight'].shape[0],
        checkpoint['last_layers.1.weight'].shape[0],
        checkpoint['last_layers.2.weight'].shape[0]
    ]
    checkpoint_enc_layers = [
        checkpoint['enc_layers.0.weight'].shape[0],
        checkpoint['enc_layers.1.weight'].shape[0],
        checkpoint['enc_layers.2.weight'].shape[0],
        checkpoint['enc_layers.3.weight'].shape[0],
        checkpoint['enc_layers.4.weight'].shape[0]
    ]
    
    print(f"Last layer channels: {checkpoint_last_layer_channels}")
    print(f"Encoder layers: {checkpoint_enc_layers}\n")

    # Initialize model
    model = SUPPORT(
        in_channels=patch_size[0],
        mid_channels=checkpoint_enc_layers,
        depth=5,
        blind_conv_channels=64,
        one_by_one_channels=[32, 16],
        last_layer_channels=checkpoint_last_layer_channels,
        bs_size=bs_size,
        bp=bp_mode
    ).cuda()

    model.load_state_dict(checkpoint)

    # Process each file
    for i, (data_file, output_file) in enumerate(zip(data_files, output_files)):
        print(f"\n[{i+1}/{len(data_files)}] Processing: {os.path.basename(data_file)}")
        
        demo_tif = torch.from_numpy(skio.imread(data_file).astype(np.float32)).type(torch.FloatTensor)
        print(f"  Input shape: {demo_tif.shape}")
        
        # Handle first/last frames if requested
        if include_first_and_last == "repeat":
            print(f"  Warning: First and last frames will be repeated (workaround)")
            demo_tif = torch.cat([
                demo_tif[0, :, :].unsqueeze(0).repeat((patch_size[0] // 2, 1, 1)),
                demo_tif,
                demo_tif[-1, :, :].unsqueeze(0).repeat((patch_size[0] // 2, 1, 1)),
            ])
        elif include_first_and_last == "mirror":
            print(f"  Warning: First and last frames will be mirrored (workaround)")
            demo_tif = torch.cat([
                demo_tif[1:(patch_size[0] // 2)+1, :, :].flip(0),
                demo_tif,
                demo_tif[-1 * (patch_size[0] // 2)-1:-1, :, :].flip(0),
            ])

        testset = DatasetSUPPORT_test_stitch(demo_tif, patch_size=patch_size, patch_interval=patch_interval)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
        
        denoised_stack = validate(testloader, model)

        # Remove padding if first/last frames were handled
        if include_first_and_last in ["repeat", "mirror"]:
            denoised_stack = denoised_stack[patch_size[0] // 2:-1 * (patch_size[0] // 2)]

        print(f'  Output shape: {denoised_stack.shape}')
        print(f'  Saving to: {output_file}')
        skio.imsave(output_file, denoised_stack[:, :, :], metadata={'axes': 'TYX'})

    print("\n" + "="*60)
    print(f"TESTING COMPLETE! Processed {len(data_files)} files")
    print("="*60)