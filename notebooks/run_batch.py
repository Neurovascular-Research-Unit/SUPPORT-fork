"""Change these for your setup"""
path_SUPPORT = "/data/garfinkeljb/SUPPORT" # NO ENDING /
path_vesselFM = "/vf/users/garfinkeljb/vesselFM" # NO ENDING /


import sys
from pathlib import Path
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.chdir(f'{path_SUPPORT}/notebooks/')
sys.path.insert(0, str(Path('./').resolve().parent.parent))

import gc
import json
import subprocess
import os
import torch
torch.cuda.empty_cache()
from SUPPORT.utils.train_collection import TrainingClass
from SUPPORT.utils.test_collection import TestingClass
import shutil

def run_all(path):
    os.chdir(f'{path_SUPPORT}/notebooks/')
    in_path = path
    exp_name = None
    dimensionality = "2+1D" # options: "2+1D" (planar + time), "3D" (static volumetric), or "3+1D" (volumetric + time)
    segment = False
    rm_dir = True # if True, store training artifacts under notebooks/tmp
    checkpoints = True
    resume = True
    rm_intermediaries = True

    # personalization: one-time change, keep after initialization
    out_dir = None
    # node-dependent: change if deviating from typical node specs
    n_cpu = 16
    gpu = '0'

    with open('./parameters.json', 'r') as file: params = json.load(file)
    train_params = params[dimensionality]['train']
    test_params = params[dimensionality]['test']

    train_params['dimensionality'] = dimensionality
    train_params['rm_dir'] = rm_dir
    train_params['out_dir'] = out_dir
    train_params['n_cpu'] = n_cpu
    train_params['cuda_device'] = gpu
    train_params['in_path'] = in_path
    train_params['exp_name'] = exp_name
    train_params['checkpoints'] = checkpoints
    train_params['resume'] = resume
    train_params['rm_intermediaries'] = rm_intermediaries
    train_dict = {
        'in_path': train_params['in_path'],
        'exp_name': train_params['exp_name'],
        'out_dir': train_params['out_dir'],
        'dimensionality': train_params['dimensionality'],
        'rm_dir': train_params['rm_dir'],
        'checkpoints': train_params['checkpoints'],
        'resume': train_params['resume'],
        'rm_intermediaries': train_params['rm_intermediaries'],
        'n_epochs': train_params['n_epochs'],
        'cuda_device': train_params['cuda_device'],
        'patch_size':train_params['patch_size'],
        'patch_interval':train_params['patch_interval'],
        'batch_size': train_params['batch_size'],
        'depth': train_params['depth'],
        'bc_channels': train_params['bc_channels'],
        'obo_channels': train_params['obo_channels'],
        'll_channels': train_params['ll_channels'],
        'unet_channels': train_params['unet_channels'],
        'bs_size': train_params['bs_size'],
        'lr': train_params['lr'],
        'loss_coef': train_params['loss_coef'],
        'n_cpu': train_params['n_cpu'],
        'prefetch_factor': train_params['prefetch_factor']
    }
    train_cd = TrainingClass(train_dict)
    train_cd.vars_dict()
    try:
        train_cd.run()

        model_path = str(train_cd.model_path)
        test_params['model_path'] = model_path.replace('/vf/users', '/data')
        test_params['out_dir'] = vars(train_cd)['out_dir'].replace('/vf/users', '/data')
        test_params['exp_name'] = vars(train_cd)['exp_name']
    finally:
        train_cd.release_resources()
        del train_cd
    gc.collect()
    torch.cuda.empty_cache()
    test_dict = {
        'in_path':          train_params['in_path'],
        'model_path':       test_params['model_path'],
        'exp_name':         test_params['exp_name'],
        'out_dir':          test_params['out_dir'],
        'dimensionality':   train_params['dimensionality'],
        'cuda_device':      train_params['cuda_device'],
        'patch_size':       train_params['patch_size'],
        'patch_interval':   test_params['patch_interval'],
        'batch_size':       test_params['batch_size'],
        'depth':            train_params['depth'],
        'bc_channels':      train_params['bc_channels'],
        'obo_channels':     train_params['obo_channels'],
        'll_channels':      train_params['ll_channels'],
        'unet_channels':    train_params['unet_channels'],
        'bs_size':          train_params['bs_size'],
        'n_cpu':            train_params['n_cpu'],
        'prefetch_factor':  test_params['prefetch_factor'],
        'pad':              test_params['pad']
    }
    test_cd = TestingClass(test_dict)
    try:
        test_cd.run()
    finally:
        test_cd.release_resources()
    torch.cuda.empty_cache()
    
    if dimensionality in ("3+1D", "3D") and segment:
        print('Running file through vesselFM for vessel segmentation...\n')
        out_path = test_cd.out_path
        seg_out_dir = test_params['out_dir']
        del test_cd
        gc.collect()
        torch.cuda.empty_cache()
        subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "vesselfm",
                "python",
                "-m",
                "vesselfm.seg.inference",
                f"image_path={out_path}",
                "image_file_ending=tif",
                f"output_folder={seg_out_dir}",
                "device=cuda:0",
            ],
            check=True,
            cwd=path_vesselFM,
            env=os.environ.copy(),
        )
    gc.collect()
    torch.cuda.empty_cache()


noisy_data_dir = '/data/garfinkeljb/data_noisy/'

paths = list(Path(noisy_data_dir).glob('*.tif'))
paths = [str(path) for path in paths]
num_paths = len(paths)
for i,path in enumerate(paths):
    if not Path(path).is_dir():
        print(f'Denoising {i} of {num_paths}')
        run_all(path)
        try: shutil.rmtree(f'{path_SUPPORT}/notebooks/tmp/{str(Path(path).stem)}')
        except: pass
        try: shutil.rmtree(path)
        except: pass
