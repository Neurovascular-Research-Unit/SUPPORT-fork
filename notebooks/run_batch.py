import sys
from pathlib import Path
sys.path.insert(0, str(Path('./').resolve().parent.parent))
import json
import torch
torch.cuda.empty_cache()
from SUPPORT.utils.train_collection import TrainingClass
from SUPPORT.utils.test_collection import TestingClass
import shutil

def run_all(path):
    in_path = path
    exp_name = None
    dimensionality = "3+1D" # options: "2+1D" (planar + time), "3D" (static volumetric), or "3+1D" (volumetric + time)
    rm_dir = True # if True, do not preserve checkpoints (saves local storage memory)

    # personalization: one-time change, keep after initialization
    out_dir = None
    # node-dependent: change if deviating from typical node specs
    n_cpu = 8
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
    train_dict = {
        'in_path': train_params['in_path'],
        'exp_name': train_params['exp_name'],
        'out_dir': train_params['out_dir'],
        'dimensionality': train_params['dimensionality'],
        'rm_dir': train_params['rm_dir'],
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

    train_cd.run()
    torch.cuda.empty_cache()

    epoch = vars(train_cd)['n_epochs']
    model_path = vars(train_cd)['pth_dir'] + f'/model{epoch:03d}.pth'
    test_params['model_path'] = model_path.replace('/vf/users', '/data')
    test_params['out_dir'] = vars(train_cd)['out_dir'].replace('/vf/users', '/data')
    test_params['exp_name'] = vars(train_cd)['exp_name']
    test_dict = {
        'in_path':          train_params['in_path'],
        'model_path':       test_params['model_path'],
        'exp_name':         test_params['exp_name'],
        'out_dir':          test_params['out_dir'],
        'dimensionality':   train_params['dimensionality'],
        'cuda_device':      train_params['cuda_device'],
        'patch_size':       train_params['patch_size'],
        'patch_interval':   test_params['patch_interval'],
        'batch_size':       train_params['batch_size']*2,
        'depth':            train_params['depth'],
        'bc_channels':      train_params['bc_channels'],
        'obo_channels':     train_params['obo_channels'],
        'll_channels':      train_params['ll_channels'],
        'unet_channels':    train_params['unet_channels'],
        'bs_size':          train_params['bs_size'],
        'n_cpu':            train_params['n_cpu'],
        'prefetch_factor':  train_params['prefetch_factor'],
        'pad':              test_params['pad']
    }
    test_cd = TestingClass(test_dict)

    test_cd.run()
    torch.cuda.empty_cache()
    
paths = list(Path('/data/garfinkeljb/noisy_data').glob('*'))
paths = [str(path) for path in paths]
for path in paths:
    run_all(path)
    shutil.rmtree(f'/data/garfinkeljb/SUPPORT/notebooks/tmp/{str(Path(path).stem)}')
    shutil.rmtree(path)