import json
import os
import sys
from pathlib import Path

############################################################
#### CHANGE THIS PATH TO THE PATH OF YOUR SUPPORT REPO #####
############################################################
path_support = Path("/Users/garfinkeljb/Downloads/s/SUPPORT")
#############################################################
os.chdir(path_support / "notebooks")
sys.path.insert(0, str(path_support.parent))

import numpy as np
import tifffile
import torch

from SUPPORT.utils.salt import simulate_lower_power
from SUPPORT.utils.test_collection import TestingClass
from SUPPORT.utils.train_collection import TrainingClass


# Basic configuration. Model parameters are read from parameters.json.
dir_noisy = "/Volumes/Amreen_Mughal_lab/20260921_snrTest/01_data"
dir_noisy = Path(dir_noisy)
dimensionality = "3+1D"
power_percent = [100, 90, 80, 70, 60, 50, 40]

# Node-dependent settings.
n_cpu = 8
gpu = "0"

rm_tmp = True # removes each raw + noise file after denoising
rm_dir = False
checkpoints = False
resume = True
rm_intermediaries = True


def main():
    torch.cuda.empty_cache()

    paths_noisy = sorted(dir_noisy.rglob("*.tif"))
    if not paths_noisy:
        raise FileNotFoundError(f"No TIFF files were found in {dir_noisy}.")

    for path in paths_noisy:
        print(path.parent.stem)

    for file_index, path in enumerate(paths_noisy):
        print(f"Loading file ({file_index + 1}/{len(paths_noisy)}): {path.stem}")
        for power in power_percent:
            exp_name = f"{path.stem}_{power:03d}"
            out_dir = f"./outputs/{path.stem}"
            simulated_path = Path(out_dir) / exp_name / f"{exp_name}.tif"

            print(f"\tSimulating {power}% power for {path.stem}")
            with tifffile.TiffFile(path) as tif:
                stack = tif.asarray()
            stack = simulate_lower_power(
                stack,
                power / 100,
                gamma=2.0,
                photons_per_unit=1 / 100,
            )

            with Path("parameters.json").open(encoding="utf-8") as file:
                params = json.load(file)
            train_params = params[dimensionality]["train"]
            test_params = params[dimensionality]["test"]

            train_params["dimensionality"] = dimensionality
            train_params["rm_dir"] = rm_dir
            train_params["out_dir"] = out_dir
            train_params["n_cpu"] = n_cpu
            train_params["cuda_device"] = gpu
            train_params["in_path"] = simulated_path
            train_params["train_gen"] = False
            train_params["exp_name"] = exp_name
            train_params["checkpoints"] = checkpoints
            train_params["resume"] = resume
            train_params["rm_intermediaries"] = rm_intermediaries

            train_cd = TrainingClass(dict(train_params))
            train_cd.vars_dict()
            train_cd.log.log_params()

            print(f"\tSaving low-power TIFF to {simulated_path}...", end=" ")
            tifffile.imwrite(
                simulated_path,
                stack.astype(np.float32, copy=False),
                imagej=True,
                metadata={"axes": "TZYX"},
            )
            print("✅")
            del stack

            train_cd.run()
            torch.cuda.empty_cache()

            model_path = str(train_cd.model_path)
            test_in_path = train_params["in_path"]
            test_params["model_path"] = model_path.replace("/vf/users", "/data")
            test_params["out_dir"] = train_cd.out_dir.replace("/vf/users", "/data")
            test_params["exp_name"] = train_cd.exp_name
            del train_cd

            test_dict = {
                "in_path": test_in_path,
                "model_path": test_params["model_path"],
                "exp_name": test_params["exp_name"],
                "out_dir": test_params["out_dir"],
                "dimensionality": train_params["dimensionality"],
                "cuda_device": train_params["cuda_device"],
                "patch_size": train_params["patch_size"],
                "patch_interval": test_params["patch_interval"],
                "batch_size": test_params["batch_size"],
                "pin_memory": test_params.get("pin_memory", True),
                "depth": train_params["depth"],
                "bc_channels": train_params["bc_channels"],
                "obo_channels": train_params["obo_channels"],
                "ll_channels": train_params["ll_channels"],
                "unet_channels": train_params["unet_channels"],
                "bs_size": train_params["bs_size"],
                "n_cpu": train_params["n_cpu"],
                "prefetch_factor": test_params["prefetch_factor"],
                "pad": test_params["pad"],
            }
            test_cd = TestingClass(test_dict)

            result = test_cd.run()
            torch.cuda.empty_cache()
            del result, test_cd

            if rm_tmp:
                os.remove(simulated_path)


if __name__ == "__main__":
    main()
