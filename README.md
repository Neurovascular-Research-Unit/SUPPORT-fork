# SUPPORT

Self-supervised imaging denoising and enhancement tailored for neurovascular research workflows. Based on NICALab's original SUPPORT repository

## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- NVIDIA GPU with CUDA support (recommended for training and inference)
  - CUDA 11.8 or higher is recommended
  - Check your CUDA version with `nvidia-smi`
- At least 8GB of RAM (more recommended for larger datasets)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Neurovascular-Research-Unit/SUPPORT.git
cd SUPPORT
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
```

This will create a new conda environment named `SUPPORT` with all required dependencies including PyTorch, NumPy, and Jupyter.

### 3. Activate the environment

```bash
conda activate SUPPORT
```

### 4. Register the kernel with Jupyter (if needed)

```bash
python -m ipykernel install --user --name SUPPORT --display-name "SUPPORT"
```

## Training One Model from a Directory

The root-level `run.py` trains one generalized model from every top-level `.tif`
or `.tiff` stack in a single directory. It is train-only, has no hard-coded
filesystem paths, and always preserves its checkpoints.

```bash
python -u run.py \
  --in-path /path/to/training_stacks \
  --out-dir /path/to/model_outputs \
  --exp-name generalized \
  --dimensionality "2+1D" \
  --n-epochs 100
```

The model, Adam optimizer, and (when AMP is enabled) AMP scaler are saved after
every completed epoch in:

```text
/path/to/model_outputs/generalized/pth/
├── model_0.pth
├── optimizer_0.pth
├── scaler_0.pth
└── ...
```

Checkpoint numbering and continuation match the original NICALab trainer. Epochs
are zero-based. To continue after epoch 36 completed, pass `--epoch 37`; this
loads `model_36.pth`, `optimizer_36.pth`, and `scaler_36.pth`, then continues at
epoch 37:

```bash
python -u run.py \
  --in-path /path/to/training_stacks \
  --out-dir /path/to/model_outputs \
  --exp-name generalized \
  --dimensionality "2+1D" \
  --n-epochs 100 \
  --epoch 37
```

`--n-epochs` is the final total, not the number of additional epochs. A
generalized job that stops during an epoch restores its newest rolling batch
checkpoint and repeats the interrupted epoch; normal training restores the last
completed epoch. Use the same model parameters and AMP setting when resuming.
Run `python run.py --help` for runtime overrides including batch size, patch size,
worker count, prefetching, and memory mapping. For large directory datasets,
consider `--no-load-to-memory` and a modest `--prefetch-factor`.

### Biowulf batch example

`scripts/biowulf_train.sbatch` is a command-line version of
`notebooks/train_gen.ipynb`: it recursively trains one generalized model,
writes one rolling batch checkpoint at four quartile points per epoch, saves
and retains each permanent epoch checkpoint before validation, and
automatically resumes from the latest complete bundle. Submit it from the
repository root with separate training and held-out validation paths:

```bash
sbatch scripts/biowulf_train.sbatch \
  --training-directory /data/$USER/training_tiffs \
  --test-directory /data/$USER/validation_tiffs \
  --dimensionality '3+1D' \
  --exp-name calcium_4D
```

The embedded Slurm defaults request one A100, 16 CPUs, 240 GB, 24 hours, and
200 GB of ephemeral node-local scratch for offloading normalized training data
during validation. To change scratch capacity, keep the complete Slurm override
before the script path, for example
`sbatch --gres=gpu:a100:1,lscratch:300 scripts/biowulf_train.sbatch ...`.
Resubmit the same command to resume; `--n-epochs` specifies the final total and
the job rebuilds its temporary scratch cache.
Optional Slack start, per-epoch loss, completion, and failure updates can be
enabled with `--slack-webhook-file /path/to/private-mode-0600-file`. Keep the
webhook URL itself out of commands and notebooks. The generalized training
notebook has an equivalent `slack_notifications = False` setting.
See `scripts/README.md` for dry runs, output layout, validation metrics,
Slack setup, environment overrides, checkpoint promotion, and recovery
guidance.

Slurm sets `CUDA_VISIBLE_DEVICES`; the batch runner leaves that allocation
unchanged and uses `SLURM_CPUS_PER_TASK` as the DataLoader worker count.

## Staged Multi-GPU SNR Sweeps

`notebooks/snr_test.py` first trains and runs SUPPORT once per raw input TIFF to
create a persistent denoised reference named `<stem>_original_SPRT.tif`. After
all successful references are available, it simulates every configured power
level from that denoised reference—not from the raw acquisition—and gives each
simulated stack its own SUPPORT model. The recursively scanned input directory
may contain any positive number of `.tif` or `.tiff` files; every discovered
TIFF becomes an independent source. Input stems must be unique across the tree
because each stem defines its output namespace. Configure `dir_noisy`,
`dimensionality`, `power_percent`, and the other sweep settings near the top of
the script, or override the paths and sweep settings with command-line options,
then run:

```bash
python -u notebooks/snr_test.py
```

On Biowulf, submit the matching batch entrypoint with shared-storage paths:

```bash
sbatch scripts/biowulf_snr_test.sbatch \
  --input-directory /data/$USER/snr-input \
  --output-directory /data/$USER/snr-output \
  --dimensionality '3+1D' \
  --power-percent 100 90 80 70 60 50 40 \
  --slack-webhook-file /data/$USER/.secrets/support-slack-webhook
```

The batch job requests one A100, 16 CPUs, 240 GB RAM, 200 GB local scratch, and
24 hours by default. Slurm resource overrides go before the script path; SNR
runner options go after it. See `scripts/README.md` for multi-GPU sizing and
scratch guidance.

Pass `--no-train-pin-memory` and/or `--no-test-pin-memory` to disable pinned
host memory independently for the training and inference DataLoaders.
The optional private mode-0600 `--slack-webhook-file` enables start, per-epoch,
completion, and failure messages for every independent original/power model.

With `photons_per_unit = None` (the default), the stage barrier independently
recomputes a photon-transfer calibration for every successful discovered
raw/original-SUPPORT pair on every invocation.
It measures raw adjacent-frame variance using the denoised SUPPORT intensity as
the cleaner horizontal axis, converts the result into SUPPORT image units, and
reuses that value for every power level from the same source. It also compares
the primary result with the quietest half of frame-pair locations; disagreement
warns about motion or biological changes being counted as shot noise.
By default, both the variance and raw-to-SUPPORT scale fits exclude the first
60 and final 60 frames. Up to 50 adjacent, non-overlapping pairs are distributed
across that central window rather than taken only from its beginning. Automatic
calibration requires repeated-time data. The full SUPPORT workflow expects
stored `TYX` or `TZYX`; the calibration reader additionally honors a
metadata-declared `T` axis. Static 3D data require a positive fixed override.
Set `photons_per_unit` in the script/notebook or pass,
for example, `--photons-per-unit 0.00075` to bypass estimation. By default the
sweep stops a source when the linearity diagnostics are unreliable; the
`--allow-provisional-photon-estimate` flag explicitly relaxes that guard.

All photon-generating intensity—including out-of-focus fluorescence—is scaled
with laser power. Only a laser-independent additive detector/digitizer baseline
should stay fixed. Low electronic-noise variance does not necessarily mean that
this baseline offset is numerically zero. Set it in SUPPORT units with
`photon_zero_level` or `--photon-zero-level`; zero is the default, and a dark
or laser-off acquisition is the preferred calibration. The photon-transfer fit
also reports an
extrapolated effective pedestal. Set `use_estimated_photon_pedestal = True` or
pass `--use-estimated-photon-pedestal` to use that model-dependent candidate
explicitly. This opt-in requires automatic photon calibration and cannot be
combined with a nonzero fixed pedestal. A dark image region is not used as the
pedestal because it may contain optical background that should scale.
The simulator draws independent Poisson samples, so it matches the calibrated
marginal mean/variance but does not reproduce scanner-correlated noise unless a
separate, empirically calibrated noise model is added.

By default, `gpu_ids = None` uses every GPU visible to the process. Select a
subset with `--gpus`; these are logical IDs within the inherited
`CUDA_VISIBLE_DEVICES` allocation (including a Slurm allocation):

```bash
python -u notebooks/snr_test.py --gpus 0 2
```

Use `--gpus 0` for the single-GPU fallback. Each selected GPU receives its own
worker process and runs complete jobs serially; this is job-level parallelism,
not `DataParallel` or distributed model training. The original-image stage
finishes before any dependent power job starts. A failed original skips only
that source's power jobs; other sources continue. Training batch size and the
other model hyperparameters therefore retain their single-GPU meaning.

`power_percent` denotes measured excitation power at the specimen. The
simulator uses the two-photon relationship `intensity_ratio = power_ratio**2`,
so 50% power produces 25% expected photon-generating intensity. The 100%
condition is a newly sampled full-power acquisition from the denoised
reference, so it follows the same simulation and second SUPPORT pass as every
lower-power condition. New power experiments use names such as
`<stem>_original_050`, which prevents `resume = True` from loading checkpoints
created by the older raw-derived workflow. `rm_tmp = True` removes successful
simulated acquisition TIFFs but always retains the denoised original reference.
As elsewhere in SUPPORT, checkpoint resume assumes the source TIFF, parameter
file, and simulation settings have not changed. Use a new output namespace or
disable/remove the prior checkpoints when changing that provenance.

Resource settings are per GPU. In particular, `n_cpu = 8` can create up to
eight DataLoader workers for each GPU worker, and each concurrent job needs its
own host RAM, GPU memory, and input I/O bandwidth. Allocate CPUs and memory for
the selected GPU count, or select fewer GPUs when those resources are limited.
Run only one SNR sweep process against a given output tree at a time; that one
process schedules all selected GPUs itself.

## Running the Notebook

### Step-by-Step Instructions

1. **Activate the conda environment** (if not already activated):

   ```bash
   conda activate SUPPORT
   ```

2. **Navigate to the notebooks directory**:

   ```bash
   cd notebooks
   ```

3. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

   This will open Jupyter in your default web browser.

4. **Open the run.ipynb notebook**:
   - In the Jupyter interface, click on `run.ipynb` to open it.
   - For generalized training with held-out validation after every epoch, open
     `train_gen.ipynb` instead. It recursively reads validation TIFFs from
     `test_dir`, retains every epoch checkpoint, and keeps paired training and
     validation loss CSVs.

5. **Configure the parameters**:

   In the notebook's configuration cell, modify these parameters according to
   your data:

   ```python
   in_path = "/path/to/your/noisy_data.tif"
   train_gen = False  # Options: True or False
   test_dir = None  # Used only when train_gen is True
   exp_name = str(Path(in_path).stem)
   dimensionality = "3+1D"  # Options: "2+1D", "3D", or "3+1D"
   rm_dir = True  # Store training artifacts under notebooks/tmp
   checkpoints = True  # Enable resumable checkpoints
   resume = True  # Automatically continue from the latest complete checkpoint
   rm_intermediaries = True  # Keep only the latest checkpoint bundle
   ```

   With the default `train_gen = False`, `in_path` is the training input and is
   also denoised by the **Test** cells. It can be a TIFF stack or a directory of
   top-level `.tif` or `.tiff` stacks (with case-insensitive extensions). This
   normal mode loads `parameters.json` and, when
   `checkpoints` is enabled, saves a resumable checkpoint after every epoch.

   To train one model for generalization across a directory of data, use:

   ```python
   in_path = "/path/to/training_data"
   train_gen = True
   test_dir = "/path/to/data_to_denoise"  # A TIFF or directory of top-level TIFF stacks
   exp_name = str(Path(in_path).stem)
   ```

   Generalized training searches `in_path` recursively for both `.tif` and
   `.tiff` files, including files in nested subdirectories. Only `test_dir` is
   passed to the **Test** cells for denoising; the training directory is not
   denoised automatically. Leave `test_dir = None` for a train-only run. This
   mode loads `gen_parameters.json`. Its defaults memory-map TIFF data and leave
   DataLoader page-locking disabled so a multi-file corpus does not require a
   second large in-memory representation or pinned batch copy. Generalized
   testing names a single-file result from the TIFF stem and a legacy 3+1D
   directory result from the test directory name. Before training, generalized
   mode computes one voxel-weighted mean and sample standard deviation across
   every discovered training TIFF and applies that shared z-score to all
   training patches. This keeps all batch losses on one normalized scale.
   Normal mode uses one mean and sample standard deviation per source TIFF; for
   a 4-D TIFF, that single pair spans all `T`, `Z`, `Y`, and `X` voxels and is
   shared by every extracted Z plane.

   A checkpoint bundle contains the model, optimizer, and AMP scaler when AMP is
   enabled. In normal mode, `rm_intermediaries` controls whether older epoch
   bundles are retained. In generalized mode with `checkpoints = True`, one
   rolling bundle is saved at each of the four batch-quartile endpoints in every
   epoch. The new batch bundle is saved completely before the preceding rolling
   bundle is deleted; permanent epoch bundles are not touched. With
   `save_every_epoch = True`, a separate permanent bundle is saved at the epoch
   boundary before validation and retained by default. If `resume` is enabled
   but no complete bundle exists, training starts from epoch 0. When
   `checkpoints` is disabled, one final bundle is still saved for testing and
   possible later continuation. Resume generalized training with the same input
   data and batch settings; a mid-epoch checkpoint preserves the newest weights
   but safely repeats that epoch, while a final-quartile checkpoint continues
   with the next epoch.

   In the following personalization cell, you can adjust system-specific
   parameters:

   ```python
   out_dir = None  # Optional: output directory (defaults to current directory)
   n_cpu = 8  # Number of CPU cores to use
   gpu = '0'  # GPU device ID to use
   ```

6. **Adjust training/testing parameters** (optional):

   Normal mode loads `parameters.json`; generalized mode loads
   `gen_parameters.json`. Modify the applicable file to adjust:

   - Number of epochs (`n_epochs`)
   - Patch size and intervals
   - Batch size
   - Model architecture parameters (depth, channels, etc.)
   - Learning rate and loss coefficients

7. **Run the notebook cells**:

   Execute the cells in order:
   - **Initialization**: Imports libraries and sets up parameters
   - **Train**: Runs the training process
   - **Test**: Applies the trained model to `in_path` in normal mode or to
     `test_dir` in generalized mode; generalized training skips these cells when
     `test_dir` is `None`

   You can run cells individually by selecting them and pressing `Shift+Enter`, or run all cells using `Cell > Run All` from the menu.

8. **Monitor training progress**:

   - Training progress is displayed with a progress bar using the running
     NaN-ignoring mean losses.
   - Loss values are plotted in ASCII format during training.
   - During generalized training, every completed batch appends the running
     ordinary means `L1_mean`, `L2_mean`, and `L_mean` and the parallel
     NaN-ignoring means `L1_nanmean`, `L2_nanmean`, and `L_nanmean` to
     `<exp_name>_train.log`. `L_mean` and `L_nanmean` are the combined training
     losses. Epoch JSON and CSV records likewise retain the existing ordinary
     mean fields and add matching `*_nanmean` fields. The same log
     records the shared training mean, standard deviation, and voxel count.
   - Normal-mode checkpoints are written at epoch boundaries. Generalized-mode
     rolling checkpoints are written at the four batch-quartile endpoints; each
     new complete batch bundle replaces only the preceding rolling bundle, while
     permanent epoch checkpoints remain retained.

9. **View results**:
   - After testing, denoised/enhanced TIFFs are saved with `_SPRT.tif` appended to their stem
   - The output path is determined by `out_dir` and `exp_name` parameters

## Troubleshooting

### GPU Issues

If you encounter CUDA errors:

- Ensure your NVIDIA drivers are up to date
- Check that CUDA is properly installed: `nvidia-smi`
- Verify PyTorch can access the GPU:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

### Memory Issues

If you run out of memory:

- Reduce the `batch_size` in the parameters
- Reduce the `patch_size` in the parameters
- Set `load_to_memory` to `false` for a multi-file training corpus
- Set `pin_memory` to `false` if page-locked DataLoader batches fail
- Close other applications using GPU memory

### Import Errors

If you get import errors:

- Make sure the SUPPORT conda environment is activated
- Verify all dependencies are installed: `conda list`
- Reinstall the environment: `conda env remove -n SUPPORT && conda env create -f environment.yml`

## Additional Resources

- Adjust `parameters.json` (normal mode) or `gen_parameters.json` (generalized
  mode) for different dimensionality modes (2+1D, 3D, or 3+1D)
- For custom training parameters, refer to the `TrainingClass` and `TestingClass` documentation in the source code
- The model architecture is defined in `support/SUPPORT.py`
