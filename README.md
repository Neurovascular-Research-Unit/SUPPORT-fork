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

`--n-epochs` is the final total, not the number of additional epochs. A job that
stops during an epoch resumes from the last completed epoch and repeats the
interrupted epoch. Use the same model parameters and AMP setting when resuming.
Run `python run.py --help` for runtime overrides including batch size, patch size,
worker count, prefetching, and memory mapping. For large directory datasets,
consider `--no-load-to-memory` and a modest `--prefetch-factor`.

### Biowulf batch example

An editable template is provided at `scripts/biowulf_train.sbatch`. After
adjusting its repository, input, output, environment, GPU, memory, and time
settings, submit it with:

```bash
sbatch scripts/biowulf_train.sbatch
```

To resume the template after `model_36.pth`, submit it with
`sbatch --export=ALL,START_EPOCH=37 scripts/biowulf_train.sbatch`.

Slurm sets `CUDA_VISIBLE_DEVICES`; `run.py` intentionally leaves that allocation
unchanged. It uses `SLURM_CPUS_PER_TASK` as the default DataLoader worker count.

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

5. **Configure the parameters**:

   In the second cell of the notebook, modify these parameters according to your data:

   ```python
   in_path = "/path/to/your/noisy_data"  # Path to your input data
   exp_name = None  # Optional: name for your experiment
   dimensionality = "3+1D"  # Options: "2+1D", "3D", or "3+1D"
   rm_dir = True  # Store training artifacts under notebooks/tmp
   checkpoints = True  # Save a resumable checkpoint after every epoch
   resume = True  # Automatically continue from the latest complete checkpoint
   rm_intermediaries = True  # Keep only the latest checkpoint bundle
   ```

   A checkpoint bundle contains the model, optimizer, and AMP scaler when AMP is
   enabled. If `resume` is enabled but no complete bundle exists, training starts
   from epoch 0. When `checkpoints` is disabled, one final bundle is still saved
   for testing and possible later continuation.

   In the third code cell (fourth cell overall), you can adjust system-specific parameters:

   ```python
   out_dir = None  # Optional: output directory (defaults to current directory)
   n_cpu = 8  # Number of CPU cores to use
   gpu = '0'  # GPU device ID to use
   ```

6. **Adjust training/testing parameters** (optional):

   The notebook loads parameters from `parameters.json`. You can modify this file to adjust:
   - Number of epochs (`n_epochs`)
   - Patch size and intervals
   - Batch size
   - Model architecture parameters (depth, channels, etc.)
   - Learning rate and loss coefficients

7. **Run the notebook cells**:

   Execute the cells in order:
   - **Initialization**: Imports libraries and sets up parameters
   - **Train**: Runs the training process
   - **Test**: Applies the trained model to denoise/enhance your data

   You can run cells individually by selecting them and pressing `Shift+Enter`, or run all cells using `Cell > Run All` from the menu.

8. **Monitor training progress**:
   - Training progress will be displayed with a progress bar
   - Loss values are plotted in ASCII format during training
   - Models are saved as checkpoints during training

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
- Close other applications using GPU memory

### Import Errors

If you get import errors:

- Make sure the SUPPORT conda environment is activated
- Verify all dependencies are installed: `conda list`
- Reinstall the environment: `conda env remove -n SUPPORT && conda env create -f environment.yml`

## Additional Resources

- Adjust `parameters.json` for different dimensionality modes (2+1D, 3D, or 3+1D)
- For custom training parameters, refer to the `TrainingClass` and `TestingClass` documentation in the source code
- The model architecture is defined in `support/SUPPORT.py`
