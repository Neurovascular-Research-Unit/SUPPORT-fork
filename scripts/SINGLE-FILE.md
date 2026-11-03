# Run `biowulf_singlefile` on Biowulf

This workflow trains and denoises with a separate SUPPORT model for each TIFF
or each channel of a split Prairie View acquisition. It defaults to `2+1D`
(`TYX`) data and reads settings from
[`notebooks/parameters.json`](../notebooks/parameters.json).

## 1. Set up the repository and environment

These commands assume Conda is installed at `/data/$USER/conda`. If yours is
elsewhere, use its initialization path here and in the submission override below.

```bash
ssh NIH_USERNAME@biowulf.nih.gov
cd /data/$USER
git clone https://github.com/Neurovascular-Research-Unit/SUPPORT.git
cd SUPPORT

source /data/$USER/conda/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate SUPPORT
```

If the repository and `SUPPORT` environment already exist, activate the
environment from the repository root. To refresh its dependencies, run
`conda env update -n SUPPORT -f environment.yml`.

## 2. Prepare and preview the inputs

Put source data under `/data/$USER/data_noisy` and keep the output directory
separate, for example `/data/$USER/data_clean`.

- Discovery is recursive. Each TIFF's full path must contain `2DT`
  (case-insensitive); use `--marker YOUR_TOKEN` to choose another marker.
- Ordinary stacks must use `.tif`. Each file gets its own model; stems must
  be unique across the input tree because outputs share one directory.
- Split OME files must follow names such as
  `scan_2DT-001_Cycle00001_Ch2_000001.ome.tif`. Within each source directory,
  chunks are grouped by acquisition and channel, sorted numerically by Cycle
  and chunk index, then concatenated in job-local scratch. Keep ordinary TIFFs
  and OME chunks in separate directories. `Reference`/`References` subtrees
  are ignored.

From `/data/$USER/SUPPORT`, preview grouping, shapes, order, output names, and
storage estimates without training or writing files:

```bash
python scripts/biowulf_singlefile.py \
  --input-directory /data/$USER/data_noisy \
  --output-directory /data/$USER/data_clean \
  --dry-run
```

The dry run can run on the login node without a GPU. Keep source files
unchanged while jobs are running and when resuming them.

## 3. Submit the job

Submit from the repository root:

```bash
sbatch scripts/biowulf_singlefile.sbatch \
  --input-directory /data/$USER/data_noisy \
  --output-directory /data/$USER/data_clean
```

Only `--input-directory` is required by Python. Passing `--output-directory`
explicitly keeps results in your chosen `/data` location instead of the
repository's default `notebooks/outputs`.

The command has three parts:

```text
sbatch [Slurm options] scripts/biowulf_singlefile.sbatch [SUPPORT options]
```

**Before the script path**, options tell Slurm what resources to allocate.
**After the script path**, options configure SUPPORT. For example,
`--cpus-per-task=6` reserves CPUs, while `--n-cpu 4` tells SUPPORT to use four
data-loading workers within that allocation. Options omitted from the command
use the defaults described below. Every SUPPORT option applies to every
discovered file/channel in that submission.

### Slurm resources and job settings

These are the submission settings used by
[`biowulf_singlefile.sbatch`](biowulf_singlefile.sbatch). Command-line values
override the corresponding `#SBATCH` lines; editing the wrapper is unnecessary.

| Option, before the script path | Wrapper default | Meaning |
| --- | --- | --- |
| `--partition=gpu` | `gpu` | Selects the GPU queue. |
| `--gres=gpu:a100:1,lscratch:200` | One A100 and 200 GB scratch | `gpu:a100:1` requests one A100; `lscratch:200` requests 200 GB of temporary local disk. |
| `--cpus-per-task=6` | `6` | Reserves six CPUs. SUPPORT also uses this number as its worker count unless `--n-cpu` overrides it. |
| `--mem=164g` | `164g` | Requests 164 GB of system RAM for the job. GPU memory is determined by the allocated GPU. |
| `--time=08:00:00` | Eight hours | Wall-clock limit for the entire submission, including all files/channels and inference. Format: `HH:MM:SS`. |
| `--job-name=support-singlefile` | `support-singlefile` | Label displayed in the job queue. |
| `--output=support-singlefile-%j.out` | `support-singlefile-%j.out` | Combined stdout/stderr log; `%j` becomes the job ID. Relative paths use the submission directory. |
| `--export=ALL` | Inherit the submitting environment | Passes exported environment variables into the job, including the path overrides below. |

See the official [Biowulf submission guide](https://hpc.nih.gov/docs/userguide.html)
and [Slurm `sbatch` reference](https://slurm.schedmd.com/sbatch.html) for scheduler
details.

Use one combined `--gres` option when changing scratch capacity, for example
`--gres=gpu:a100:1,lscratch:300`. This replaces the embedded resource request,
so include both the GPU and scratch. The runner uses one logical GPU (`0`)
and processes files/channels sequentially; requesting additional GPUs does not
parallelize this runner.

Scratch holds one concatenated OME channel at a time. Size it for the largest
logical OME input reported by the dry run, with headroom. The wrapper creates
`/lscratch/$SLURM_JOB_ID/support-singlefile` and uses it for staging and temporary
files. Biowulf deletes job-local scratch when the job ends; keep checkpoints and
outputs on `/data`. See [Biowulf local disk guidance](https://hpc.nih.gov/docs/userguide.html).

### Repository and Conda overrides

These are environment variables, rather than Python arguments. Leave them
unset when using the default layout.

| Variable | Default | When to set it |
| --- | --- | --- |
| `SUPPORT_REPO_DIR` | Directory where you ran `sbatch` (`SLURM_SUBMIT_DIR`) | Your repository is elsewhere. The wrapper changes into this directory before running Python. |
| `SUPPORT_CONDA_SH` | `/data/$USER/conda/etc/profile.d/conda.sh` | Your Conda installation has a different initialization path. Point to `conda.sh`, not the `conda` executable. |
| `SUPPORT_CONDA_ENV` | `SUPPORT` | Your environment has a different name, or you activate it by its full path. |

For example, replace `/path/to/conda` with your installation and export the
values before submitting. Use the absolute script path if submitting from
outside the repository:

```bash
export SUPPORT_REPO_DIR=/data/$USER/SUPPORT
export SUPPORT_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export SUPPORT_CONDA_ENV=SUPPORT

sbatch --export=ALL "$SUPPORT_REPO_DIR/scripts/biowulf_singlefile.sbatch" \
  --input-directory /data/$USER/data_noisy \
  --output-directory /data/$USER/data_clean
```

### SUPPORT paths and input selection

All options in this and the following tables go **after** the script path.
`PATH`, `N`, and other uppercase labels stand for values you supply. Flags
such as `--dry-run` take no value. For paired flags, choose one, for example
`--no-use-amp`; do not append `true` or `false`.

Use absolute paths for inputs and outputs. Relative SUPPORT paths resolve from
the repository root because the wrapper changes into it before running Python.

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-directory PATH` | Required | Root directory searched recursively for eligible input TIFFs. Alias: `--in-path`. |
| `--output-directory PATH` | `notebooks/outputs` under the repository | Flat directory for final float32 `_SPRT.tif` files. Must not overlap the input tree. Alias: `--out-dir`. |
| `--work-directory PATH` | `OUTPUT_DIRECTORY/.support_singlefile_models` | Persistent root for per-file/channel models, checkpoints, manifests, and training logs. Keep it outside the input tree. Use the same location to resume. |
| `--scratch-directory PATH` | Under the batch wrapper: `/lscratch/$SLURM_JOB_ID/support-singlefile` | Temporary staging root for concatenated OME channels. Normally leave this to the wrapper. A path override does not request disk capacity; `--gres` does. Must be outside the input tree. |
| `--marker TEXT` | `2DT` | Case-insensitive text required anywhere in each TIFF's full path. This filters discovery; it does not change the data mode. Cannot be empty. |
| `--dimensionality MODE` | `2+1D` | `2+1D` for `TYX` time series, `3D` for `ZYX` static volumes, or `3+1D` for `TZYX` time series. Legacy `TYX` also works in `3+1D`. Split OME acquisitions require `2+1D`. |
| `--parameters PATH` | `notebooks/parameters.json` | JSON containing `train` and `test` dictionaries for the selected mode. Read without modifying it; explicit command-line settings override its values. |

For direct Python runs, scratch falls back in this order: `--scratch-directory`,
`SUPPORT_SINGLEFILE_SCRATCH_DIR`, `TMPDIR`, then the system temporary directory.
The batch wrapper sets both environment variables to its job-local directory.

### Training settings

**The defaults below describe the checked-in `2+1D` configuration.** Selecting
another mode or parameter file can change them. An epoch is one pass through
the training DataLoader; a batch is one group of patches processed together.

| Option | Current `2+1D` default | Meaning and effect |
| --- | --- | --- |
| `--n-epochs N` | `10` | Final total epochs for each model. If 10 epochs are complete, `--n-epochs 20` trains 10 more. A target below the saved completed count is refused. |
| `--train-batch-size N` | `512` | Training patches processed together. Smaller values reduce GPU memory demand and change training behavior. Alias: `--batch-size`. |
| `--train-patch-size T Y X` | `91 128 128` | Size of each training patch: 91 frames by 128 by 128 pixels by default. The first value is depth in `3D` mode. Must fit the input stack; inference uses the same patch size. Alias: `--patch-size`. |
| `--train-patch-interval T Y X` | `1 64 64` | Spacing used to determine the training patch grid and patch count. Smaller intervals increase patches per epoch and runtime; training samples random patch positions. Alias: `--patch-interval`. |
| `--learning-rate VALUE` | `5e-5` (`0.00005`) | Optimizer step size; must be positive. Changes how weights are learned. |
| `--random-seed N` | `0` | Nonnegative seed for model/training randomness. It does not guarantee identical results on every machine. |
| `--load-to-memory` / `--no-load-to-memory` | Load to memory | Controls training input loading. Loading creates normalized float32 data in RAM. Disabling attempts TIFF memory mapping and normalizes patches as needed; incompatible TIFF layouts can still fall back to a full in-memory read. It does not make inference memory-mapped. |
| `--use-amp` / `--no-use-amp` | AMP enabled | Controls automatic mixed precision during training. Enabling it can reduce GPU memory use and improve speed. The setting must match saved training state when resuming. |

All counts, patch dimensions, and intervals in this table must be positive
integers; the seed may be zero. Changing these training settings, other than
the epoch target, requires a fresh run directory; see the resume rules below.

### Data loading and host memory

Workers prepare patches for the model. They do not train additional models or
reserve extra Slurm resources.

| Option | Current `2+1D` default | Meaning and effect |
| --- | --- | --- |
| `--n-cpu N` | `SLURM_CPUS_PER_TASK` (`6` in the wrapper); `16` outside Slurm | DataLoader worker count for training and inference. May be zero to load data in the main process. Keep within the CPUs you requested; fewer workers can reduce host memory use. |
| `--train-prefetch-factor N` | `2` | Training batches prepared ahead per worker. With six workers and a factor of two, roughly 12 batches can be in flight. Lower values can reduce host memory pressure. |
| `--test-prefetch-factor N` | `1` | Same setting for inference. Both prefetch factors must be positive and only apply when workers are enabled. |
| `--train-pin-memory` / `--no-train-pin-memory` | Pinning enabled | Controls pinned host RAM for training transfers to the GPU. Pinning can improve transfers but increases pressure on host RAM. |
| `--test-pin-memory` / `--no-test-pin-memory` | Pinning enabled | Same choice for inference. Pinning has no effect when running on CPU. |

### Inference settings

Inference runs after each file/channel's model finishes training and writes
that file/channel's denoised TIFF.

| Option | Current `2+1D` default | Meaning and effect |
| --- | --- | --- |
| `--test-batch-size N` | `256` | Patches denoised together; must be positive. Lower this if inference exhausts GPU memory. It can change without retraining. |
| `--test-patch-interval T Y X` | `1 24 24` | Positive strides for inference patches. Smaller spatial strides increase overlap and runtime. The first stride **must be `1`** so every temporal/depth plane is covered. |
| `--test-pad` / `--no-test-pad` | Padding enabled | Reflect-pad the first axis before inference, then remove the padding. In `2+1D`, this provides temporal context for the first and last frames. Disabling changes boundary handling. |

### Resume, checkpoint retention, and execution

| Option | Default | Meaning |
| --- | --- | --- |
| `--resume` / `--no-resume` | Resume enabled | Continue from the latest complete checkpoint. `--no-resume` starts from scratch only in an experiment without existing checkpoints; it refuses to overwrite saved checkpoints. Use a new output/work directory for a fresh run. |
| `--keep-only-latest` / `--keep-all-checkpoints` | Keep only latest | Keep the newest completed epoch's checkpoint bundle, or retain all completed epoch bundles. This controls checkpoint disk use, not final TIFF retention. |
| `--rerun-completed` | Off | Repeat inference even if the saved completion record, model, and output verify. It reuses compatible trained weights; it does not restart training. Raising `--n-epochs` still extends training first. |
| `--use-cpu` | Off; use GPU | Run training and inference on CPU. This flag does not change the wrapper's Slurm GPU request. |
| `--dry-run` | Off | Read TIFF headers, validate input grouping and patch sizes, and print the plan without training, writing files, or sending Slack messages. Run directly with Python on the login node to avoid allocating a GPU. It does not measure runtime or GPU/RAM use. |
| `--help` / `-h` | Off | Print the argument list and exit. Use `python scripts/biowulf_singlefile.py --help`. |

Resume compatibility is checked per file/channel:

- You can change Slurm resources, worker counts, prefetching, memory pinning,
  checkpoint retention, and the inference batch size while reusing checkpoints.
  Increasing the epoch target continues training.
- Changing inference patch intervals or padding reuses training weights and
  regenerates the TIFF. Changing only inference batch size does not force a
  completed TIFF to be regenerated; add `--rerun-completed` if needed.
- Changed source files, dimensionality, training batch/patch settings, learning
  rate, random seed, AMP, loading/normalization path, or model architecture
  make saved training state incompatible. Choose a new output directory and,
  if you set `--work-directory`, a new work directory too.

### Optional Slack notifications

Notifications are disabled by default.

| Option | Meaning |
| --- | --- |
| `--slack-webhook-file PATH` | Read a Slack Incoming Webhook URL from a private mode-0600 file. This option enables notifications by itself. |
| `--slack-notifications` | Enable notifications using `SUPPORT_SLACK_WEBHOOK_FILE` from the exported environment. `SUPPORT_SLACK_WEBHOOK_URL` is a fallback. |

Messages cover model start, epoch progress/loss, completion, and ordinary Python
failures. Store the webhook URL in the private file rather than in the command.
See [Slack setup](README.md#optional-slack-notifications).

### Example with explicit resources and training options

For a new run, this requests the default compute resources with 300 GB scratch,
trains each model to 20 total epochs, and uses smaller batches than the current
JSON defaults:

```bash
sbatch \
  --partition=gpu \
  --gres=gpu:a100:1,lscratch:300 \
  --cpus-per-task=6 \
  --mem=164g \
  --time=08:00:00 \
  scripts/biowulf_singlefile.sbatch \
  --input-directory /data/$USER/data_noisy \
  --output-directory /data/$USER/data_clean_20epochs \
  --n-epochs 20 \
  --train-batch-size 64 \
  --test-batch-size 128
```

These batch sizes are examples, not measured memory requirements. Preview the
same SUPPORT options with `python scripts/biowulf_singlefile.py ... --dry-run`
before submitting; omit the Slurm options from the Python command.

## 4. Monitor, find outputs, and resume

Replace `JOB_ID` with the number returned by `sbatch`:

```bash
squeue -u "$USER"
tail -f support-singlefile-JOB_ID.out
```

- Denoised TIFFs: `/data/$USER/data_clean/<output-stem>_SPRT.tif`.
  Ordinary files retain their input stem; OME output stems include the relative
  source path, acquisition, and channel. Exact names appear in the dry run.
- Models, checkpoints, and training logs:
  `/data/$USER/data_clean/.support_singlefile_models/<output-stem>/`.

Resubmit the same command to resume from the latest completed epoch. Verified
completed outputs are skipped. `--n-epochs` specifies the final total;
increasing it continues training. `--rerun-completed` repeats inference for
completed jobs.

Busy files/channels and incompatible saved runs are skipped with an explanation
in the Slurm log. For an intentional fresh run with changed inputs or training
settings, choose a new output directory.

For GPU memory errors, reduce the batch size for the stage that failed.
For host memory pressure, reduce workers/prefetching or disable pinning.
Consider `--no-load-to-memory` when starting a new model; switching an existing
model's loading/normalization path makes its saved state incompatible.
