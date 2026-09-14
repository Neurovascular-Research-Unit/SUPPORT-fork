# Biowulf SUPPORT training and denoising

The batch entrypoints use the same environment overrides and request these
resources by default:

```text
1 A100 GPU, 16 CPUs, 240 GB memory, 24 hours
```

The generalized and per-file training jobs additionally request 200 GB of
job-local `/lscratch`. Generalized training uses it for an ephemeral normalized
training-data cache; per-file training uses it to concatenate split OME-TIFFs
into one logical channel stack. Scratch data are recreated for every job and
automatically deleted when that job exits; checkpoints, logs, CSVs, outputs,
and source TIFFs remain on shared `/data`. With `--no-load-to-memory`, the
generalized trainer attempts to memory-map TIFFs and skips its normalized-tensor
cache; TIFF layouts that cannot be memory-mapped may still require an in-memory
fallback.

Slurm resource overrides belong before the script path; SUPPORT runner options
belong after it. All three batch entrypoints expect the repository in
`SLURM_SUBMIT_DIR` (normally submit from `/data/$USER/SUPPORT`), the conda
initialization script at
`/data/$USER/conda/etc/profile.d/conda.sh`, and an environment named `SUPPORT`.
Set `SUPPORT_REPO_DIR`, `SUPPORT_CONDA_SH`, or `SUPPORT_CONDA_ENV` through
`sbatch --export` when those defaults differ.

## Per-file and per-channel 2DT training

`biowulf_singlefile.sbatch` recursively finds TIFFs whose full path contains
`2DT` (case-insensitive), then trains and tests a separate, non-generalized
model for each logical input. It reads normal-mode settings from
`notebooks/parameters.json` and supports two strict layouts:

- A directory containing exactly two ordinary `.tif` files. Each file gets its
  own model and the legacy `<input-stem>_SPRT.tif` output name.
- Prairie View acquisitions named like
  `..._Cycle00001_Ch2_000001.ome.tif`. Chunks are sorted numerically by Cycle
  and trailing index, concatenated as one temporal stack in `/lscratch`, and
  trained/tested independently per channel. A `References` subtree is ignored.

For split OME acquisitions, the flat output stem contains the source-relative
parent path, normalized acquisition name, and channel. Given a `data_noisy`
root, files under
`C57_293_ChronicCrani/2026_0805_BaselineImaging1/FOV2_250Deep_3X_2D` with the
acquisition prefix `FOV2_250Deep_3X_2DT-001` produce:

```text
C57_293_ChronicCrani_2026_0805_BaselineImaging1_FOV2_250Deep_3X_2D_FOV2_250Deep_3X_2DT_Ch2_SPRT.tif
C57_293_ChronicCrani_2026_0805_BaselineImaging1_FOV2_250Deep_3X_2D_FOV2_250Deep_3X_2DT_Ch3_SPRT.tif
```

Preview all grouping, temporal order, shapes, and destinations on a login node:

```bash
python scripts/biowulf_singlefile.py \
  --input-directory /data/$USER/data_noisy \
  --output-directory /data/$USER/data_clean \
  --dry-run
```

Then submit the same inputs through Slurm:

```bash
sbatch scripts/biowulf_singlefile.sbatch \
  --input-directory /data/$USER/data_noisy \
  --output-directory /data/$USER/data_clean \
  --slack-webhook-file /data/$USER/.secrets/support-slack-webhook
```

OME concatenation uses private job-local scratch; the final TIFF still uses
SUPPORT's existing atomic float32 ImageJ `TYX` writer. Persistent checkpoints
and logs live under
`OUTPUT_DIRECTORY/.support_singlefile_models/<output-stem>/`, allowing a
resubmitted command to resume at the latest completed epoch. A saved manifest
prevents a checkpoint from being silently reused with changed inputs or model
settings. Increase the `lscratch` request if the dry-run's logical input size
exceeds the default allocation, and leave enough output quota for a temporary
float32 TIFF alongside any output being replaced. On resubmission, a logical
job whose completion record, model, and output all verify is skipped; use
`--rerun-completed` to regenerate it. Increasing `--n-epochs` continues each
model, while lowering it below an existing checkpoint is refused. If another
process currently holds a logical job's training lock, that item is reported in
the Slurm log and skipped without a Slack failure notification; the batch
continues with the remaining TIFFs/channels and exits successfully. Merely
finding an old unlocked lock file does not cause a skip.

The saved-run compatibility check ignores machine/runtime-only differences such
as filesystem mount-device values, memory pinning, and data prefetch counts.
Changing these does not invalidate a checkpoint. Model architecture, training
semantics (including the data-loading/normalization path), source
paths/shapes/dtypes/sizes/modification times/inodes, and temporal ordering must
still match. A logical job with genuinely incompatible saved checkpoints is
preserved and skipped without a Slack failure notification, and the batch
continues. The Slurm log lists the mismatched identity fields. To retrain it
intentionally, first ensure no job is using its output, then select a new work
and output directory or archive the old experiment state manually.

## Generalized training with validation

`biowulf_train.sbatch` is the non-notebook equivalent of
`notebooks/train_gen.ipynb`. It trains one generalized model from every TIFF
below a training directory, saves that epoch's complete checkpoint bundle, and
then validates the same model against every held-out TIFF.

Training and validation search recursively for case-insensitive `.tif` and
`.tiff` extensions. The training input must be a directory; the validation
input may be one TIFF or a directory. Choose the SUPPORT mode explicitly:

| Data | TIFF array | `--dimensionality` | Pretrained label |
| --- | --- | --- | --- |
| 2D time series | `TYX` | `2+1D` | `2D` |
| 3D volume | `ZYX` | `3D` | `3D` |
| 4D time series | `TZYX` | `3+1D` | `4D` |

Legacy `TYX` input is also accepted in `3+1D` mode. Because both `TYX` and
`ZYX` have three array dimensions, the runner never guesses the mode from TIFF
`ndim`.

The runner reads copied `train` and `test` dictionaries from
`notebooks/gen_parameters.json`; it never edits that file. Defaults match the
notebook: generalized normalization across the training corpus, validation
after every epoch, automatic resume, and retention of every completed epoch's
model/optimizer/(when AMP is enabled) scaler bundle. `--keep-only-latest` is an
explicit disk-saving opt-out from retaining all bundles; the preceding clean
epoch is still kept while the next epoch is in progress and is pruned only when
the new permanent epoch bundle commits.

During each epoch, the trainer also writes a complete rolling batch bundle at
the 25%, 50%, 75%, and 100% batch endpoints. Each new batch bundle is committed
before the preceding rolling bundle is removed, and permanent epoch bundles are
left intact. Thus only one batch bundle is retained at a time; the permanent
epoch save supersedes it immediately before validation. An epoch with fewer
than four batches has only that many distinct checkpoint points. A mid-epoch
resume loads the newest rolling weights and safely repeats that epoch because
DataLoader/sampler state is not checkpointed.

Validation loss is self-supervised against held-out noisy center pixels, not
clean ground truth. It uses the training corpus's shared normalization and
records both the aggregate loss and per-file losses. Each validation TIFF has a
stable source-relative output path, so a new epoch atomically replaces that
file instead of accumulating one denoised TIFF per epoch. The validation
temporal/depth patch interval must be `1`; the launcher rejects a larger value
because the stitcher emits one center plane per patch and otherwise would leave
planes uncovered.

Training and validation TIFFs are never deleted. The training and validation
trees must be disjoint, and neither input may overlap its output tree. These
guards prevent a resumed recursive scan from training on generated `_SPRT`
outputs or from validating its own earlier results. Review the dry-run listing:
all TIFFs below the training root—including TIFFs in metadata or old result
folders—are intentionally part of generalized training. TIFF symlinks and any
symlinks, hard-linked files, or special filesystem entries in the managed
experiment or validation output trees are refused, preventing ambiguous input
identity or a checkpoint, log, or denoised TIFF from aliasing another file.

Keep both input trees immutable for the entire job. The launcher freezes the
discovered path lists and rechecks identity metadata at launch, but it cannot
prevent another process from rewriting an already-open TIFF during training or
validation.

### Prepare and preview training

Stage the repository and both TIFF trees on shared `/data/$USER`, connect to
Biowulf, and make lightweight checks on the login node:

```bash
ssh NIH_USERNAME@biowulf.nih.gov
checkquota
cd /data/$USER/SUPPORT

find /data/$USER/training_tiffs -type f \
  \( -iname '*.tif' -o -iname '*.tiff' \) -print | LC_ALL=C sort
find /data/$USER/validation_tiffs -type f \
  \( -iname '*.tif' -o -iname '*.tiff' \) -print | LC_ALL=C sort
```

Then activate the environment and preflight every TIFF header, path guard,
resolved setting, and normalized float32 scratch-cache estimate without
creating an output directory:

```bash
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate SUPPORT
python scripts/biowulf_train.py \
  --training-directory /data/$USER/training_tiffs \
  --test-directory /data/$USER/validation_tiffs \
  --dimensionality '3+1D' \
  --exp-name calcium_4D \
  --dry-run
```

The default batch, prefetch, pin-memory, and `load_to_memory` values come from
`gen_parameters.json`. Sixteen workers can have roughly
`n_cpu × prefetch_factor` batches in flight, so 240 GB does not by itself make a
large configuration safe. For a large corpus, start conservatively by combining
`--no-load-to-memory` with a smaller `--train-batch-size` and
`--train-prefetch-factor`; reduce `--n-cpu` or add `--no-train-pin-memory` when
host or shared-memory pressure remains. A smaller training batch may also be
needed for A100 GPU memory. Run `python scripts/biowulf_train.py --help` for
epoch, batch, patch, prefetch, memory, AMP, pin-memory, validation, and loss
overrides.

### Submit and resume training

Submit from the repository root:

```bash
sbatch scripts/biowulf_train.sbatch \
  --training-directory /data/$USER/training_tiffs \
  --test-directory /data/$USER/validation_tiffs \
  --dimensionality '3+1D' \
  --exp-name calcium_4D
```

The default 200 GB local-scratch allocation must be large enough for the
normalized float32 training corpus plus the headroom reported by the dry run.
Request at least that displayed amount rather than relying on otherwise free
node disk. To change it, put the Slurm override before the script path and keep
the GPU and scratch requests together because a command-line `--gres` replaces
the embedded value:

```bash
sbatch --gres=gpu:a100:1,lscratch:300 scripts/biowulf_train.sbatch \
  --training-directory /data/$USER/training_tiffs \
  --test-directory /data/$USER/validation_tiffs \
  --dimensionality '3+1D' \
  --exp-name calcium_4D
```

Inside the allocation, both `TMPDIR` and `SUPPORT_TRAINING_CACHE_DIR` point to
the private `/lscratch/$SLURM_JOB_ID/support-training` directory. Do not place
anything there that must survive cancellation, node failure, or normal job
completion. A resumed job receives a new empty cache and reconstructs it from
the unchanged training TIFFs on `/data`.

#### Optional Slack notifications

Training can post an initial job summary, the beginning and successful end of
each epoch, current training/validation losses, a compact loss curve, final
completion, and ordinary Python failures. Notifications are disabled by
default and do not affect checkpoint/resume identity.

A Slack Incoming Webhook URL is a bearer secret. Do not put it in this
repository, a notebook cell, a command-line URL argument, or a Slurm output
file. Create or rotate the Incoming Webhook, then store the new plain URL as
the only line of a private file on Biowulf:

```bash
install -d -m 700 /data/$USER/.secrets
nano /data/$USER/.secrets/support-slack-webhook
chmod 600 /data/$USER/.secrets/support-slack-webhook
```

`requests` is now a direct dependency in `environment.yml`. If the existing
Biowulf environment predates this change and does not already provide it, run
`conda install -n SUPPORT requests` once before submitting.

Enable notifications by passing that file path. The file path is not secret;
its contents are never printed, added to trainer parameters, or saved in the
run manifest:

```bash
sbatch scripts/biowulf_train.sbatch \
  --training-directory /data/$USER/training_tiffs \
  --test-directory /data/$USER/validation_tiffs \
  --dimensionality '3+1D' \
  --exp-name calcium_4D \
  --slack-webhook-file /data/$USER/.secrets/support-slack-webhook
```

Alternatively, use `--slack-notifications` and set
`SUPPORT_SLACK_WEBHOOK_FILE` before submission. An environment-only
`SUPPORT_SLACK_WEBHOOK_URL` is supported as a less-preferred fallback. Do not
type the secret URL directly into an `sbatch --export` command because it can
remain in shell history.

After each completed epoch, the notification contains comparable Unicode
sparklines for training and validation losses. A full PNG is atomically updated
at `<experiment>/<exp_name>_loss.png`. Incoming Webhooks cannot upload that
local PNG; Slack file upload would require a separate bot token and channel, so
the webhook message reports the shared-filesystem path instead.

Webhook/network/plot failures only produce a redacted warning and never stop
training. The runner can report normal Python exceptions, including DataLoader
errors, before re-raising them. No in-process notifier can guarantee a message
after `SIGKILL`, node loss, kernel death, total network loss, or an abrupt power
failure.

For `train_gen.ipynb`, set `slack_notifications = True` in its configuration
cell and arrange `SUPPORT_SLACK_WEBHOOK_FILE` before launching Jupyter. Keep the
URL itself out of the notebook because saved cells and outputs persist.

`--n-epochs` is the final total, not a number of additional epochs. The value
defaults to the selected `gen_parameters.json` section. There is no longer a
`START_EPOCH` setting: resubmit the same command and the trainer automatically
continues after the latest complete bundle. Each epoch's model/optimizer/(when
AMP is enabled) scaler bundle is committed before validation starts. If a job
ends during training, the newest rolling bundle is loaded and that unfinished
epoch repeats. If it ends during validation, a small atomic pending-validation
record remains beside the checkpoint; resubmitting reloads the saved epoch,
retries only its validation, and then advances to the next training epoch.

The first launch atomically creates `.biowulf_train_config.json`. It records the
effective train/test settings, parameter-file hash, validation destinations,
and each TIFF's resolved path, shape, dtype, size, timestamp, device, and inode.
Every resume must match that identity; only the final `--n-epochs` target may be
changed. The recursive inventories are checked again while the job lock is held
and those exact path lists are then frozen for the trainer and validator. This
is a fast metadata-based TIFF identity check rather than a full content hash, so
do not deliberately rewrite a TIFF while preserving all of its filesystem
metadata.

An older notebook experiment has no manifest and is refused by default even if
checkpoint-looking files exist. After manually confirming that its data and
settings match the command, pass `--adopt-existing-run` once; adoption still
requires a complete model/optimizer/(when AMP is enabled) scaler bundle, or a
legacy model-only checkpoint. To intentionally start at epoch zero, use a new
experiment name. `--no-resume` is also available but is rechecked while locked
and refuses any existing experiment or validation artifacts.

Do not run two training jobs against the same output-directory/experiment-name
pair or equal/nested validation output. The runner holds nonblocking
`.biowulf_train.lock` and `.biowulf_validation.lock` files plus hierarchical
byte-range locks in `notebooks/outputs/.biowulf_training_namespace.lock` for the
entire run. Equal, nested, and cross-role output roots conflict; independent
sibling experiments remain concurrent. The per-run lock files remain afterward
as owner information, but only active advisory locks block a job.

All jobs from one repository checkout automatically share the namespace
registry. If multiple SUPPORT checkouts can write the same outputs, point every
job at one absolute shared registry file with, for example,
`sbatch --export=ALL,SUPPORT_TRAIN_LOCK_REGISTRY=/data/$USER/support-train.lock ...`.
The registry file must be outside all training, validation, experiment, and
validation-output roots.
Never delete, rename, or replace that persistent registry while any training
job may be running: a replacement inode would create a separate lock namespace.

By default, outputs are under `notebooks/outputs/EXP_NAME`:

```text
notebooks/outputs/calcium_4D/
├── .biowulf_train_config.json
├── .biowulf_train.lock
├── calcium_4D_train.log
├── calcium_4D_train.json
├── calcium_4D_train_losses.csv
├── calcium_4D_train_validation_losses.csv
├── calcium_4D_validation_files.csv
├── calcium_4D_loss.png             # when Slack notifications are enabled
├── pth/
│   ├── .pending_validation.json    # only while validation is pending
│   ├── model_0.pth
│   ├── optimizer_0.pth
│   ├── scaler_0.pth
│   ├── model_1_batch_2134.pth      # one rolling bundle during an epoch
│   ├── optimizer_1_batch_2134.pth
│   ├── scaler_1_batch_2134.pth
│   └── ...
└── validation/
    ├── .biowulf_validation.lock
    └── source-relative/path/example_SPRT.tif
```

Checkpoint indices are zero-based. Keeping every bundle can consume substantial
quota, and each quartile briefly writes a new bundle before deleting the old
rolling one, so monitor `checkquota` and the `pth` directory. To promote a
chosen model for `biowulf_test.sbatch`, copy only its model state into
`pretrained_models` using the test runner's convention, for example:

```bash
cp notebooks/outputs/calcium_4D/pth/model_13.pth \
  pretrained_models/calcium_4D_epoch013_20260904.pth
```

Use the analogous `2D`, `3D`, or `4D` pretrained label from the table above.
Keep the optimizer and scaler bundles in the experiment directory for resume.

Monitor with:

```bash
squeue --me
tail -f support-train-JOB_ID.out
tail -f notebooks/outputs/calcium_4D/calcium_4D_train.log
column -s, -t notebooks/outputs/calcium_4D/calcium_4D_train_validation_losses.csv | tail
sacct -j JOB_ID
```

Validation outputs can temporarily represent a mix of epochs if a job stops
mid-validation. The paired CSV is written only after all validation files
finish; resubmitting reuses the already-saved checkpoint, reruns that epoch's
validation files in full, and refreshes the complete set without retraining.

## Pretrained-model TIFF denoising

`biowulf_test.sbatch` recursively denoises channel-separated TIFFs with pretrained
SUPPORT models. It is designed so that the same command can be submitted more
than once: workers claim one TIFF at a time through one shared, locked progress
CSV and skip files already begun or finished by another worker.

The denoising workflow is intentionally destructive. After a denoised TIFF has
been written atomically, reopened, checked for the same shape and `float32`
dtype, and recorded in the CSV, its original noisy TIFF is deleted. Source
metadata folders are retained. Keep an independent backup of every noisy TIFF
before running this script. Biowulf `/data` is active storage and is not backed
up.

### Naming and directory layout

Input TIFF paths must contain both an exact channel token (`Ch1`, `Ch2`, and so
on) and a nominal dimension token (`_2D`, `_3D`, or `_4D`). Matching is
case-insensitive, and `Ch1` does not match `Ch10`. The default channel map is:

| Channel | Type |
| --- | --- |
| `Ch3` | `calcium` |
| `Ch2` | `plasma` |
| `Ch1` | `leukocyte` |

Dimension tokens select these existing SUPPORT modes:

| Name token | TIFF array | SUPPORT mode |
| --- | --- | --- |
| `4D` | `TZYX` | `3+1D` |
| `3D` | `ZYX` | `3D` |
| `2D` | `TYX` (a planar time series) | `2+1D` |

Do not rely on TIFF `ndim` to distinguish nominal 2D and 3D data: both are
three-dimensional arrays. If a whole input tree lacks dimension tokens, pass a
single explicit `--data-dims 2D`, `3D`, or `4D` override.

Model files may be anywhere below the model directory and must follow:

```text
[type]*[2D|3D|4D]*epoch[exactly three digits]*[YYYYMMDD].pth
```

For example:

```text
pretrained_models/calcium_4D_epoch013_20260826.pth
```

When several models match one type/dimension pair, the runner chooses the newest
training date, then the highest epoch, then the alphanumerically last path. The
selected full model path is saved in the CSV before inference starts.

The model architecture is constructed from `notebooks/gen_parameters.json` by
default. A checkpoint must have been trained with the matching patch depth and
channel settings or model loading will fail and preserve the source. The runner
only reads that JSON and never edits it. If weights require another compatible
configuration, pass its path with `--parameters`.

Outputs mirror paths relative to `test_directory`; only the test root itself is
removed from the mirrored path. For example:

```text
test_directory/
└── 6xFAD_TRD_GC8_M154_20250916_FOV1_4D/
    ├── 6xFAD_TRD_GC8_M154_20250916_FOV1_4D_Ch1.tif
    └── info/
        └── acquisition.xml

SUPPORT/notebooks/outputs/
└── 6xFAD_TRD_GC8_M154_20250916_FOV1_4D/
    ├── 6xFAD_TRD_GC8_M154_20250916_FOV1_4D_Ch1_SPRT.tif
    └── info/
        ├── acquisition.xml
        └── *_denoise_*.log
```

The contents of source directories named `info` or beginning with `info_`,
`info-`, or `info ` (including `info_bidirectionalz`) are recursively copied
into the corresponding output `info` directory. TIFFs inside an `info*`
directory are treated as metadata and copied, not denoised. Source `info*`
directories are never deleted.

### 1. Prepare the files

Biowulf access requires the NIH network or NIH VPN. Stage the repository, noisy
TIFF tree, and pretrained models under shared `/data/$USER`; do not use
job-local `/lscratch` for inputs, outputs, or the progress CSV. Helix and Biowulf
share `/home` and `/data`, and NIH recommends Globus for substantial transfers.
For a smaller transfer from a macOS or Linux workstation, for example:

```bash
rsync -a --progress "/local/path/SUPPORT/" \
  NIH_USERNAME@helix.nih.gov:/data/NIH_USERNAME/SUPPORT/

rsync -a --progress "/local/path/noisy_tiffs/" \
  NIH_USERNAME@helix.nih.gov:/data/NIH_USERNAME/noisy_tiffs/
```

Do not perform heavy transfers or computations on the Biowulf login node. NIH
HPC also prohibits PII/PHI unless special arrangements have been made. See the
official [connection guide](https://hpc.nih.gov/docs/connect.html),
[transfer guide](https://hpc.nih.gov/docs/transfer/), and
[storage guide](https://hpc.nih.gov/storage/).

Connect and make lightweight checks:

```bash
ssh NIH_USERNAME@biowulf.nih.gov
checkquota
cd /data/$USER/SUPPORT

find /data/$USER/noisy_tiffs -type f \
  \( -iname '*.tif' -o -iname '*.tiff' \) -print | LC_ALL=C sort
find /data/$USER/SUPPORT/pretrained_models -type f \
  -iname '*.pth' -print | LC_ALL=C sort
```

Confirm that the account running every worker can read the models and can create
outputs and delete TIFFs in `test_directory`. Also confirm enough free quota for
one source and its output to coexist while that file is being verified.

The batch template expects the conda initialization script at
`/data/$USER/conda/etc/profile.d/conda.sh` and an environment named `SUPPORT`.
If yours differs, use the environment overrides shown below. The environment
must have been created from this repository's `environment.yml`.

### 2. Preview the run

Preview discovery, channel/type matching, dimension matching, selected models,
and output paths before enabling deletion:

```bash
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate SUPPORT
python scripts/biowulf_test.py \
  --test-directory /data/$USER/noisy_tiffs \
  --dry-run
```

`--dry-run` does not create the output tree or CSV, does not copy metadata, and
does not delete anything. It reads TIFF headers and `gen_parameters.json`; it
does not modify `gen_parameters.json`.

For a preview away from the login node, an optional small CPU allocation comes
after SSH login (a GPU is unnecessary for `--dry-run`):

```bash
sinteractive --cpus-per-task=2 --mem=4g
cd /data/$USER/SUPPORT
source /data/$USER/conda/etc/profile.d/conda.sh
conda activate SUPPORT
python scripts/biowulf_test.py \
  --test-directory /data/$USER/noisy_tiffs --dry-run
exit
```

An interactive node is not required for the real batch run. See the NIH
[Biowulf user guide](https://hpc.nih.gov/docs/userguide.html) for current Slurm
usage and allocation guidance.

### 3. Submit the job

From the repository root on the Biowulf login node, the defaults are enough when
the standard channel map and `pretrained_models` directory apply:

```bash
sbatch scripts/biowulf_test.sbatch \
  --test-directory /data/$USER/noisy_tiffs
```

To replace the entire default channel map and use another model directory:

```bash
sbatch scripts/biowulf_test.sbatch \
  --test-directory /data/$USER/noisy_tiffs \
  --channel-map Ch3=calcium \
  --channel-map Ch2=plasma \
  --channel-map Ch1=leukocyte \
  --model-directory /data/$USER/models
```

Supplying any `--channel-map` option replaces, rather than extends, the defaults.
Unmapped channels, missing/invalid dimension tokens, missing models, unreadable
TIFFs, and output-name collisions are marked `skipped`; their source TIFFs stay
in place while valid later paths continue.

Command-line Slurm resource flags go before the script path and runner flags go
after it:

```bash
sbatch --time=12:00:00 \
  scripts/biowulf_test.sbatch \
  --test-directory /data/$USER/noisy_tiffs
```

If the repository or conda setup differs from the template defaults:

```bash
sbatch \
  --export=ALL,SUPPORT_REPO_DIR=/data/$USER/SUPPORT,SUPPORT_CONDA_SH=/data/$USER/miniconda3/etc/profile.d/conda.sh,SUPPORT_CONDA_ENV=SUPPORT \
  scripts/biowulf_test.sbatch \
  --test-directory /data/$USER/noisy_tiffs
```

Run `python scripts/biowulf_test.py --help` for all options. Useful overrides
include `--output-directory`, `--progress-csv`, `--batch-size`, and
`--data-dims`. `--use-cpu` is intended for small debugging runs.

### Running two or more workers

Submit the exact same command again; Slurm chooses the nodes:

```bash
sbatch scripts/biowulf_test.sbatch \
  --test-directory /data/$USER/noisy_tiffs
sbatch scripts/biowulf_test.sbatch \
  --test-directory /data/$USER/noisy_tiffs
```

Every worker must resolve the same absolute `test_directory`, output directory,
and progress CSV on shared `/data`. The default CSV is:

```text
SUPPORT/notebooks/outputs/denoising_progress.csv
```

A stable `denoising_progress.csv.lock` sidecar coordinates updates using POSIX
advisory locks, and the CSV itself is replaced atomically after each update.
Workers inspect files in natural alphanumeric order and claim only one at a
time. A second worker skips the current `begun` row and claims the next available
path; it does not reserve the rest of the queue.

The `.lock` sidecar is persistent, so seeing it while no job is running is
normal. Never delete or rename it while any worker might still run: a replacement
lock file would be a different inode and could split coordination between old
and new workers.

### Progress, logs, and status meanings

The Slurm job writes stdout/stderr to `support-test-JOB_ID.out` in the submission
directory. Each claimed TIFF also appends an operation log under its mirrored
output `info` folder.

Useful checks are:

```bash
squeue --me
sjobs
tail -f support-test-JOB_ID.out
watch -n 10 'tail -n 30 /data/$USER/SUPPORT/notebooks/outputs/denoising_progress.csv'
jobload -j JOB_ID
sacct -j JOB_ID
jobhist JOB_ID
```

Do not poll `squeue` in a tight loop; NIH recommends roughly two minutes between
scheduler queries. `jobload`, `jobhist`, and related commands are described in
the official [Biowulf tools guide](https://hpc.nih.gov/docs/biowulf_tools.html).

The CSV contains the canonical source/output/model paths, channel/type,
dimension, source fingerprint, owner job/node/PID, attempt count, error, and UTC
timestamps. Statuses are:

| Status | Meaning |
| --- | --- |
| `begun` | One worker owns the TIFF; other workers skip it. |
| `output_saved` | A final output exists; verification, deletion, or final bookkeeping still needs recovery. |
| `finished` | The output was verified and the original noisy TIFF was deleted. |
| `failed` | Inference or verification failed; the source was preserved and a later submission retries it. |
| `source_changed` | Size/inode/mtime changed during work; the source was preserved for manual inspection. |
| `skipped` | No safe runnable plan existed; the reason is in `error`. |

### Interruption and recovery

Cancel a job with:

```bash
scancel JOB_ID
```

Normal `SIGTERM`/`SIGINT` cancellation records the active row as `failed`, keeps
the source, and allows a later submission to retry it. A node loss or `SIGKILL`
can leave a row at `begun`, or at actively owned `output_saved` after publication
but before deletion. These states are deliberately not reclaimed automatically,
because a large 4D inference may legitimately run for many hours and duplicate
work is more dangerous than waiting.

First prove that the recorded `slurm_job_id` is no longer queued or running:

```bash
squeue -j OLD_JOB_ID
sacct -j OLD_JOB_ID
```

Then resubmit with a conservative age threshold:

```bash
sbatch scripts/biowulf_test.sbatch \
  --test-directory /data/$USER/noisy_tiffs \
  --reclaim-after-hours 12
```

Never use a reclaim threshold shorter than a valid inference can take. Each
attempt has a unique claim token and writes to a token-specific staging TIFF; a
worker that has lost its claim cannot publish or delete the source. If a crash
happened after final output publication, the next legitimate claimant verifies
that output and completes deletion without denoising it again. If deletion
already happened but the final CSV update did not, startup reconciliation marks
the verified output `finished`.

A hard node loss can also leave a hidden `.*.partial.tif` or a `.support_cache`
slice directory below the mirrored output. After proving the recorded job is
terminal, locate these and inspect their paths/sizes:

```bash
find /data/$USER/SUPPORT/notebooks/outputs -type f \
  -name '.*.partial.tif' -print
find /data/$USER/SUPPORT/notebooks/outputs -type d \
  -name '.support_cache' -print
```

They are intermediates, not final `_SPRT.tif` outputs. Remove only the exact
orphan paths belonging to the dead job; never clean them while any associated
worker could still be running.

A preexisting output without a matching recoverable CSV row stops that source
instead of overwriting it. Inspect both files first; only then, if replacement is
intended, add `--overwrite-existing`. A `source_changed` row is blocked by
default. After manually inspecting it, use `--retry-source-changed` to establish
a new source fingerprint; also add `--overwrite-existing` if an old output must
be replaced.

The protocol prevents simultaneous duplicate denoising when every job uses the
same shared CSV. No workflow can guarantee that computation is never repeated
after an unrecorded machine crash before output publication, but completed,
verified output is recovered idempotently and the original is deleted only from
the worker that still owns the matching source fingerprint.
