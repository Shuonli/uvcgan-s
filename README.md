# UVCGAN-S: Stratified CycleGAN for Unsupervised Data Decomposition



## Overview


This repository provides a reference implementation of Stratified CycleGAN
(UVCGAN-S), an architecture for unsupervised signal extraction from mixed data.

What problem does Stratified CycleGAN solve?

Suppose you have three datasets. The first contains clean signals. The second
contains backgrounds. The third contains mixed data where signals and
backgrounds have been combined in some complicated way. You don't know exactly
how the mixing happens, and you can't find pairs that show which clean signal
corresponds to which mixed observation. But you need to take new mixed data and
decompose it back into signal and background components.

Stratified CycleGAN learns to do this decomposition from unpaired examples.
You show it random samples of signals, random samples of backgrounds and mixed
data, and it figures out both how to combine signals and backgrounds into
realistic mixed data, and how to decompose mixed data back into its parts.

<p align="center">
  <img src="https://raw.githubusercontent.com/LS4GAN/gallery/refs/heads/main/uvcgan-s/sphenix.png" width="75%" title="Stratified CycleGAN automatically decomposes mixed data into signal and background components">
</p>


See the [Quick Start](#quick-start-guide) section for a concrete example
using cat and dog images.


## Installation

The package was tested only under Linux systems.


### Environment Setup

Development environment based on
`pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime` container.

There are several ways to setup the package environment:

**Option 1: Docker**

Download the docker container `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`.
Inside the container, create a virtual environment to avoid package conflicts:
```bash
python3 -m venv --system-site-packages ~/.venv/uvcgan-s
source ~/.venv/uvcgan-s/bin/activate
```

**Option 2: Conda**
```bash
conda env create -f contrib/conda_env.yaml
conda activate uvcgan-s
```

### Install Package

Once the environment is set, install the `uvcgan-s` package and its
requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

### Environment Variables

By default, UVCGAN-S reads datasets from `./data` and saves models to
`./outdir`. If any other location is desired, these defaults can be overriden
with:
```bash
export UVCGAN_S_DATA=/path/to/datasets
export UVCGAN_S_OUTDIR=/path/to/models
```

## Quick Start Guide

This package was developed for sPHENIX jet signal extraction. However, jet
signal analysis requires familiarity with sPHENIX-specific reconstruction
algorithms and jet quality analysis procedures. This section demonstrates
application of Stratified CycleGAN method on a simpler toy problem with
intuitive interpretation. The toy example illustrates the basic workflow and
serves as a template for applying the method to your own data.

<p align="center">
  <img src="https://raw.githubusercontent.com/usert5432/gallery/refs/heads/uvcgan-s/uvcgan-s/toy_mixes.png" width="95%" title="Toy images">
</p>

The toy problem is this: we have images that contain a blurry mix of cat and
dog faces. The goal is to automatically decompose these mixed images into
separate cat and dog images. To this end, we present Stratified CycleGAN with
the mixed images, a random sample of cat images, and a random sample of dog
images. By observing these three collections, the model learns how cats look on
average, how dogs look on average, and what would be the best way to decompose
the current mixed image into a cat and a dog. Importantly, the model is never
shown training pairs like "this specific mixed image was created from this
specific cat image and this specific dog image." It only sees random examples
from each collection and figures out the decomposition on its own.

### Installation

Before proceeding further, the package needs to be installed following
instructions at the top of this README, if not installed already.

### Dataset Preparation

The toy example uses cat and dog images from the AFHQ dataset. To download and
preprocess it:

```bash
# Download the AFHQ dataset
./scripts/download_dataset.sh afhq

# Resize all images to 256x256 pixels
python3 scripts/downsize_right.py -s 256 256 -i lanczos \
    "${UVCGAN_S_DATA:-./data}/afhq/" \
    "${UVCGAN_S_DATA:-./data}/afhq_resized_lanczos"
```

The resizing script creates a new directory `afhq_resized_lanczos` containing
256x256 versions of all images, which is the format expected by the training
script.

### Training

To train the model, run the following command:
```bash
python3 scripts/train/toy_mix_blur/train_uvcgan-s.py
```

The script trains the Stratified CycleGAN model for 100 epochs. On an RTX 3090
GPU, each epoch takes approximately 3 minutes, so the complete training process
requires about 5 hours. The trained model and intermediate checkpoints are
saved in the directory
`${UVCGAN_S_OUTDIR:-./outdir}/toy_mix_blur/uvcgan-s/model_m(uvcgan-s)_d(n_layers)_g(vit-modnet)_cat_dog_sub/`.

The structure of the model directory is described in the
[F.A.Q.](#what-is-the-structure-of-a-model-directory).

### Evaluation

<p align="center">
  <img src="https://raw.githubusercontent.com/usert5432/gallery/refs/heads/uvcgan-s/uvcgan-s/toy_decomposition.png" width="95%" title="Toy images">
</p>


After training completes, the model can be used to decompose images from the
validation set of AFHQ. Run:
```bash
python3 scripts/translate_images.py \
    "${UVCGAN_S_OUTDIR:-./outdir}/toy_mix_blur/uvcgan-s/cat_dog_sub" \
    --split val \
    --domain 2 \
    --format image
```

This command takes the mixture cat-dog images (Domain B) and decomposes them
into separate cat and dog components (Domain A). The `--domain 2` flag
specifies that the input images come from Domain B, which contains the mixed
data.

The results are saved in the model directory under
`evals/final/translated(None)_domain(2)_eval-val/`. This evaluation directory
contains several subdirectories:

- `fake_a0/` - extracted cat components
- `fake_a1/` - extracted dog components
- `real_b/` - original blurred mixture inputs

Each subdirectory contains numbered image files (`sample_0.png`,
        `sample_1.png`, etc.) corresponding to the validation set.


### Adapting to Your Own Data

To apply Stratified CycleGAN to a different decomposition problem, use the toy
example training script as a starting point. The script
`scripts/train/toy_mix_blur/train_uvcgan-s.py` contains a declarative
configuration showing how to structure the three required datasets and set up
the domain structure for decomposition. For a more complex example, see
`scripts/train/sphenix/train_uvcgan-s.py`.


## sPHENIX Application: Jet Background Subtraction

The package was developed for extracting particle jets from heavy-ion collision
backgrounds in sPHENIX calorimeter data. This section describes how to
reproduce the paper results.

### Dataset

The sPHENIX dataset can be downloaded from Zenodo: https://zenodo.org/records/17783990

Alternatively, use the download script:
```bash
./scripts/download_dataset.sh sphenix
```

The dataset contains HDF5 files with calorimeter energy measurements organized
as 24×64 eta-phi grids. The data is split into training, validation, and test
sets. Training data consists of three components: PYTHIA jets (the signal
component), HIJING minimum-bias events (the background component), and
embedded PYTHIA+HIJING events (the mixed data). The test set uses JEWEL jets
embedded in HIJING backgrounds. JEWEL models jet-medium interactions
differently from PYTHIA, providing an out-of-distribution test of the model's
generalization capability.


### Training or Using Pre-trained Model

There are two options for obtaining a trained model: training from scratch or
downloading the pre-trained model from the paper.

To train a new model from scratch:
```bash
python3 scripts/train/sphenix/train_uvcgan-s.py
```

The training configuration uses the same Stratified CycleGAN architecture as
the toy example, adapted for single-channel calorimeter data. The trained model
is saved in the directory
`${UVCGAN_S_OUTDIR:-./outdir}/sphenix/uvcgan-s/model_m(uvcgan-s)_d(resnet)_g(vit-modnet)_sgn_bkg_sub`
(see [F.A.Q.](#what-is-the-structure-of-a-model-directory) for details on the
 model directory structure).

Alternatively, a pre-trained model can be downloaded from Zenodo: https://zenodo.org/records/17809156

The pre-trained model can be used directly for evaluation without retraining.


### Evaluation

To evaluate the model on the test set:
```bash
python3 scripts/translate_images.py \
    "${UVCGAN_S_OUTDIR:-./outdir}/path/to/sphenix/model" \
    --split test \
    --domain 2 \
    --format ndarray
```

The `--format ndarray` flag saves results as NumPy arrays rather than images.
The output structure is similar to the toy example: extracted signal and
background components are saved in separate directories under `evals/final/`.
Each output file contains a 24×64 calorimeter energy grid that can be used for
physics analysis.


# Batch Size

The step of this model is dominated by kernel launches rather than by
arithmetic, so on a single GPU a larger batch is almost free until the GPU
finally saturates. Measured on one RTX 6000 Ada (48 GB), sPHENIX
configuration:

| batch | ms / step | samples / s | vs batch 4 | gain over previous | memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
|   1 |  330 |   3.0 |  0.26 |      |       |
|   2 |  340 |   5.9 |  0.50 | 1.94 |       |
|   4 |  340 |  11.8 |  1.00 | 2.00 |       |
|   8 |  340 |  23.5 |  2.00 | 2.00 |       |
|  16 |  342 |  46.8 |  3.98 | 1.99 |       |
|  32 |  357 |  89.7 |  7.63 | 1.92 |       |
|  64 |  519 | 123.4 | 10.50 | 1.38 |  7 GB |
| 128 | 1001 | 127.9 | 10.88 | 1.04 | 11 GB |
| 256 | 2207 | 116.0 |  9.87 | 0.91 | 19 GB |

Every doubling up to 32 returns a full factor of two in throughput for a
few percent of extra time per step; past 64 the step time grows in
proportion to the batch and the throughput saturates, and at 256 it
declines. Memory is never the limit. `batch_size` 32 is a good default on
one GPU, as `Config` already assumes -- the sPHENIX training script
overrides it to 4, which leaves most of the GPU idle.

Note that the batch size is a property of the training run, not only of
its speed: a larger batch means fewer, better gradient estimates per
sample, so the learning rate and the length of the run have to be revised
along with it.

## Does a larger batch shorten training?

Higher throughput only shortens training if the extra samples reduce the
number of updates needed. `scripts/slurm/diag_batch_size.sbatch` trains
batch 4, 32 and 128 concurrently on identical GPUs and
`scripts/slurm/plot_batch_diag.py` draws the quality metrics against both
axes. On the sPHENIX configuration the curves of every batch size fall on
top of each other against the number of **updates**, and are separated by
exactly the batch ratio against the number of **samples**: progress is
limited by the updates, not by the data.

Updates needed to reach a given value of `idt_aa_a1`, relative to batch 4
(1.0 would mean the larger batch saves nothing; 8 and 32 would mean it
saves in proportion to its size):

| target | batch 4 | batch 32 | batch 128 |
| ---: | ---: | ---: | ---: |
| 0.045 | 1.00 | 1.09 | 0.92 |
| 0.040 | 1.00 | 1.29 | 1.06 |
| 0.036 | 1.00 | 1.35 | 1.07 |
| 0.034 | 1.00 | 1.75 | not reached |

Batch 128 needs as many updates as batch 4 while reading 32 times the
data, so its 10x higher throughput is spent entirely on samples that do
not help. In wall clock, reaching 0.034 took 1.49 h at batch 4, 1.07 h at
batch 32 and more than 4.5 h at batch 128. Batch 128 also diverged at a
learning rate of 2e-4, while 1e-4 was stable.

The consequence for several GPUs is that they pay off only by splitting a
fixed batch, not by enlarging it, and splitting is worth about 1.1x here
(c.f. the strong scaling rows above). These runs cover the first few
percent of a full training; the batch size that a run can profit from is
known to grow as training proceeds, so the measurement should be repeated
later in training before drawing conclusions about the whole run.

The figure is written to `OUTDIR/sphenix/diag/batch_diag.png`.


# Distributed Training

Training scripts run unchanged on several GPUs with `torch.distributed`,
one process per GPU. `batch_size` is the batch of a single process, so `N`
processes see `N * batch_size` samples per step; the losses recorded in
`history.csv` are averaged over processes, and each epoch also records its
wall time (`epoch_time`) and throughput (`samples_per_sec`). The process
group is configured from the environment, so either launcher works:

```bash
# torchrun, single node
torchrun --standalone --nproc_per_node=4 scripts/train/sphenix/train_uvcgan-s.py

# SLURM, one task per GPU (c.f. scripts/slurm/bench_ddp.sbatch)
srun --ntasks-per-node=4 --gres=gpu:4 python scripts/train/sphenix/train_uvcgan-s.py
```

Without a distributed launcher the previous behavior is preserved, i.e. a
single process using `DataParallel` over all visible GPUs.

## Gradient synchronization

`UVCGAN_S_DDP_MODE` selects how gradients are synchronized:

 - `ddp` (default) wraps every model into `DistributedDataParallel`, which
   overlaps the all-reduce with the backward pass.
 - `manual` leaves the models bare and all-reduces the gradients of each
   optimizer once per step, from Python, in the fixed order of
   `optimizer.param_groups`.

**A CycleGAN step runs several forward and backward passes per iteration
with `no_sync` interleaved, which is outside the one forward, one backward
model that `DistributedDataParallel` assumes.** On eight processes this
deadlocks in an all-reduce in roughly a quarter of the runs, with one
process left out of step and the others waiting; the same mismatch makes
`static_graph=True` fail outright. `manual` has not deadlocked in any run
and costs nothing at small batches, where the step is bound by kernel
launches rather than by communication, so **prefer `manual` on more than
four processes**.

Both modes keep the weights identical across processes, which
`tests/test_ddp.py` checks by training two processes and comparing them.
BatchNorm statistics are per process in both modes.

Other knobs, all optional:

| variable | effect |
| :--- | :--- |
| `UVCGAN_S_DDP_COMPRESS` | `fp16` / `bf16` gradient compression (both modes) |
| `UVCGAN_S_DDP_BUCKET_MB` | `ddp` bucket size, default 25 |
| `UVCGAN_S_DDP_BUCKET_VIEW` | `ddp` `gradient_as_bucket_view` |
| `UVCGAN_S_DDP_FIND_UNUSED` | `ddp` `find_unused_parameters` |
| `UVCGAN_S_DDP_TIMEOUT_MIN` | collective timeout in minutes, default 30 |

## Measured throughput

sPHENIX configuration on one node of eight RTX A6000, samples per second,
warm-up epoch discarded. `manual` synchronization is safe at every point;
`ddp` deadlocked in about a quarter of the eight process runs.

| processes | batch / process | `ddp` | `ddp` + bf16 | `manual` | `manual` + bf16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 |  4 |   8.6 |      |      |       |
| 2 |  4 |  15.9 |      |      |       |
| 4 |  4 |  15.0 | 24.8 |      |       |
| 8 |  4 |  17.2 | 38.7 | 18.0 |       |
| 1 | 32 |  54.0 |      |      |       |
| 4 | 32 | 106.8 |      |      |       |
| 8 | 32 | 194.2 | 272.8 | 151.0 | 217.2 |

At `batch_size` 4 the step is bound by kernel launches rather than by
communication, so overlapping the all-reduce with the backward pass buys
nothing and `manual` is free; raising the batch is worth far more than
adding processes (a single process at batch 32 is six times faster than at
batch 4). At `batch_size` 32 the overlap does matter and `manual` costs
about a fifth of the throughput, though it is still faster than `ddp`
without gradient compression.

## Two GPUs

Whether a second GPU halves the training time depends on what is held
fixed. Measured on two RTX A6000, three runs per point:

| | 1 GPU | 2 GPUs | |
| :--- | ---: | ---: | ---: |
| same run, effective batch 4 (2 per process) | 8.5 | 8.0 | 0.94 |
| same run, effective batch 32 (16 per process) | 53.7 | 60.2 | 1.12 |
| batch 4 per process, effective batch 4 -> 8 | 8.5 | 15.9 | 1.88 |
| batch 32 per process, effective batch 32 -> 64 | 53.7 | 103.8 | 1.93 |

Splitting an unchanged run across two GPUs gains nothing, because the step
is bound by kernel launches: halving the work per process barely shortens
it, while the all-reduce is added on top. A second GPU pays off only by
doubling the effective batch, which is a different training run.

## Two GPUs on one node or on two nodes

Measured on RTX 6000 Ada, samples per second and time per step, two runs
per point. The nodes have no InfiniBand; between them the all-reduce
reached 102 MiB/s over 1 Gb Ethernet.

| batch / GPU | 1 GPU | 2 GPUs, one node | 2 GPUs, two nodes |
| :--- | ---: | ---: | ---: |
| 4 | 11.8 (340 ms) | 22.6 (355 ms), 1.92x | 1.7 (4845 ms), 0.14x |
| 32 | 89.7 (357 ms) | 165.1 (388 ms), 1.84x | 13.4 (4789 ms), 0.15x |
| 32, bf16 gradients | | 165.4 (387 ms), 1.84x | 23.9 (2682 ms), 0.27x |

A step all-reduces about 470 MiB of gradients, which takes a few
milliseconds inside a node and several seconds between nodes on this
network: two nodes are several times slower than a single GPU. Across
nodes each process has to bind NCCL to an interface the other nodes can
reach, c.f. `scripts/slurm/with_subnet_iface.sh`; otherwise the process
group hangs while initializing.

## Cost of a step

A step of this configuration issues about 33000 CUDA kernels of 7 us each:
on 24x64 images almost none of them is limited by arithmetic, so most of
the cost is per-kernel overhead. At `batch_size` 4 the GPU is idle for
about half of the step waiting for kernels to be launched; at 32 the same
number of kernels simply do more work each, which is why a larger batch is
nearly free until the GPU saturates.

Two changes reduce the cost of a step without touching the batch size,
measured on one RTX A6000:

| | batch 4 | batch 32 |
| :--- | ---: | ---: |
| as configured | 467 ms | 593 ms |
| `gp_cache_period` 4 | 371 ms (1.26x) | 443 ms (1.34x) |
| `torch.compile` on the generators | 405 ms (1.15x) | 552 ms (1.07x) |
| both | 330 ms (1.41x) | 399 ms (1.48x) |

The gradient penalty takes a second backward pass through all three
discriminators and accounts for 38% of the step; `gp_cache_period` reuses
its gradient for several steps, which the model already supports. Removing
the penalty entirely would give 1.38x, so caching recovers most of what it
costs. Whether a stale penalty gradient harms the result is not measured
here and should be checked before relying on it.

`torch.compile` helps less at the larger batch because the GPU is busy
anyway. CUDA graphs (`mode='reduce-overhead'`) would address the idle half
of the small-batch step, but the step keeps tensors between its several
backward passes and did not capture; it would need the training step
restructured.

Unlike a larger batch, neither change alters the optimization, so the
speedup is not paid for in convergence.

## Benchmarks

`scripts/slurm/bench_pack.sbatch` runs a scaling benchmark inside a single
allocation and `scripts/slurm/collect_bench.py` tabulates it;
`scripts/slurm/hang_stats.sbatch` repeats a configuration to measure how
often it deadlocks.


# F.A.Q.

## I am training my model on a multi-GPU node. How to make sure that I use only one GPU?

You can specify GPUs that `pytorch` will use with the help of the
`CUDA_VISIBLE_DEVICES` environment variable. This variable can be set to a list
of comma-separated GPU indices. When it is set, `pytorch` will only use GPUs
whose IDs are in the `CUDA_VISIBLE_DEVICES`.


## What is the structure of a model directory?

`uvcgan-s` saves each model in a separate directory that contains:
 - `MODEL/config.json` -- model architecture, training, and evaluation
    configurations
 - `MODEL/net_*.pth`  -- PyTorch weights of model networks
 - `MODEL/opt_*.pth`  -- PyTorch weights of training optimizers
 - `MODEL/shed_*.pth` -- PyTorch weights of training schedulers
 - `MODEL/checkpoints/` -- training checkpoints
 - `MODEL/evals/`     -- evaluation results


## Training fails with "Config collision detected" error

`uvcgan-s` enforces a one-model-per-directory policy to prevent accidental
overwrites of existing models. Each model directory must have a unique
configuration - if you try to place a model with different settings in a
directory that already contains a model, you'll receive a "Config collision
detected" error.

This safeguard helps prevent situations where you might accidentally lose
trained models by starting a new training run with different parameters in the
same directory.

Solutions:
1. To overwrite the old model: delete the old `config.json` configuration file
   and restart the training process.
2. To preserve the old model: modify the training script of the new model and
   update the `label` or `outdir` configuration options to avoid collisions.


# LICENSE

`uvcgan-s` is distributed under `BSD-2` license.

`uvcgan-s` repository contains some code (primarily in `uvcgan_s/base`
subdirectory) from [pytorch-CycleGAN-and-pix2pix][cyclegan_repo].
This code is also licensed under `BSD-2` license (please refer to
`uvcgan_s/base/LICENSE` for details).

Each code snippet that was taken from
[pytorch-CycleGAN-and-pix2pix][cyclegan_repo] has a note about proper copyright
attribution.

[cyclegan_repo]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
