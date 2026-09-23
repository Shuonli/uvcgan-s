# Scaling and speed of uvcgan-s on the sPHENIX configuration

Session notes, 2026-09-15 .. 2026-09-22. Branch `ddp` of
`github.com/Shuonli/uvcgan-s`, environment `fm4npp`
(`/home/shuhang/miniconda3/envs/fm4npp/bin/python`, torch 2.7.1+cu118,
python 3.12). Data: `data/sphenix/2025-06-05_jet_bkg_sub` (23 GB, untracked,
never `git add` it). Cluster: partition `a6k`; dahlia and ceres are RTX
A6000, curvelet and venus are RTX 6000 Ada (faster) -- never compare numbers
across node types.

All measurements below are of the sPHENIX UVCGAN-S configuration, unless
stated otherwise. The README carries the same tables in a shorter form.

## What to do

1. **`batch_size` 32 with learning rate 1e-4** on a single GPU. Best of
   everything measured: it was the only setting to reach the hardest
   quality targets, roughly 1.7x faster to a given quality than the
   configured batch 4 at 5e-5.
2. **`gp_cache_period` 4** (currently 0). 1.34x shorter steps at batch 32,
   no effect on the optimization other than a gradient penalty that is a
   few steps stale. Validate the quality before trusting it.
3. **`torch.compile` on the two generators** if convenient: a further 1.07x
   at batch 32, 1.15x at batch 4.
4. Multi-GPU is worth little here (~1.12x for two GPUs at a fixed batch);
   use extra GPUs to run several experiments at once instead.
5. Never train across two nodes on this cluster: it is several times slower
   than one GPU.

Items 1-3 together are roughly 2.5x over the configuration as it stands,
all on one GPU.

## Throughput against batch size

One RTX 6000 Ada, `scripts/slurm/batch_sweep.sbatch`, 2 runs per point.

| batch | ms/step | samples/s | iters/s | memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 330 | 3.0 | 3.03 | |
| 4 | 340 | 11.8 | 2.94 | |
| 8 | 340 | 23.5 | 2.94 | |
| 16 | 342 | 46.8 | 2.92 | |
| 32 | 357 | 89.7 | 2.80 | |
| 64 | 519 | 123.4 | 1.93 | 7 GB |
| 128 | 1001 | 127.9 | 1.00 | 11 GB |
| 256 | 2207 | 116.0 | 0.45 | 19 GB |

The step time is flat from batch 1 to 16 and only bends at 64. Memory is
never the limit (19 GB of 48 GB at batch 256). Iterations per second is
*highest* at small batch and falls with it: the two metrics peak at
opposite ends, which is why throughput alone is the wrong thing to
optimize.

## Does a larger batch shorten training?

No, beyond about 32. Progress is limited by the number of updates, not by
the data. `scripts/slurm/diag_batch_size.sbatch` trains several batch sizes
concurrently on identical GPUs; `scripts/slurm/plot_batch_diag.py` plots
each metric against updates and against samples (figure:
`outdir/sphenix/diag/batch_diag.png`). The curves collapse onto the
*update* axis and separate by exactly the batch ratio on the *sample* axis.

Updates needed relative to batch 4 (1.0 = the larger batch saves nothing;
8 or 32 would mean it saves in proportion to its size):

| target `idt_aa_a1` | batch 4 | batch 32 | batch 128 |
| ---: | ---: | ---: | ---: |
| 0.045 | 1.00 | 1.09 | 0.92 |
| 0.040 | 1.00 | 1.29 | 1.06 |
| 0.036 | 1.00 | 1.35 | 1.07 |
| 0.034 | 1.00 | 1.75 | not reached |

Batch 128 needs as many updates as batch 4 while reading 32 times the
data. Batch 32 wins the clock only because its updates cost nearly as
little as batch 4's while making somewhat more progress each.

Repeated from the middle of training (branch off a 20k-update checkpoint,
`scripts/slurm/diag_mid_training.sbatch`, every arm continuing the same
optimizer state) the ordering holds. Hours to reach a `cycle_b` target:

| target | batch 4 | batch 32 | batch 128 |
| ---: | ---: | ---: | ---: |
| 0.030 | 1.83 | 0.98 | never |
| 0.028 | 3.37 | 1.93 | never |
| 0.026 | never | 3.83 | never |
| best in 4.5 h | 0.0265 | 0.0247 | 0.0311 |

**Metric caveat.** `idt_aa_a1` saturates by ~20k updates and stops
discriminating; `cycle_b`, `idt_bb` and `cycle_a1` keep moving and should
be used for any comparison at depth. All of these are training losses:
the val and test splits hold only the `embed` domain, so the separation
metric cannot be computed on held-out data without re-splitting the inputs.

## Can the learning rate be raised with the batch?

No: the rate ceiling is set by stability and does not move with the batch.
`scripts/slurm/diag_lr_scaling.sbatch`, best `cycle_b` in 4.5 h from
initialization:

| batch | 5e-5 | 1e-4 | 1.5e-4 | 2e-4 | 3e-4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.0305 | | | | |
| 16 | | 0.0300 | | | |
| 32 | 0.0323 | **0.0264** | diverged | diverged | diverged |
| 64 | | 0.0433 | diverged | diverged | |
| 128 | 0.0533 | 0.0524 | | diverged | |

1e-4 is stable and 1.5e-4 diverges at batch 32 *and* at batch 64, so the
ceiling is a property of the model, not of the batch. Square-root scaling
would want ~2.8e-4 at batch 128, which is far past where this GAN breaks:
the larger batch can never be cashed in.

## What a step costs, and how to shorten it

A step issues about **33000 CUDA kernels of 7 us each**. On 24x64 images
almost none is limited by arithmetic. At batch 4 the GPU is idle ~49% of
the step; at batch 32 the same number of kernels each do more work and the
GPU reaches ~98% busy. Phases at batch 4: generator step 217 ms,
discriminator step 265 ms, EMA 3.6 ms (fusing the EMA saves 0.4%, not
worth it).

Measured on one A6000:

| | batch 4 | batch 32 |
| :--- | ---: | ---: |
| as configured | 467 ms | 593 ms |
| `gp_cache_period` 4 | 371 ms (1.26x) | 443 ms (1.34x) |
| `torch.compile` generators | 405 ms (1.15x) | 552 ms (1.07x) |
| both | 330 ms (1.41x) | 399 ms (1.48x) |
| no gradient penalty at all | 334 ms (1.38x) | |

The gradient penalty's second backward through all three discriminators is
38% of the step. `torch.compile` helps less at batch 32 because the GPU is
busy anyway. CUDA graphs (`mode='reduce-overhead'`) would address the idle
half of the small-batch step but never finished capture in 25 minutes: the
step keeps tensors across its several backward passes and would have to be
restructured.

## Multi-GPU

DDP was added in this branch; training scripts run unchanged under
`torchrun` or SLURM `srun`, one process per GPU. Samples per second:

| processes | batch/proc | ddp | ddp + bf16 | manual | manual + bf16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 8.6 | | | |
| 2 | 4 | 15.9 | | | |
| 4 | 4 | 15.0 | 24.8 | | |
| 8 | 4 | 17.2 | 38.7 | 18.0 | |
| 1 | 32 | 54.0 | | | |
| 4 | 32 | 106.8 | | | |
| 8 | 32 | 194.2 | 272.8 | 151.0 | 217.2 |

Two GPUs, same node vs two nodes (Ada, 1 Gb ethernet between nodes, no
InfiniBand, measured all-reduce 102 MiB/s):

| batch/GPU | 1 GPU | 2 GPUs one node | 2 GPUs two nodes |
| :--- | ---: | ---: | ---: |
| 4 | 11.8 | 22.6 (1.92x) | 1.7 (0.14x) |
| 32 | 89.7 | 165.1 (1.84x) | 13.4 (0.15x) |
| 32, bf16 | | 165.4 (1.84x) | 23.9 (0.27x) |

A step all-reduces ~470 MiB; across nodes that is ~4.5 s against ~0.35 s of
compute. **Since training is update-limited, the weak-scaling numbers above
do not translate into faster training** -- the honest value of a second GPU
is the strong-scaling figure, 1.12x at a fixed effective batch.

### The DDP deadlock

**Eight processes deadlock in an all-reduce in about a quarter of runs**
(9 of 35), always at `SeqNum=57 ALLREDUCE` with one rank out of step. Every
configuration is affected: batch 4 and 32, with and without compression,
every bucket setting. 1, 2 and 4 processes never deadlocked.

Cause: a CycleGAN step runs several forward and backward passes per
iteration with `no_sync` interleaved and `requires_grad` toggled, which is
outside the model DDP assumes -- the same mismatch makes `static_graph=True`
fail outright with an `expect_autograd_hooks_` assert. The exact race
inside the reducer was not isolated.

Workaround, and what to use above four processes:
`UVCGAN_S_DDP_MODE=manual` leaves the models unwrapped and all-reduces each
optimizer's gradients once per step from Python, in a fixed order. **0
deadlocks in 12 runs**, free at batch 4, about 20% slower at batch 32.

Knobs: `UVCGAN_S_DDP_MODE`, `UVCGAN_S_DDP_COMPRESS` (fp16/bf16, worth
1.4-2.25x at 4-8 processes, nothing at 2), `UVCGAN_S_DDP_BUCKET_MB`,
`UVCGAN_S_DDP_BUCKET_VIEW`, `UVCGAN_S_DDP_FIND_UNUSED`,
`UVCGAN_S_DDP_TIMEOUT_MIN`.

## Bugs found and fixed on this branch

- LR schedulers no longer take `verbose` (torch 2.7); it was forced on
  every scheduler and blocked *every* configuration at startup.
- `DataFrame.append` was removed in pandas 2.
- `prefetch_factor` may not be passed when `workers` is 0.
- Checkpoints: strip any wrapper (DDP, DataParallel, and `torch.compile`,
  whose `_orig_mod.` parameter prefix broke the moving average and would
  have leaked into checkpoints).
- Resuming restored the optimizer's saved learning rate, silently ignoring
  the configured one; the configuration now wins, with a warning.
- h5 datasets open their file lazily in the reading process (handles are
  not fork-safe).
- Single-node rendezvous uses the loopback address: venus resolves its own
  name to an unroutable address and curvelet to 127.0.1.1, which hung every
  multi-process run. Across nodes each task must bind NCCL to its 10.20.x
  interface (`scripts/slurm/with_subnet_iface.sh`) or initialization hangs.
- The benchmark scripts read `$?` after a `$(date)` substitution and so
  reported the exit status of `date`; a deadlocked run looked successful.
- `clip_gradients` now returns the pre-clip gradient norm, logged as
  `gnorm_gen` / `gnorm_disc`: the same nominal learning rate is not the same
  step once `grad_clip` binds, and whether it binds was never measured.

## Still open

- **Job 19989** (base run, ceres) was still training when these notes were
  written: batch 32 at 5e-5, ~76k updates and 7 checkpoints so far, 26 h
  limit. It exists to allow a *deeper* repeat of the batch size measurement
  (`BRANCH_EPOCH=<epoch> sbatch scripts/slurm/diag_mid_training.sbatch`).
  Given the measurement already agrees from scratch and from 20k updates,
  a third point has low marginal value -- but the checkpoints are the
  furthest-trained model available.
- The mid-training branches all ran at 5e-5 because resuming pinned the
  rate; now that the configuration wins, a deep branch can include rate
  variants.
- Whether a stale gradient penalty (`gp_cache_period`) costs quality.
- A held-out quality metric. Everything here is a training loss, and the
  data layout offers no paired ground truth outside the training split.
- The 8-process deadlock is worked around, not understood.

## Scripts

| path | purpose |
| :--- | :--- |
| `scripts/slurm/bench_pack.sbatch` | multi-GPU scaling inside one allocation |
| `scripts/slurm/collect_bench.py` | tabulate the above |
| `scripts/slurm/batch_sweep.sbatch` | throughput against batch size, one GPU |
| `scripts/slurm/scaling_2gpu.sbatch` | weak vs strong scaling on two GPUs |
| `scripts/slurm/bench_topology.sbatch` | one node vs two nodes |
| `scripts/slurm/hang_stats.sbatch` | how often a configuration deadlocks |
| `scripts/slurm/diag_batch_size.sbatch` | batch sizes from initialization |
| `scripts/slurm/diag_lr_scaling.sbatch` | batch size and rate raised together |
| `scripts/slurm/diag_base_run.sbatch` | base run keeping checkpoints |
| `scripts/slurm/diag_mid_training.sbatch` | repeat the above from a checkpoint |
| `scripts/slurm/branch_from_checkpoint.sh` | copy a checkpoint into a new model dir |
| `scripts/slurm/plot_batch_diag.py` | updates-vs-samples figure and crossings |
| `tests/test_ddp.py` | two-process gloo test, weights must stay identical |

## How far to trust this

Every number is from at least two runs except where noted, and runs that
are compared always shared a node. The convergence conclusions rest on
training losses over the first few percent of a full-length run (the
configured run is 400 x 2000 = 800k updates; these experiments reach 10-80k),
measured on one configuration and one dataset. The throughput and deadlock
numbers are solid; the batch size and learning rate conclusions are well
supported but were not verified against a physics metric, which is the
obvious next step before committing to a long run.
