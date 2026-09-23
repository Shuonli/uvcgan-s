# Scaling and speed of uvcgan-s on the sPHENIX configuration

Session notes, 2026-09-15 .. 2026-09-23. Branch `ddp` of
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

**Items 1 and 2 rest on training losses** (item 1 partly on `cycle_b`,
which turned out not to track the extraction, c.f. "Held-out quality
against truth"). Job 20016 re-checks both against held-out truth.

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

**Metric caveat.** `idt_aa_a1` saturates by ~20k updates, while
`cycle_b`, `idt_bb` and `cycle_a1` keep moving, so the comparison above
and the learning rate sweep below used `cycle_b` at depth. All of these
are training losses, and the held-out scores (next sections) show that
`cycle_b` does **not** track the extraction: `idt_aa_a1` does. The
conclusions drawn from `cycle_b` are being re-checked against held-out
truth (job 20016).

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

## Held-out quality against truth

**The val split does carry truth.** Every val `embed` event is an event of
the PYTHIA production in `train/signal.h5` embedded into HIJING: the index
files name the same `(file, event)` -- all 200k val and 633k train mixed
events have their counterpart, and the keys are unique in the signal
file -- and the mixed image contains the signal image tower by tower
(embed >= signal in every tower for all 20k sampled val pairs and 2k
sampled train pairs; for shuffled pairs half the signal energy is
missing). The background truth is `embed - signal`, there is no noise.
So the training data are paired too; the model is trained unpaired and
never uses it. Caveat: the val events' signal images are among the 2.6M
unpaired training images of the signal domain -- the mixed events and the
pairing are held out, the signal images are not.

`scripts/slurm/eval_val_truth.py` scores checkpoints on a fixed random
sample of 20k val events (cached under `OUTDIR/sphenix/val_truth/`,
~5 s per network on a GPU):

- `l1_sig`, `l1_bkg`: per-tower L1 error of the extracted signal and
  background, GeV -- the held-out counterparts of `idt_aa_a1`, `idt_aa_a0`;
- jet energy in a R=0.4 cone around the leading truth jet (|eta| < 0.7 and
  >= 10 GeV: 17.5k of the 20k events), extracted against truth: `jes`
  (mean ratio), `bias` (mean difference) and resolutions `jer` (standard
  deviation of the difference) and **`jer_cal`** (spread around a linear
  fit of extracted against true energy, divided by its slope, i.e. the
  resolution after an offset and scale calibration). Use `jer_cal`: the
  networks' energy scales differ (0.75 to 1.0), and an energy scale below
  one shrinks `jer` without making the extraction any better.

Truth cone energies: median 32 GeV (5-95%: 23-44); the HIJING background
in the same cone is 40 +- 10 GeV. Reference points for `jer_cal`: a
per-eta-row median-rho subtraction of the mixed event gives 5.28 GeV
(`bias` +4.4: the median under-estimates the skewed background);
subtracting the *true* background's per-row mean -- the best any per-row
rho can do -- 5.09 GeV.

Base run (batch 32 at 5e-5, job 19989), one row per checkpoint:

| updates | raw `l1_sig` | raw `jes` | raw `jer_cal` | ema `l1_sig` | ema `jes` | ema `jer_cal` | init share of EMA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 0.0365 | 0.87 | 4.12 | 0.272 | 0.75 | 9.25 | 37% |
| 20k | 0.0304 | 0.81 | 3.92 | 0.097 | 0.84 | 5.77 | 14% |
| 30k | 0.0412 | 0.93 | 3.86 | 0.049 | 0.95 | 4.04 | 5% |
| 40k | 0.0361 | 0.99 | 3.77 | 0.0375 | 0.98 | 3.74 | 1.8% |
| 50k | 0.0296 | 0.92 | 3.70 | 0.0339 | 0.98 | 3.68 | 0.7% |
| 60k | 0.0295 | 0.90 | 3.69 | 0.0327 | 0.97 | 3.65 | 0.2% |
| 70k | 0.0294 | 0.91 | 3.70 | 0.0324 | 0.96 | 3.64 | 0.1% |
| 80k | 0.0305 | 0.90 | 3.67 | 0.0329 | 0.96 | 3.63 | 0.03% |

1. The model resolves the jet energy ~30% better than either reference
   (3.6 against 5.1-5.3 GeV), and it keeps improving slowly with
   training: `jer_cal` is still falling at 80k updates.
2. **The EMA generator -- the one inference uses -- starts as a copy of
   the random initialization and keeps a share `0.9999^updates` of it**:
   37% at 10k updates, 14% at 20k. It is useless before ~30k updates, and
   from ~40k on it is ahead of the raw generator on `jer_cal`, by a few
   hundredths of a GeV (not on `l1_sig`, where averaging costs a little).
   Runs shorter than ~50k
   updates have to be judged on the raw generator; for the configured
   800k updates it does not matter.
3. The raw generator's energy scale wanders from checkpoint to checkpoint
   (`jes` 0.81 to 0.99), and so does its `l1_sig` (0.029 to 0.041); its
   calibrated resolution is much steadier.
4. **`idt_aa_a1` tracks held-out `l1_sig`** (Spearman 0.93 over the
   seven checkpoints to 70k, the bump at 30k included). **`cycle_b` does
   not**: it drops by 40% from 20k to 70k updates while the held-out
   `l1_sig` stays flat.
5. The gradient clip (`grad_clip` 0.5) binds on every step, by a lot: in
   the first hours of the runs of job 20016 the pre-clip norm is
   ~150-190 for the generators and ~20-110 for the discriminators, at
   batch 4 and 32 alike. Adam therefore runs on normalized gradients; the
   clip is a normalization, not a safeguard.

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
  step once `grad_clip` binds. It binds on every step (see above).

## Still open

- **Job 20016** (dahlia, 6 GPUs, 7 h from 2026-09-22 23:55): the
  configured batch 4 at 5e-5 against batch 32 at 1e-4 with
  `gp_cache_period` 0 and 4, two seeds each, scored on held-out truth
  every epoch (`MODEL/val_truth_history.csv`) and checkpointed every 10k
  updates. Analyse with `python scripts/slurm/plot_heldout.py --job 20016`
  and score the checkpoints with `eval_val_truth.sbatch` for `jer_cal`,
  which the inline scores lack (they were started before it existed).
  Its seed-0 runs of batch 4 and batch 32 at 1e-4 repeat the runs of job
  19988 (identical losses to 4-5 digits in the first epoch).
- **Job 19989** (base run, ceres, batch 32 at 5e-5, 26 h limit, ends
  ~2026-09-23 12:30): checkpoints every 10k updates, the furthest-trained
  model available; its held-out scores are in `MODEL/evals/val_truth.csv`
  (rerun `eval_val_truth.sbatch` on it to add new checkpoints). A deeper
  repeat of the batch size measurement from it has low value now that
  `cycle_b` is known not to track the extraction.
- The mid-training branches all ran at 5e-5 because resuming pinned the
  rate; now that the configuration wins, a deep branch can include rate
  variants.
- The EMA carries the random initialization for tens of thousands of
  updates. A warm-up of its momentum (e.g. `min(m, (1 + t) / (10 + t))`)
  would make short runs usable at inference; it does not change the
  training.
- The test split (JEWEL) has no truth in these files.
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
| `scripts/slurm/eval_val_truth.py` | score checkpoints against the val truth |
| `scripts/slurm/eval_val_truth.sbatch` | the above on a GPU node |
| `scripts/slurm/diag_heldout.sbatch` | recommended settings, scored on truth every epoch |
| `scripts/slurm/plot_heldout.py` | held-out scores against time and updates |
| `tests/test_ddp.py` | two-process gloo test, weights must stay identical |

## How far to trust this

Every number is from at least two runs except where noted, and runs that
are compared always shared a node. The convergence conclusions rest on
training losses over the first few percent of a full-length run (the
configured run is 400 x 2000 = 800k updates; these experiments reach 10-80k),
measured on one configuration and one dataset. The throughput and deadlock
numbers are solid. The batch size and learning rate conclusions were
drawn from training losses, partly from `cycle_b`, which the held-out
scores show does not track the extraction; they are being re-checked
against held-out truth (job 20016). The held-out scores are a proxy for
the physics as well -- the energy in a fixed cone around the true jet
axis, not jets found in the extracted image -- but they are measured
against truth on events the model never saw.
