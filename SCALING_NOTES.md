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

Revised on 2026-09-23 against held-out truth (see "Held-out quality
against truth"). The first version of this list rested on training losses
and recommended batch 32 at 1e-4 and `gp_cache_period` 4; neither holds up.

1. **Train ~100k updates, not the configured 800k.** The published model
   (800k updates at batch 4, five days) resolves the held-out jet energy
   to 3.59 GeV. The base run (batch 32 at 5e-5) reaches 3.62-3.63 GeV
   from 80k updates on, 14-19 h on one A6000, with per-tower errors as
   small or smaller. That is 6-8x less training for <1% in resolution. The warm-up
   has to shrink with the run: the configured one lasts 64k updates, the
   base run used 200. One run each, and only the val PYTHIA embedding and
   a cone energy were measured: confirm with two seeds and the JEWEL test
   before relying on it.
2. **Batch 32 at 5e-5**, the base run's setting. Batch 32 at 1e-4 gets the
   raw generator to a given held-out quality faster during the first ~4 h,
   but the EMA generator, which inference uses, is never better at equal
   time than the configured batch 4 at 5e-5, and at 40k updates both of
   its seeds are behind the (single) batch 32 at 5e-5 run. The "1.7x
   faster" of the first version came from training losses, partly
   `cycle_b`, which does not track the extraction.
3. **Leave `gp_cache_period` at 0.** Caching the gradient penalty for 4
   steps shortens the steps by 1.32x, but the trained generator's held-out
   extraction is worse at every point of a 7 h run (per-tower error +20%,
   jet resolution +0.13-0.20 GeV against the same batch without it), and
   its EMA stays behind that of the configured batch 4 at 5e-5.
4. **`torch.compile` on the two generators** if convenient: 1.07x at batch
   32, 1.15x at batch 4, no effect on the optimization.
5. Judge models with `scripts/slurm/eval_val_truth.py` (`jer_cal`,
   `l1_sig`). Among the training losses `idt_aa_a1` tracks the extraction;
   `cycle_b` does not.
6. Multi-GPU is worth little here (~1.12x for two GPUs at a fixed batch);
   use extra GPUs to run several experiments at once instead.
7. Never train across two nodes on this cluster: it is several times slower
   than one GPU.

Item 1 is worth 6-8x on its own and item 4 another ~1.1x; items 2 and 3
are about not losing quality.

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
are training losses, and the held-out scores show that `cycle_b` does
**not** track the extraction, while `idt_aa_a1` does. Against held-out
truth batch 32 at 1e-4 is not faster where it matters, c.f. "The
recommended settings against held-out truth". The throughput numbers and
the "progress is limited by updates" observation stand.

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
the larger batch can never be cashed in. (The divergences stand; that 1e-4
beats 5e-5 at batch 32 was measured on `cycle_b` and is not borne out by
the held-out scores at 40k updates.)

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
| 90k | 0.0293 | 0.91 | 3.59 | 0.0338 | 0.96 | 3.63 | 0.01% |
| 100k | 0.0304 | 0.97 | 3.60 | 0.0352 | 0.96 | 3.63 | <0.01% |
| 110k | 0.0305 | 0.89 | 3.73 | 0.0357 | 0.96 | 3.62 | <0.01% |

1. The model resolves the jet energy ~30% better than either reference
   (3.6 against 5.1-5.3 GeV). The EMA's `jer_cal` flattens at 3.62-3.63
   GeV from 80k updates on (13.7 training hours).
2. **The EMA generator -- the one inference uses -- starts as a copy of
   the random initialization and keeps a share `0.9999^updates` of it**:
   37% at 10k updates, 14% at 20k. It is useless before ~30k updates.
   From ~40k on it matches the raw generator on `jer_cal` and is much
   steadier (3.62-3.63 from 80k on, against 3.59-3.73 for the raw one);
   on `l1_sig` averaging costs a little. Runs shorter than ~50k updates
   have to be judged on the raw generator; for runs of 100k updates and
   more it does not matter.
3. The raw generator's energy scale wanders from checkpoint to checkpoint
   (`jes` 0.81 to 0.99), and so does its `l1_sig` (0.029 to 0.041); its
   calibrated resolution is much steadier.
4. **`idt_aa_a1` tracks held-out `l1_sig`** (Spearman 0.93 over the
   seven checkpoints to 70k, the bump at 30k included). **`cycle_b` does
   not**: it drops by 40% from 20k to 70k updates while the held-out
   `l1_sig` stays flat.
5. The gradient clip (`grad_clip` 0.5) binds on every step, by a lot. Over
   the 7 h runs of job 20016 the pre-clip norm (epoch means) never drops
   below 120 for the generators (medians 136-251) and grows over training
   for the discriminators (medians 156-1834; below 1 only in the first
   epoch), at batch 4 and 32 alike. Adam therefore runs on normalized
   gradients; the clip is a normalization, not a safeguard.

### The recommended settings against held-out truth (job 20016)

`scripts/slurm/diag_heldout.sbatch`: the configured batch 4 at 5e-5 and
batch 32 at 1e-4 with `gp_cache_period` 0 and 4, two seeds each, 7 h from
initialization on one A6000 each (dahlia, all at once), scored every epoch
on 5k val events (`MODEL/val_truth_history.csv`; figure
`OUTDIR/sphenix/heldout/heldout_20016.png` from `plot_heldout.py`) and at
every 10k-update checkpoint on the 20k (`MODEL/evals/val_truth.csv`, for
`jer_cal`). Calibrated jet resolution in GeV at equal training time,
interpolated between checkpoints, mean of the two seeds [half their
difference]:

| net, training hours | batch 32, 1e-4 | + `gp_cache_period` 4 | batch 4, 5e-5 |
| :--- | ---: | ---: | ---: |
| raw, 2.5 h | **3.89** [0.02] | 4.09 [0.26] | 4.03 [0.08] |
| raw, 3.5 h | **3.79** [0.04] | 3.94 [0.12] | 3.96 [0.06] |
| raw, 4.5 h | 3.84 [0.06] | 3.97 [0.06] | 3.90 [0.06] |
| raw, 5.5 h | 3.86 [0.05] | 4.01 [0.04] | 3.84 [0.04] |
| raw, 6.25 h | 3.85 [0.02] | | 3.85 [0.02] |
| ema, 4.5 h | 4.60 [0.13] | 4.17 [0.27] | **4.05** [0.17] |
| ema, 5.5 h | 4.02 [0.07] | 3.98 [0.16] | **3.86** [0.07] |
| ema, 6.25 h | 3.90 [0.06] | | **3.81** [0.05] |

Raw per-tower error `l1_sig` after 5 h (rolling mean of 10 epochs):
0.0308 [0.0008] at batch 32, 1e-4; 0.0380 [0.0002] with the cache; 0.0347
[0.0007] at batch 4. By updates, at 40k: raw `jer_cal` 3.84 / 3.84 (batch
32, 1e-4), 4.07 / 3.98 (cache), 3.88 / 3.79 (batch 4), and 3.77 for the
base run (batch 32, 5e-5, one seed). Updates in 7 h: 41k at batch 32,
54k with the cache, 51k at batch 4.

- Batch 32 at 1e-4 leads early on the raw generator, by 0.14-0.17 GeV at
  2.5-3.5 h; batch 4 catches up by ~5 h. On the EMA, which inference
  uses, batch 4 is ahead throughout: it makes 25% more updates per hour,
  and the EMA sheds the initialization by updates, not by hours.
- The cached gradient penalty is worse throughout on the raw generator,
  most clearly in the per-tower error. On the EMA its 1.32x more updates
  put it ahead of batch 32 without it (less initialization left), but
  never ahead of batch 4.
- The seed-0 runs of batch 4 and of batch 32 at 1e-4 repeat those of job
  19988 (identical losses to 4-5 digits in the first epoch), and the
  scoring does not perturb the training.

### The published model

The pre-trained model of the paper (Zenodo record 17809156, 1.7 GB, md5
checked, in `OUTDIR/sphenix/pretrained/`) was trained as configured:
batch 4 at 5e-5, 400 x 2000 = 800k updates with a 64k-update warm-up; its
history spans 5 days 3.5 hours at 557 ms per update (~105 h on an A6000
at the 470 ms measured here). Scored the same way:

| | `l1_sig` | `l1_bkg` | `jes` | `jer_cal` |
| :--- | ---: | ---: | ---: | ---: |
| published, 800k updates, ema | 0.0334 | 0.0735 | 0.94 | 3.59 |
| published, 800k updates, raw | 0.0335 | 0.0530 | 0.90 | 3.59 |
| base run, 80k updates, ema | 0.0329 | 0.0438 | 0.96 | 3.63 |
| base run, 100k updates, ema | 0.0352 | 0.0442 | 0.96 | 3.63 |
| base run, 100k updates, raw | 0.0304 | 0.0498 | 0.97 | 3.60 |
| median-rho | 0.2822 | 0.2822 | 1.14 | 5.28 |

The base run matches the published jet resolution to within 1% after
80k updates (13.7 training hours), with per-tower errors as small or
smaller (`l1_bkg` of both networks, `l1_sig` of the raw one; the EMA's
`l1_sig` is 0.0329-0.0357 against 0.0334).
The published run's own history agrees: its `idt_aa_a1` was lowest around
120k updates (0.0314, rolling mean of 10 epochs) and ended at 0.0334,
while `cycle_b` kept falling to the end. What the remaining 700k updates
do buy is not visible here: the signal discriminator is fooled more and
more (`disc_a1` 0.02 -> 0.19, `gen_ba1` 1.19 -> 0.65), which may matter
for jet shapes or for the JEWEL test, neither of which is measured.

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
38% of the step, **but caching it costs quality** (job 20016: the
trained generator's held-out extraction is worse at every point of a 7 h
run), so leave it at 0.
`torch.compile` does not change the optimization and helps less at batch
32 because the GPU is busy anyway. CUDA graphs (`mode='reduce-overhead'`) would address the idle
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

- **Confirm the short run** before switching to it: batch 32 at 5e-5 with
  a short warm-up for ~100k updates, two more seeds (two GPUs for ~17 h:
  `DIAG_SEED=1 DIAG_LABEL=base_b32_lr5e-5_s1 sbatch
  scripts/slurm/diag_base_run.sbatch`, likewise seed 2; the base run is
  seed 0), scored with `eval_val_truth.sbatch`; and compare the
  short model with the published one on the JEWEL test split and on the
  physics the paper uses (jets found in the extracted image, shapes),
  which the cone energy here does not capture.
- **Job 19989** (base run, ceres, 26 h limit, ends ~2026-09-23 12:30) keeps
  writing a checkpoint every 10k updates; rerun `eval_val_truth.sbatch` on
  it to extend its table (scored rows are skipped).
- Whether 5e-5 beats 1e-4 at batch 32 at depth rests on one seed of 5e-5.
- The EMA carries the random initialization for tens of thousands of
  updates. A warm-up of its momentum (e.g. `min(m, (1 + t) / (10 + t))`)
  would make short runs usable at inference; it does not change the
  training. Not needed for runs of 100k updates.
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
numbers are solid. The recommendations now rest on held-out truth:
two seeds per setting for the 7 h comparison (seed spreads are given
with the numbers), but a single run each for the long base run and the
published model, on which the "train ~100k updates" advice rests. The
held-out scores are a proxy for the physics -- the energy in a fixed cone
around the true jet axis on PYTHIA-in-HIJING val events, not jets found in
the extracted image, and not the JEWEL test -- but they are measured
against truth on mixed events the model never saw.
