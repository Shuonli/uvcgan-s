# Flow matching against UVCGAN-S on the sPHENIX decomposition

Started 2026-09-24. Branch `ddp` of `github.com/Shuonli/uvcgan-s`. The
scaling notes (`SCALING_NOTES.md`) hold the baseline's training and its
held-out scores; this file holds the flow-matching comparison.

Questions: can a flow-based model reach the current UVCGAN-S physics
performance in fewer GPU-hours; is it more stable across seeds; how do final
quality and inference cost compare. Treated as hypotheses.

## Pre-registration (written before any flow model was trained)

### The task, as it actually is

The active configuration (`scripts/train/sphenix/train_uvcgan-s.py`) is
**mixture-to-components decomposition**, not equal-dimensional translation:

- towers: 24 (eta) x 64 (phi) calorimeter images in GeV, no noise;
- `gen_ba` maps a 1-channel mixture (PYTHIA jet embedded in central HIJING,
  domain `embed`) to 2 channels (background, signal); `gen_ab` maps them back;
- data norm `log`, bias 0.1: psi(E) = log(E + 0.1);
- training data, sampled independently (`merge_type` unpaired): 986k HIJING
  background events, 2.64M PYTHIA signal events, 633k mixed events. Val:
  200k mixed events; test: 833k JEWEL-in-HIJING mixed events.
- inference network: `ema_gen_ba` (the EMA of `gen_ba`), 1 forward pass.

### Audit of what the baseline learns from

- No hidden correspondence: the three domains are shuffled independently.
- **Synthetic pairs are already used.** The `idt-aa` term feeds
  `gen_ba` the sum `real_a0 + real_a1` of an independently drawn background
  and signal and penalises the L1 distance (GeV, weights 2.5 and 25) of its
  output to those two images: supervised training on synthetic mixtures,
  built from the known additive mixing. The `idt-bb` term embeds a real
  mixture as (mixture, 0) in the component domain.
- Real mixtures enter through `cyc-bab` (decompose, remix, L1 to the input)
  and the three adversarial losses.
- Held-out truth (the (file, event) pairing of the index files) is used
  only by `scripts/slurm/eval_val_truth.py`. Caveat carried over: the val
  events' signal images are among the unpaired training signals.

So a candidate may use synthetic mixtures b + s of independently drawn
events without using more information than the baseline; it may not use the
index files or any val/test event.

### Metric, tolerance and targets

Primary metric: `jer_cal`, the calibrated jet energy resolution on the fixed
20k val events (17,542 jets) of `eval_val_truth.py`. Also recorded:
`l1_sig`, `l1_bkg`, `jes`, `bias`. JEWEL (`--truth jewel`) is reserved
for the final evaluation of checkpoints selected on val.

Tolerances, measured before any candidate existed: statistical error of
`jer_cal` 0.028 GeV (bootstrap of the published model's jets); paired
difference of two similar models on the same events 0.006 GeV; seed spread
of the baseline (half range of three seeds) 0.01-0.07 GeV.

Reference values: published model (800k updates) 3.59 GeV; base run
(batch 32, 5e-5) 3.63 at 80k updates, 3.60 at 140k; median-rho subtraction
5.28; per-row oracle rho 5.09.

| target | val `jer_cal` | meaning |
| :--- | ---: | :--- |
| **T_acc** (primary) | **<= 3.70 GeV** | published + 3%: indistinguishable from the baseline at the level of its seed spread |
| T_match | <= 3.65 GeV | published + 2 statistical errors |
| T_useful | <= 4.00 GeV | a clear gain over the median-rho reference (5.28) |

Time to a target: the first evaluation at which a run meets it, **confirmed
by the next evaluation**; otherwise "not reached within budget". Time is
the training-loop time on one RTX A6000 (dahlia or ceres; the baseline's
hours are on dahlia), GPU-synchronised, excluding evaluation; end-to-end
time (with start-up, data preparation and evaluation) is reported
separately. The baseline's time to each target is computed from its
checkpoint scores and `history.csv` by the same rule, once jobs
20040-20044 have ended.

### Candidates

Common: psi(E) standardised per channel with a mean and deviation fitted on
training events only; a compact time-conditioned U-Net (TorchCFM's ADM
`UNetModel`, native 24 x 64, three down-samplings, attention at 6 x 16);
Adam, EMA of the weights; the same batch for all; the ODE integrated with a
fixed-step solver so the number of network evaluations (NFE) is exact.

- **A. OT-CFM, unpaired, augmented state.** State x = (background, signal),
  2 x 24 x 64. Source x0 = (psi(m), psi(0)): a real mixture with all its
  energy in the background channel and an empty signal channel (the
  baseline's own embedding of a mixture, c.f. `idt-bb`). Target x1 =
  (psi(b), psi(s)), b and s drawn independently. Exact minibatch OT between
  independently drawn source and target batches (squared Euclidean in the
  standardised state; TorchCFM `ExactOptimalTransportConditionalFlowMatcher`,
  sigma 0); velocity v(t, x), no conditioning. Decoding: integrate from
  x0 of the test mixture; signal = channel 1 (direct), and as a second
  reading signal = mixture - channel 0 (additivity imposed at decoding).
  Differs from the published method: the source is a degenerate
  augmentation of a 1-channel distribution, and the coupling pairs
  mixtures with independently drawn component pairs.
- **B. SB-CFM, unpaired, augmented state.** Same endpoints and network.
  Entropic OT plan with reg = 2 sigma^2 (Sinkhorn, log domain), Brownian
  bridge interpolant and SB conditional velocity (TorchCFM
  `SchrodingerBridgeConditionalFlowMatcher`; the published entropic
  coupling rather than the library's default exact plan); sigma 1 in
  standardised units. Velocity only, sampled with the probability-flow
  ODE: this is SB-CFM, not SF2M (no score model, no SDE).
- **C. Conditional CFM on synthetic mixtures** (the follow-up if A and B
  fail on correspondence). x1 = (psi(b), psi(s)), x0 ~ N(0, I), condition
  psi(b + s) concatenated to the network input; the pairing is exact by
  construction, the coupling independent (I-CFM). Uses the additive mixing,
  as the baseline's `idt-aa` does. At test the condition is the real
  mixture. Stochastic: scored per single sample and as the mean of K
  samples, with the cost of each.
- **D. Regression control for C**: the same network, trained to output
  (psi(b), psi(s)) from psi(b + s) by mean squared error. Same information as
  C, no flow; a 1-step Euler solve of C at t = 0 would reproduce it.

### Budget

Pilot: one seed, at most 30 GPU-minutes of training per configuration, at
most two GPU-hours in total including diagnostics and evaluation, on A6000s.
Before the pilot: shapes, finite gradients, coupling behaviour (does the
minibatch coupling pair a mixture with a signal whose jet is where the
mixture's jet is -- scored with held-out truth, never trained on), and a
checkpoint save/load round trip. Configurations that look promising are
extended to longer runs and three seeds, capped by the baseline's own time
to T_acc (~10-15 GPU-hours per seed): a candidate that needs longer is not
faster.

## Implementation

All code is in `scripts/flow/`; nothing in `uvcgan_s/` or in the
evaluator changed.

| file | purpose |
| :--- | :--- |
| `fm_common.py` | data, normalisation, network, the four methods, ODE solver, `Decomposer` (a flow as a generator for `eval_val_truth.score_generator`) |
| `fm_train.py` | training, resumable, with a budget in minutes of training time |
| `fm_eval.py` | scores checkpoints with the baseline's evaluator; NFE / solver / decoding / sample-count sweeps; `--latency` |
| `coupling_diag.py` | does a minibatch plan pair a mixture with its own signal (scored with held-out truth) |
| `smoke_checks.py` | pre-pilot checks |
| `make_cache.py` | one-off flat copy of the training h5 files |
| `fm_compare.py` | time-to-target table and figure, flows against the baseline |
| `*.sbatch` | SLURM wrappers: `fm_smoke`, `fm_run` (train + score), `make_cache` |

**Dependencies.** Not in the `fm4npp` env; installed without their
dependencies into `~/pyext/flow`, so the shared env is unchanged:

    python -m pip install --no-deps --target ~/pyext/flow \
        torchcfm==1.0.7 POT==0.9.7.post1
    export PYTHONPATH=~/pyext/flow:$PWD:$PWD/scripts/flow

torchcfm 1.0.7 is upstream tag `1.0.7` of
github.com/atong01/conditional-flow-matching (commit
`3fd278f9ef2f02e17e107e5769130b6cb44803e2`). The rest is the env: torch
2.7.1+cu118, numpy 2.3.1, python 3.12.11.

**Reused from TorchCFM:** `ExactOptimalTransportConditionalFlowMatcher`,
`SchrodingerBridgeConditionalFlowMatcher`, `ConditionalFlowMatcher` (time
sampling, interpolants, conditional velocities), `OTPlanSampler` (exact
plan; drawing pairs from a plan with replacement, as the library does), and
`UNetModel` (the ADM U-Net of the CIFAR-10 experiments: 96 channels,
multipliers 1-2-2-2, two residual blocks per level, attention at 6 x 16 and
in the middle block, scale-shift norm; 21.6M parameters). The plan is
solved outside the matcher's own call only so that it can be timed; the
arithmetic is the library's.

**Adapted: the entropic plan.** The library's SB-CFM with
`ot_method = 'sinkhorn'` calls numpy `ot.sinkhorn`, which underflows at
reg = 2 sigma^2 = 2 against costs of 3000-7500 (standardised state of 3072
numbers) and falls back to the independent plan with only a warning. POT's
log-domain solver is correct but takes 0.35 s per batch of 256 (2000
sweeps of tiny kernels). `fm_common.GraphedSinkhorn` solves the same
problem (log-domain Sinkhorn in dual potentials, float32, sweeps replayed
as a CUDA graph) to an L1 row-marginal error of 1e-3; its plan differs
from POT's converged plan by 5e-4 in L1. At sigma 1 the plan is nearly a
permutation (1.6 partners per row on average), so SB-CFM here differs
from OT-CFM mostly by its Brownian-bridge noise.

**Data path.** The training h5 files hold one lzf chunk per event. On this
file system that caps reads at 50-150 events/s at random (300-1300 on a
node whose page cache holds the files) and 5-55k/s for contiguous slices.
The baseline's loader never notices (it needs < 200 events/s); a flow
model at batch 256 needs thousands. `make_cache.py` copies the three
training domains once (9 min, 12.8 GB of float16, lossless, checked
against the source) into flat `.npy` files, which read at 1 GB/s cold and
7 GB/s from the page cache. Each run holds them in GPU memory and draws
i.i.d. indices per domain for every batch (0.1 ms per batch of 3 x 256).
Loading counts as start-up, not training time.

**Cost of a step** (one A6000, batch 256, fp32 with the default TF32
convolutions, like the baseline): 385 ms for every method (the network
dominates); the exact OT plan adds 8 ms (CPU network simplex), the
entropic plan ~0.1 s. Width against step time: 64 channels 9.6M
parameters 226 ms, 96 channels 21.6M 385 ms, 128 channels 38.3M 523 ms.

## Results

### Pre-pilot checks (jobs 20058, 20061)

`smoke_checks.py` passed for every method: batch shapes (256 x 24 x 64 per
domain, energies >= 0), standardised state and condition finite (unit
variance), every parameter receives a finite gradient, the exact plan is a
permutation, the entropic plan has unit mass and matches POT's converged
plan (L1 5.5e-3), the solver makes exactly NFE network evaluations, a
checkpoint round trip reproduces the network bit for bit. Each method was
trained 40 steps, resumed to 80 (resume restores weights, EMA, optimizer and
RNG states) and scored through the baseline's evaluator with every solver,
decoding and sample-count option. Normalisation (fitted on 20k training
events of each domain): psi of background -0.26 +- 0.70, of signal
-2.19 +- 0.44, of synthetic mixtures -0.22 +- 0.71.

Evaluation cost: 0.5 ms per event per network evaluation (batch 500), so
scoring the 20k val events at 16 NFE takes 160 s per checkpoint and
network, against ~5 s for one forward pass of the UVCGAN-S generator.

### Coupling behaviour: the minibatch plans carry no correspondence

`coupling_diag.py` (40 s on one A6000; `OUTDIR/sphenix/flow/coupling_diag.csv`).
Real val mixtures m are paired by each plan with independently drawn
training (background, signal) pairs; the pairing is scored against the
mixtures' held-out truth. `hit`: the paired signal's leading jet lies
within dR < 0.4 of the mixture's true jet. `jer_cal (m - b)`: resolution
of reading the signal as the mixture minus the paired background. Means of
4 repeats (spread in brackets).

| batch | plan | hit, paired signal | `jer_cal` (m - b), GeV | plan time |
| ---: | :--- | ---: | ---: | ---: |
| 256 | random | 3.9% [1.2] | 12.5 [0.9] | - |
| 256 | exact OT (A) | 3.1% [1.6] | 8.4 [0.9] | 23 ms |
| 256 | entropic, sigma 1 (B) | 3.0% [1.6] | 8.4 [0.9] | 104 ms |
| 256 | exact OT, cost on b + s | 9.7% [2.2] | 8.3 [1.1] | 11 ms |
| 1024 | random | 4.6% [0.1] | 13.3 [0.7] | - |
| 1024 | exact OT (A) | 4.4% [0.8] | 8.0 [0.2] | 0.27 s |
| 1024 | entropic, sigma 1 (B) | 4.5% [0.8] | 8.0 [0.1] | 0.14 s |
| 1024 | exact OT, cost on b + s | 10.8% [0.7] | 8.2 [0.1] | 0.19 s |
| 2048 | random | 4.0% [0.4] | 12.8 [0.8] | - |
| 2048 | exact OT (A) | 4.1% [0.4] | 7.8 [0.2] | 1.2 s |
| 2048 | entropic, sigma 1 (B) | 4.1% [0.3] | 7.7 [0.1] | 0.48 s |
| 2048 | exact OT, cost on b + s | 12.0% [0.5] | 7.9 [0.2] | 1.1 s |

median-rho on the same val events: 5.28 GeV.

- **The OT and entropic plans pair a mixture with a signal whose jet is
  where the mixture's jet is no more often than random pairing does**
  (3-4.5%, the cone's share of the acceptance), and a batch 8 times
  larger does not change that. The cost is dominated by the 1536
  background towers; the ~50 towers of the jet hardly move it. Even a cost
  that knows the mixing (distance of m to b + s) reaches only 10-12%.
- The plans do pick backgrounds of the right overall level (reading the
  signal as m - b improves from 12.5-13.3 GeV for random pairs to 7.7-8.4),
  but that is still far worse than median-rho (5.28), which uses the
  event's own background.
- At sigma 1 the entropic plan is nearly a permutation (1.6-1.7 partners
  per row) and scores like the exact one.

So the training pairs of A and B teach a map with no event-level jet
information: a flow can only average over couplings like these. This is
the correspondence failure the conditional follow-up (C) addresses; C and
D were launched on this evidence, before the A and B pilots finished.

## Running (keep current)

- Pilot (saturn, A6000, seed 0), `OUTDIR/sphenix/flow/pilot_<method>_s0/`:
  OT-CFM (job 20062) and SB-CFM (job 20063), 20 min of training;
  conditional CFM (job 20064), 25 min; regression (job 20065, done), 10 min.
  Each scores its checkpoints on the 20k val events when training ends.
- Extension of the regression (dahlia, jobs 20066-20068, from 21:04):
  seeds 0-2, 180 min of training each, checkpoint every 5 min,
  `OUTDIR/sphenix/flow/ext_regress_s{0,1,2}/`.
- Next: extend conditional CFM the same way when its pilot is in; mixture
  decoding of the A/B pilots; `fm_compare.py` against the baseline runs of
  jobs 20040-20044 (they end ~23:35 and need their final checkpoints
  scored, c.f. SCALING_NOTES.md); JEWEL, NFE curve and latency of the
  selected checkpoints.
