# Flow matching against UVCGAN-S on the sPHENIX decomposition

Started 2026-09-24. Branch `ddp` of `github.com/Shuonli/uvcgan-s`. The
scaling notes (`SCALING_NOTES.md`) hold the baseline's training and its
held-out scores; this file holds the flow-matching comparison.

Questions: can a flow-based model reach the current UVCGAN-S physics
performance in fewer GPU-hours; is it more stable across seeds; how do final
quality and inference cost compare. Treated as hypotheses.

## Answer (2026-09-25)

**Yes, in fewer GPU-hours -- by two different flows, each with a catch.
The published minibatch-OT recipes do not work as written for this task;
what works is an adaptation of each.** Everything below is on one RTX
A6000, val `jer_cal` of the 20k held-out mixtures (lower is better),
JEWEL at the val-selected checkpoint; the full tables are under "Results".

| | trained on | best val `jer_cal`, GeV | hours to 3.70 GeV (T_acc) | JEWEL `jer_cal` | per-tower `l1_sig` | inference, ms/event |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| UVCGAN-S batch 32, 3 seeds | unpaired + synthetic mixtures | 3.62 +- 0.03 (24 h) | 8.4 / 8.6 / 15.2 | 4.00-4.15 | 0.032-0.034 | 0.27 |
| UVCGAN-S batch 4, 3 seeds | same | 3.64 +- 0.02 (21-23 h) | 14.7 / 16.5 / 16.7 | 3.91-4.03 | 0.032-0.033 | 0.27 |
| UVCGAN-S published, 800k updates | same | 3.59 (~105 h) | | 3.99 | 0.033 | 0.27 |
| **conditional CFM**, mean of 16 samples, 3 seeds | synthetic mixtures (as `idt-aa`) | **3.60 / 3.60 / 3.60** (2.5-3 h) | **<= 1.0** (seed 0: 3.66 at 1 h; seeds 1-2 scored so only at the end) | 4.25-4.28 | 0.036 | 65 |
| conditional CFM, mean of 4 samples, 3 seeds | same | 3.89 / 3.90 / 3.90 (3 h) | not in 3 h | 4.51-4.54 | 0.037 | 16 |
| **OT-CFM**, unpaired, m - b, 4 Euler steps, 3 seeds | real mixtures, backgrounds, signals, unpaired | 3.67 / 3.68 / 3.68 (2-3 h) | **1.25 / 1.25 / 0.75** | **3.55 / 3.55 / 3.56** | 0.15 | 2.0 |
| SB-CFM, unpaired, m - b | same | 6.69 (20-min pilot) | - | - | 0.060 | 8.1 |
| regression, baseline's L1 loss, 3 seeds (control) | synthetic mixtures | 3.80 +- 0.01 (1 h) | not in 1 h (plateau) | 4.29-4.35 | 0.026 | 0.5 |
| regression, log-space MSE (control) | synthetic mixtures | 4.12 (1 h) | - | 4.17 | 0.032 | 0.5 |

1. **The published unpaired recipes fail on the signal they produce.**
   Minibatch OT (exact or entropic) between real mixtures and independent
   (background, signal) pairs pairs a mixture with a signal whose jet is
   where its own jet is no more often than chance (3-4.5%, batch 256-2048):
   the correspondence the flow would have to learn is not in its training
   pairs. OT-CFM's and SB-CFM's signal channel are useless.
2. **OT-CFM's background channel does learn it.** It becomes a
   least-change map from mixtures to backgrounds, i.e. it removes the jet;
   reading the signal as mixture minus background, with 4 coarse Euler
   steps (an averaging solve, chosen on val), it reaches T_acc after 45-75
   min (three seeds) and 3.67-3.68 GeV after 2-3 h, with no synthetic
   pairs and no mixing in training -- and it is **the best model on JEWEL
   by 0.35-0.45 GeV**, having no PYTHIA signal prior to be misled by. Its
   catch: a coarse image (`l1_sig` 4.5x the baseline's); only its cone
   energies are sharp. The converged solve (16 NFE) gives better towers
   (`l1_sig` 0.05) and a worse val resolution (4.08; JEWEL 3.60-3.66).
3. **Conditional CFM on synthetic mixtures is the best in distribution.**
   A single posterior sample is poor (5.0 GeV: it carries the posterior's
   spread), but the mean of K samples improves as sigma^2 = a + b / K:
   16 samples give 3.66 GeV after 1 h of training and 3.60 after 2.5-3 h
   for all three seeds -- the published model's resolution, 10x sooner
   than the baseline reaches 3.70. Its catches: 240x the baseline's inference cost (128 network
   evaluations per event, 65 ms) and a PYTHIA prior that costs it on
   JEWEL (4.27 against the baseline's 3.91-4.03).
4. **Much of the speed comes from supervised training on synthetic
   mixtures, which the baseline already contains.** A plain U-Net
   regression with the baseline's own `idt-aa` loss reaches 3.92 GeV in 15
   minutes and 3.80 +- 0.01 (3 seeds) in 40-60, with the smallest per-tower
   errors of all, at 0.5 ms/event -- but it plateaus above T_acc and is the
   worst on JEWEL. The flows' advantage over it is the posterior mean
   (conditional CFM) and the absence of a signal prior (OT-CFM).
5. **Seeds: yes, more stable.** Three seeds of each flow agree to
   +-0.01-0.02 GeV at every matched training time, on val and on JEWEL,
   and their curves are smooth and monotone. The baseline's final
   resolution is as reproducible (+-0.02-0.03), but its path is not: its
   time to T_acc spans 8.4-15.2 h at batch 32, its raw network moves by
   0.1-0.3 GeV between checkpoints, and its EMA is unusable for ~5 h.

**Recommendation.** Continue with flow matching, in two modified forms,
and keep UVCGAN-S (batch 4, 20-24 h, c.f. SCALING_NOTES.md) as the
reference until the flows pass the analysis the paper uses (jets found
in the extracted image), which the cone energy does not test:

- For robustness and cheap inference: **OT-CFM, unpaired, read as
  mixture minus background.** Next: improve its per-tower image (e.g. a
  1-channel embed -> background flow instead of the augmented state, whose
  signal channel is dead weight; or a per-tower readout between the coarse
  and the converged solve), and test it on the jet-level analysis.
- For the best in-distribution resolution per training hour:
  **conditional CFM on synthetic mixtures with a posterior-mean readout.**
  Next: cut the inference cost (distil the posterior mean into one
  network; fewer samples with coarse solves already give 3.84 GeV for 16
  evaluations), and address its JEWEL deficit (e.g. train with more
  varied signal shapes).
- Do not pursue SB-CFM here (worse than OT-CFM at equal time, its
  entropic plan costs 11% of the step and is still a near-permutation at
  the conventional sigma); UOT-FM (population imbalance) is not indicated
  -- the synthetic and real mixtures match to 0.1% in mean energy.

**How far to trust this.** One metric family (cone energy at the true
jet axis, per-tower L1), one dataset, one network size, single runs of 1-3
h per seed. The inference settings (the m - b reading, 4 Euler steps for
OT-CFM, the 4- and 16-sample means for conditional CFM) were chosen on the
val events after the pre-registration, which fixed only the metric, the
targets and the time rule; JEWEL, never used for a choice, confirms their
ranking. Time is the training loop on one A6000; start-up is seconds (data
from a one-off 9-min cache), while scoring the flows costs 10 s-15 min per
checkpoint on 20k events (160 s at 16 NFE), against ~5 s for the baseline.
Budget: the pilot took 3.0 GPU-h against the planned 2 (scoring and a rerun
of the checks); the whole study 33 GPU-h, against 156 for the six
baseline runs it compares with.

Files: `docs/flow/` has the tables (`compare_arms.csv`: per model;
`compare.csv`: per run; `compare_matched.csv`: seeds at matched times;
`compare_curves.csv`: every scored checkpoint; `latency.csv`;
`coupling_diag.csv`), the figures (`compare_report.png`: val `jer_cal` and
`l1_sig` against hours, JEWEL against val; `compare_nfe.png`: the solver
curves) and each run's config, timing summary and scores (`runs/`).

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

### Pilot (seed 0, one A6000 each, jobs 20062-20065 and 20073)

Batch 256, the same 21.6M-parameter network for all four, fp32. val
`jer_cal` of the EMA network on the 20k events, midpoint 16 NFE
(regression: one evaluation), at the training time given:

| method | 5 min | 10 min | 15 min | 20 min | 25 min | `jes` | `l1_sig` | steps/s | coupling |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A. OT-CFM, signal channel | (-336) | (-255) | (2835) | (120) | | 0.05 | 0.108 | 2.50 | 2.0% |
| A. OT-CFM, read as m - b | | | | **4.56** | | 0.67 | 0.058 | | |
| B. SB-CFM, signal channel | 40.8 | 32.2 | 29.0 | 71.5 | | 0.19-0.83 | 0.129 | 2.29 | 10.8% |
| B. SB-CFM, read as m - b | | | | 6.69 | | 0.48 | 0.060 | | |
| C. conditional CFM | 6.80 | 5.96 | 5.71 | 5.57 | 5.46 | 0.94 | 0.044 | 2.54 | - |
| D. regression | 4.95 | 4.49 | | | | 0.81 | 0.032 | 2.53 | - |

(raw network at the end: A signal channel meaningless, B 70.2, C 5.20,
D 4.72 GeV. D also at 2.6 min: 6.76 and 7.5 min: 4.63.) References:
median-rho 5.28; T_useful 4.00; T_acc 3.70. The baseline at these times
has an unusable EMA; its raw generator needs 2.7-4.5 h to reach 4.00.

- **A and B, read from their signal channel, fail** as the coupling
  diagnostic predicted: A's signal channel holds 5% of the true energy,
  B's energy scale wanders, neither correlates with the truth. Values in
  brackets are meaningless (slope of the response near zero).
- **A's background channel does learn the decomposition.** Read as mixture
  minus background, the unpaired OT-CFM scores 4.56 GeV after 20 minutes,
  better than median-rho, and better than its own coupling (7.8-8.4):
  the flow averages over couplings into a least-change map from mixtures
  to backgrounds, i.e. it removes the jet. No synthetic pairs and no
  mixing knowledge in training; additivity only at read-out.
- D improves fastest (6.76 -> 4.49 in 10 min, still falling); C is slower
  (6.80 -> 5.46 in 25 min, still falling). One sample of C carries the
  posterior's spread; its K-sample mean is the natural comparison with D.
- None reached T_useful within the pilot. **The pilot is inconclusive on
  time-to-quality**: every curve was still falling. A (read as m - b), C and
  D are extended.

Throughput is set by the network: 2.3-2.5 steps/s = 590-650 samples/s,
against 8.6-54 samples/s for the baseline (batch 4-32). The exact plan
costs 2% of A's time, the entropic plan 11% of B's. Peak GPU memory
41-43 GB, of which 11-13 GB are the resident training data. Start-up
(imports aside) 5 s: data from the page cache in 4 s.

**Pilot budget: exceeded.** ~3.0 GPU-hours against the 2 planned: training
75 min, scoring ~60 min (the 16-NFE scoring of the flows costs 160 s per
checkpoint and network, which the plan underestimated), pre-pilot checks
~35 min (run twice, after a resume bug was fixed), Sinkhorn checks ~5 min.

### The regression plateau is the loss, not the model (job 20076)

The log-space regression (D) stalls: its loss is flat from 10 min on and
its monitor score creeps from 4.69 (15 min) to 4.43 GeV (60 min; 1000-event
monitor, which reads ~0.4 GeV above the 20k score). It was stopped at 60
min. Squared error of standardised log energies weights the high towers
that make the jet energy least, and its optimum is a geometric-type mean.

Follow-up `regress_l1` (not pre-registered; motivated by that plateau):
the same network and data, trained with the baseline's own `idt-aa` loss,
L1 of the energies in GeV with background : signal weights 1 : 10. val, 20k
events, EMA network, seed 0:

| training time | 5 min | 10 | 15 | 20 | 30 | 40 | 50 | 60 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `jer_cal`, GeV | 4.81 | 4.03 | **3.92** | 3.86 | 3.82 | 3.80 | 3.81 | 3.79 |

At 60 min: `l1_sig` 0.0264, `l1_bkg` 0.0280, `jes` 0.86 (raw network:
`jer_cal` 3.89). T_useful (4.00) is reached at 15 min (confirmed at 20);
T_acc (3.70) is not reached in 60 min, the curve is flat from 40 min on
at 3.79-3.81. For comparison, the published model: `jer_cal` 3.59,
`l1_sig` 0.0334, `l1_bkg` 0.0735 (EMA); the baseline reaches 4.00 after
5-7 h (EMA) and 3.70 after 8-17 h.

So **plain supervised training on synthetic mixtures, with the loss the
baseline already uses for them, gets within 0.2 GeV (5%) of the
baseline's jet resolution in under an hour of one GPU, and beats it on
per-tower errors by 20-60%.** It is not a flow; it is the part of the
baseline that the flows here compete with. The remaining 0.2 GeV is what
the adversarial and cycle terms, or longer training, buy the baseline.

### Conditional CFM: the posterior mean reaches the baseline (jobs 20081, 20086)

C is a generative model of p(background, signal | mixture): one sample is
one plausible decomposition and carries the posterior's spread. For an
energy resolution the natural estimator is the posterior mean, estimated
by averaging the energies of K samples. At the 120 min checkpoint of
`ext_condcfm_s0` (EMA network, midpoint solver; 20k val events):

| samples K | 1 | 4 | 8 | 16 |
| :--- | ---: | ---: | ---: | ---: |
| `jer_cal`, 8 NFE | 5.03 | 3.92 | 3.72 | **3.62** |
| `jer_cal`, 16 NFE | 5.06 | 3.94 | | |
| `l1_sig` / `l1_bkg`, 8 NFE | 0.0414 / 0.0434 | 0.0375 / 0.0383 | 0.0366 / 0.0371 | 0.0361 / 0.0364 |
| network evaluations per event | 8 | 32 | 64 | 128 |

- 16 NFE is no better than 8: the paths are straight enough.
- The resolution follows sigma_K^2 = a + b / K (independent sample noise):
  K = 1 and 4 give a = 3.48^2 and b = 3.63^2, predicting 3.71 at K = 8
  and 3.60 at K = 16 (measured 3.72 and 3.62). The posterior mean itself
  (K -> infinity) resolves the jet energy to ~3.48 GeV, better than the
  published model (3.59) -- after two hours of training on one A6000.
- **With 16 samples conditional flow matching reaches T_acc and T_match
  (3.62 GeV) after at most 2 h of training; the baseline needs 8-17 h
  for 3.70.** The price is inference: 128 network evaluations of 0.5 ms,
  ~65 ms per event against 0.27 ms for the UVCGAN-S generator (~240x).
- The regressions are point estimates of per-tower medians or log-means;
  the flow's sample mean in energy is the minimum-variance estimate of a
  cone energy, which is why the flow overtakes the L1 regression (3.79)
  once enough samples are averaged.

Checkpoints of C are therefore selected, and its time curves drawn, on
the mean of 4 samples at 8 NFE (`fm_common.SELECTION`, 32 evaluations per
event), with K = 16 predicted from K = 1 and 4 and measured directly at
some checkpoints.

### Regressions: seeds, JEWEL (job 20085)

Log-space regression D, 20k val events, EMA: seed 0 4.90 (5 min), 4.49
(10), 4.34 (15), 4.25 (20), 4.18 (30), 4.12 (60); seeds 1 and 2 at 15 min
4.31 and 4.27 (seed spread +-0.04 at matched time). JEWEL at the
val-selected checkpoint: D 4.17 (seed 0, 55 min), the 10-min pilot 4.30,
**`regress_l1` 4.31** -- worse than its val score (3.79) by more than any
other model's val-to-JEWEL step, and worse than the baseline at batch 4
(3.91-4.03) though its per-tower error on JEWEL is the smallest of all
(`l1_sig` 0.019). The L1 regression has learned a PYTHIA prior that the
quenched JEWEL jets do not follow.

### Inference cost (`OUTDIR/sphenix/flow/latency.csv`, A6000, batch 500)

| model | parameters | ms per event |
| :--- | ---: | ---: |
| UVCGAN-S published generator (`ema_gen_ba`) | 32.1M | 0.27 |
| regression (D, `regress_l1`) | 21.6M | 0.50 |
| flow, one network evaluation | 21.6M | 0.50 |
| OT-CFM / conditional CFM, 8 NFE | | 4.1 |
| conditional CFM, 4 samples x 8 NFE | | 16.3 |
| conditional CFM, 16 samples x 8 NFE | | 65 |

Linear in the evaluations (0.505 ms each); the solver type does not
matter at equal NFE.

### Unpaired OT-CFM: few Euler steps reach the target (jobs 20075, 20095, 20097)

Seed 0 was continued from the 20-min pilot to 180 min. Read as mixture
minus background and solved to convergence (midpoint, 16 NFE) it
improves from 4.55 GeV (5 min) to a best of 4.08 (105 min), then drifts
to 4.14 (180 min); `l1_sig` 0.051-0.055, `jes` 0.69-0.70. The solver
sweep at the 105-min checkpoint (val `jer_cal`, GeV; `l1_sig` in brackets):

| NFE | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Euler | 5.10 (0.42) | 3.99 (0.24) | **3.68** (0.15) | 3.73 (0.10) | 3.84 (0.074) | | |
| midpoint | | | 4.26 (0.057) | 4.15 (0.053) | 4.08 (0.048) | 4.11 (0.052) | 4.13 (0.051) |

A few coarse Euler steps follow the flow's mean direction (at t = 0 the
velocity is the average displacement over all the backgrounds the plan
pairs a mixture with), so the background they reach is an average rather
than one sharp sample: better cone energies, much worse towers. It is the
same mean-against-sample trade as conditional CFM's, obtained here in
four network evaluations. **4 Euler steps are the selection setting of
OT-CFM from here on** (chosen on val; `fm_common.SELECTION`). Its time
curve at that setting:

| training time | 5 min | 10 | 15 | 20 | 30 | 45 | 60 | 75 | 90 | 120 | 150 | 180 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val `jer_cal` | 4.14 | 4.11 | 4.04 | 3.96 | 3.78 | 3.71 | 3.71 | **3.70** | 3.69 | 3.68 | 3.67 | 3.67 |

T_useful at 20 min (confirmed at 30), **T_acc at 75 min (confirmed at
90)**, T_match not reached in 3 h. No synthetic pairs, no mixing in
training: the additivity enters only at read-out. On JEWEL (the
out-of-distribution test) at the 105-min checkpoint: **3.56 GeV with 4
Euler steps, 3.63 with the converged solve** -- better than every
UVCGAN-S run (batch 4: 3.91-4.03, published 3.99) and better than its own
val score. It has no signal prior to be wrong about: it removes what does
not look like background. SB-CFM does not share this (read as m - b with
1-4 Euler steps: 12.5 / 11.0 / 9.2 GeV at its 20-min pilot checkpoint).

Conditional CFM with coarse solves, at its 180-min checkpoint (4 samples;
total evaluations per event in brackets): midpoint 4 NFE 3.84 (16), 8 NFE
3.89 (32), 16 NFE 3.91 (64); 16 samples at 8 NFE 3.60 (128). A single
sample with 1-2 Euler steps (the posterior mean in log space, like the
regressions): 4.18 / 4.02 (120 min).

`regress_l1` with a cosine-decaying rate (job 20091): 3.82 GeV after 60
min, level with the constant rate (3.79-3.81): its plateau is the
estimator, not the schedule.

### JEWEL: the out-of-distribution test (jobs 20079, 20094, 20096, 20098)

JEWEL (quenched jets in HIJING, never seen in training) at each run's
val-selected checkpoint, EMA network, `jer_cal` in GeV:

| model | selection setting | val | JEWEL |
| :--- | :--- | ---: | ---: |
| UVCGAN-S batch 32 (3 seeds) | | 3.60 / 3.60 / 3.65 | 4.12 / 4.15 / 4.00 |
| UVCGAN-S batch 4 (3 seeds) | | 3.63 / 3.62 / 3.66 | 3.91 / 3.99 / 4.03 |
| UVCGAN-S published, 800k updates | | 3.59 | 3.99 |
| median-rho | | 5.28 | 5.34 |
| **A. OT-CFM, unpaired, m - b** (180 min) | 4 Euler steps | 3.67 | **3.55** |
| same checkpoint, other solves | Euler 16 / midpoint 4 / midpoint 16 | | 3.52 / 4.03 / 3.66 |
| C. conditional CFM (180 min) | mean of 4, 8 NFE | 3.89 | 4.53 |
| same, mean of 16 | 8 NFE | 3.60 | 4.27 |
| same, one sample | 8 NFE | 4.99 | 5.50 |
| `regress_l1` (3 seeds, 45-60 min) | | 3.79 / 3.81 / 3.80 | 4.31 / 4.35 / 4.29 |
| D. regression, log space (60 min) | | 4.12 | 4.17 |

- The models that learn a signal prior from PYTHIA (conditional CFM, both
  regressions, and UVCGAN-S through `idt-aa`) lose 0.3-0.7 GeV on the
  quenched JEWEL jets; the more supervised, the more they lose (the L1
  regression and conditional CFM, which fit PYTHIA best, lose most).
- **The unpaired OT-CFM gains**: it only learns what background looks
  like and reads the signal off as the remainder, so the jet's shape
  cannot mislead it. It is the best model on JEWEL by 0.35-0.45 GeV.
- Conditional CFM's 16-sample mean, the best model on val, is 0.28 GeV
  behind the baseline on JEWEL.

### Seeds (jobs 20077-20100, 20104-20106)

val `jer_cal` of the EMA network at the selection setting, per seed
(`OUTDIR/sphenix/flow/compare_matched.csv` has mean and half range):

| model | 15 min | 30 min | 1 h | 2 h | 3 h | time to T_acc (3.70) | JEWEL |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| OT-CFM, m - b, 4 Euler steps | 4.04 / 4.03 / 4.06 | 3.78 / 3.76 / 3.73 | 3.71 / 3.70 / 3.69 | 3.68 / 3.68 / 3.68 | 3.67 / - / - | 75 / 75 / 45 min | 3.55 / 3.55 / 3.56 |
| conditional CFM, mean of 4 | 4.28 / - / - | 4.07 / 4.10 / 4.08 | 3.97 / 3.96 / 3.97 | 3.92 / 3.92 / 3.93 | 3.89 / 3.90 / 3.90 | not in 3 h | 4.53 / 4.51 / 4.54 |
| conditional CFM, mean of 16 | | | 3.66 / - / - | 3.62 / - / - | 3.60 / 3.60 / 3.60 | <= 60 min (seed 0) | 4.27 / 4.25 / 4.28 |
| `regress_l1` | 3.92 / 3.93 / 3.92 | 3.82 / 3.83 / 3.85 | 3.79 / 3.81 / 3.81 | | | not in 1 h | 4.31 / 4.35 / 4.29 |
| regression, log space | 4.34 / 4.31 / 4.27 | 4.18 / - / - | 4.12 / - / - | | | - | 4.17 / - / - |
| UVCGAN-S batch 32 | EMA unusable | | | 8.3 +- 0.2 (EMA) | 6.2 +- 0.3 | 8.4 / 8.6 / 15.2 h | 4.12 / 4.15 / 4.00 |
| UVCGAN-S batch 4 | | | | 7.4 +- 0.7 | 5.3 +- 0.6 | 14.7 / 16.5 / 16.7 h | 3.91 / 3.99 / 4.03 |

(OT-CFM seeds 1-2 ran 2 h, `regress_l1` 1 h, the log-space regression's
seeds 1-2 15 min; seed 0 of conditional CFM was scored every 15 min,
seeds 1-2 every 30 min.) The baseline EMA at 8 / 16 / 24 h: 3.74 / 3.65 /
3.63 (batch 32), 3.78 / 3.69 / 3.65 (batch 4), seed spreads 0.00-0.05.

- **Every flow and regression is reproducible to +-0.01-0.03 GeV at
  matched training time**, and its curve is smooth and monotone. The
  baseline's final resolution is as reproducible (+-0.02-0.03), but its
  path is not: its time to T_acc spans 8.4-15.2 h at batch 32, its raw
  network moves by 0.1-0.3 GeV between checkpoints, and its EMA is
  unusable for the first ~5 h.
- OT-CFM's three seeds sit at 3.69-3.70 from 45 min on, right at T_acc,
  and settle at 3.68 by 2 h; the confirmation rule places the crossing at
  45-75 min. Their JEWEL scores agree to 0.01 GeV.

### A clean image from the unpaired OT-CFM, without retraining (2026-09-25)

`scripts/flow/readout_test.py` (tables `docs/flow/readout_final.csv`,
event display `docs/flow/readout_events_val.png`). The 4-step read-out's
noise is almost all away from the jet: summed |error| of the signal image
outside the true leading-jet cone is 216 GeV per event (val), against 40
for UVCGAN-S and 42 for an empty image (the other PYTHIA particles). The
converged solve is clean (61 GeV/event) but loses 30% of the jet (`jes`
0.69, `jer_cal` 4.1).

Read-outs from the same trained flow (none uses the truth; thresholds
first chosen on val, the seed threshold then set below the analysis'
10 GeV jet threshold after JEWEL showed why, see below):

- **tower threshold**: keep m - b only where it exceeds t GeV. It cleans
  the image *and* improves the jet energy (val 3.665 -> 3.604 at 0.5 GeV):
  the dropped towers are mostly noise, and the lost soft jet constituents
  only lower the scale (`jes` 0.77), which the calibration absorbs.
- **jet-seeded subtraction** (as the iterative subtraction of heavy-ion
  jet analyses): towers whose coarse cone energy (R = 0.4) exceeds E GeV
  seed a region of all towers within R = 0.4; inside it the coarse
  subtraction (thresholded), outside it nothing. It leaves the jet energy
  exactly as it was, as long as the region covers the jet.

Three seeds each (seed 0 at 180 min, seeds 1-2 at 120 min):

| read-out | val `jer_cal` | JEWEL `jer_cal` | `l1_sig` val / JEWEL | off-jet error, GeV/event, val / JEWEL | jets in region, val / JEWEL | `jes` val / JEWEL |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 Euler steps, m - b (before) | 3.67-3.68 | 3.55-3.56 | 0.150 / 0.150 | 216-221 / 216-221 | | 0.93 / 0.98 |
| converged, m - b | 4.06-4.14 | 3.60-3.66 | 0.051-0.054 / 0.047-0.050 | 61-67 / 58-64 | | 0.69-0.70 / 0.69-0.71 |
| threshold 0.5 GeV | 3.60-3.62 | 3.48-3.49 | 0.062-0.064 / 0.057-0.058 | 82-84 / 76-78 | | 0.77 / 0.77 |
| seeds 8 GeV, threshold 0.5 | 3.60-3.62 | 3.48-3.50 | 0.051-0.052 / 0.045-0.047 | 65-67 / 59-61 | 100% / 99.9% | 0.77 / 0.77 |
| **seeds 10 GeV, threshold 0.7** | 3.64-3.65 | 3.48-3.49 | **0.037-0.038 / 0.030-0.031** | **44-45 / 35-36** | 100% / 99.3-99.4% | 0.71 / 0.70 |
| seeds 12 GeV, threshold 0.5 | 3.60-3.62 | 3.62-3.63 | 0.037-0.038 / 0.029-0.030 | 44-45 / 34-35 | 100% / 97.5-97.8% | 0.77 / 0.76 |
| UVCGAN-S published | 3.59 | 3.99 | 0.033 / 0.028 | 40 / 34 | | 0.94 / 0.95 |
| L1 regression | 3.79 | 4.31 | 0.026 / 0.019 | 30 / 20 | | 0.86 / 0.87 |

- **With a jet-seeded, thresholded read-out the unpaired OT-CFM (2-3 h of
  training) has an image about as clean as UVCGAN-S's, the same jet
  resolution on val and a better one on JEWEL by ~0.5 GeV.**
- The seed threshold is a physics setting, not a tuning knob: it must sit
  below the lowest jet energy kept (10 GeV here) times the read-out's scale.
  12 GeV, the first val choice, covers every PYTHIA jet but misses 2.4% of
  the softer quenched JEWEL jets, and each missed jet reads as zero energy
  (JEWEL 3.62 instead of 3.48). 8-10 GeV covers 99.3-99.9%.
- Costs: the thresholds lower the jet energy scale to 0.70-0.77 (UVCGAN-S
  0.94), so it has to be calibrated; the scale moves by <= 1.5% between
  PYTHIA and JEWEL (the 4-step read-out's by 5%). Inside the jet cone the
  per-tower error is ~20% above UVCGAN-S's (val 0.254 against 0.209):
  the cone energy is right, the pattern of towers inside the jet less so --
  which matters for jet shapes, not for jet energies.

### Why OT-CFM's signal channel carries nothing (checked 2026-09-25)

Training is unpaired: each batch draws 256 mixtures, 256 HIJING events and
256 PYTHIA events independently; the index files that record which events
make up a mixture are read only by the evaluator. The only pairing is the
OT plan inside a batch. Its cost splits into a background part, |psi(m) -
psi(b)|^2, and a signal part, |psi(0) - psi(s)|^2. The signal part does not
depend on the mixture (the start's signal channel is the same empty image
for every mixture): it is constant down each column of the cost matrix
(to 1e-14 relative, 8 batches of 256), and a constant per column cannot
change a one-to-one assignment. **The plan with and without the signal
channel is the same permutation in 8 of 8 batches**, although the signal
part is large (mean 1675 against 2887 for the background part). So each
mixture is paired with a background resembling it and with whichever
PYTHIA event was drawn next to that background (jet at the right place
3-4.5% of the time, as for random pairs). Flow matching then regresses the
signal channel onto random jets: it learns their average, the same for
every mixture, and ends at 4-5% of the true cone energy, uncorrelated with
the event. The background channel's targets were chosen to resemble the
mixture, so it learns "the nearest background", which removes the jet.

### Jet substructure (`scripts/flow/substructure.py`, 2026-09-25)

R = 0.4 cone around the true leading-jet axis, towers as constituents
(transverse energy = tower value, negative towers dropped), 10k events of
val and of JEWEL (8.8k / 8.4k jets). Observables: cone E_T, mass, girth,
p_T^D, core fraction, leading-tower fraction, soft drop z_g and R_g
(C/A, z_cut 0.1, beta 0). Tables: `docs/flow/substructure.csv`
(W1 / sigma and per-jet resolution per model and observable),
`substructure_modification.csv`; figure `docs/flow/substructure.png`.

- **Seeding changes nothing inside the jet** (identical numbers), unless
  the seeded region misses or cuts a jet (JEWEL at 10 GeV seeds: per-jet
  girth resolution 0.31 -> 0.44).
- **The threshold redefines the observables** (drops constituents below
  0.5-0.7 GeV). Against truth with the same threshold -- what a
  detector-level analysis would measure -- the thresholded OT-CFM
  distributions match closely: W1 / sigma 0.01 (girth), 0.02 (p_T^D), 0.02
  (core), 0.03 (z_lead), 0.05 (z_g), 0.08 (R_g) on val, 0.04-0.12 on JEWEL;
  UVCGAN-S against the full truth: 0.06-0.23. Per-jet resolutions are
  UVCGAN-S's (girth 0.27, p_T^D 0.37 of the true spread on val).
- The plain 4-step read-out's noise broadens the jets (girth up, R_g
  pushed to large angles).
- Nobody measures z_g or R_g jet by jet at this granularity and background
  (per-jet resolution 0.8-1.2 of the true spread), nor the mass (~0.8);
  their distributions and means are measurable.
- Conditional CFM: single samples reproduce the distributions (like
  UVCGAN-S) but are noisy per jet; the 16-sample mean is best per jet but
  too narrow in places (the average of plausible jets is smoother than a
  jet).

The quenching signal: JEWEL minus PYTHIA of the mean, jets of true cone
energy 20-30 GeV (3.1k each). In truth JEWEL jets are narrower and harder:
p_T^D +0.049 +- 0.002, z_lead +0.057 +- 0.003, R_g -0.027 +- 0.003, mass
-0.69 +- 0.02 GeV. Fraction of it each extraction keeps (+- 0.05-0.10):

| | p_T^D | z_lead | R_g | mass |
| :--- | ---: | ---: | ---: | ---: |
| OT-CFM, 4 Euler steps, m - b | 0.63 | 0.68 | 0.23 | 0.38 |
| OT-CFM, seeds 8 GeV + 0.5 GeV threshold | 0.82 | 0.86 | 0.68 | 0.40 |
| OT-CFM, seeds 10 GeV + 0.7 GeV threshold | 0.88 | 0.93 | 0.79 | 0.39 |
| OT-CFM, converged | 0.77 | 0.79 | 0.65 | 0.38 |
| UVCGAN-S published | 0.81 | 0.84 | 0.74 | 0.64 |
| conditional CFM, mean of 16 / 1 sample | 0.73 / 0.70 | 0.75 / 0.75 | 0.77 / 0.62 | 0.49 / 0.50 |
| L1 regression | 0.77 | 0.81 | 0.79 | 0.50 |
| truth with 0.5 / 0.7 GeV threshold | 1.04 / 1.05 | 1.06 / 1.08 | 1.08 / 1.05 | 0.88 / 0.84 |

The threshold is what makes OT-CFM's substructure usable: the plain
read-out's noise, identical for PYTHIA and JEWEL, dilutes the difference;
thresholded, OT-CFM keeps as much of it as UVCGAN-S or more, except the
mass (40% against 64%: partly the lower energy scale, mass scales with it,
partly wide-angle residuals). Every extraction dilutes the quenching
signal (keeps 64-93% in the shapes): an analysis must correct for it.

### Vacuum -> medium: what an unpaired OT coupling would pair (`vac_med_coupling.py`)

Exact minibatch OT between PYTHIA val truth jets (median cone E_T 32 GeV)
and JEWEL truth jets (26 GeV), 4 batches each:

| representation | batch | axis distance of pairs | E_T rank corr. | girth corr. |
| :--- | ---: | ---: | ---: | ---: |
| random pairs | 1024 | 1.69 | -0.01 | 0.01 |
| full 24 x 64 images, log | 256 / 1024 | 1.45 / 1.37 | 0.07 / 0.11 | 0.01 / 0.00 |
| jet-centred 9 x 9, log | 256 / 1024 | | 0.13 / 0.21 | 0.70 / 0.78 |
| jet-centred 9 x 9, GeV | 256 / 1024 | | 0.44 / 0.56 | 0.78 / 0.85 |

On full images OT pairs jets by where they are; on centred log images by
shape, hardly by energy; in GeV partly by energy. The monotone (quantile)
coupling of cone energies, which larger batches approach in 1-D, maps
23 -> 13, 30 -> 22, 39 -> 37, 44 -> 44 GeV. The "modification" an
unpaired OT-CFM learns is the least change in whatever metric is chosen --
a modelling assumption that unpaired data cannot test.

### Other ways to set up the matching (`coupling_variants.py`, 2026-09-25)

Question: start the right panel from the mixture instead of empty, or
train two separate one-panel models (mixture -> HIJING, mixture ->
PYTHIA)? Matching only, scored with the val truth (4 batches each):

| set-up | matched jet at the right place, batch 256 / 1024 | its energy tracks the true jet (correlation) |
| :--- | ---: | ---: |
| random pairs | 4% / 4% | 0.00 |
| empty start (as trained), log or GeV | 4% / 4-5% | -0.06-0.09 |
| mixture start, (HIJING, PYTHIA) bundled, log | 28% / 35% | 0.09-0.11 |
| mixture start, bundled, GeV | 29% / 39% | 0.21-0.25 |
| mixture -> PYTHIA alone, log | 34% / 41% | 0.12-0.13 |
| mixture -> PYTHIA alone, GeV | 41% / 52% | 0.19-0.23 |
| mixture -> HIJING, jet = mixture - background, log | 69% / 69% | 0.64-0.65 |
| mixture -> HIJING, jet = mixture - background, GeV | 66-67% / 66-67% | 0.53-0.58 |

Starting the right panel from the mixture lets the matching look for a
jet event whose jet sits on the mixture's bright spot: 30-50% right
places instead of 4%, more in GeV (where the jet towers are the biggest
numbers) and with more candidates. But the matched jet is somebody else's
jet that happens to sit there: its energy hardly follows the true one
(0.1-0.25), and reading its energy gives 25-120 GeV resolution. Reading
the jet as mixture minus the matched background keeps the jet's own energy
(0.53-0.65, the right place by construction). A mixture -> PYTHIA flow
would learn "a typical jet where the mixture has an excess"; the
background route is the one that carries the energy.

### With the true pieces in the batch (`pieces_in_batch.py`, 2026-09-25)

Question: build each batch so that its HIJING and PYTHIA events are the
pieces of its mixtures (shuffled)? With the trained OT-CFM's cost, every
mixture is matched to its own pieces: **100% of the time, for mixtures
made by adding the pieces and for real val mixtures with their true
pieces, batches of 256 and 1024** (4 each). The true background differs
from the mixture only in the jet's ~50 towers, so it is by far the most
similar candidate. The matching then only undoes the shuffle: this is
supervised training on synthetic mixtures (the information of UVCGAN-S's
`idt-aa`, conditional CFM and the regressions), with that family's
behaviour -- best on PYTHIA val (3.60-3.79 GeV), worst on JEWEL
(4.25-4.35) -- and for real mixtures it needs the pieces, which only the
simulation's bookkeeping has.

## Commands

From the repository root, on the a6k partition (A6000 nodes for anything
timed). One-off setup:

    python -m pip install --no-deps --target ~/pyext/flow \
        torchcfm==1.0.7 POT==0.9.7.post1
    sbatch scripts/flow/make_cache.sbatch            # flat .npy copy, ~9 min
    sbatch -w saturn scripts/flow/fm_smoke.sbatch    # checks + coupling diagnostic

Train and score one run (`fm_run.sbatch` = `fm_train.py` + `fm_eval.py`;
rerunning the same command with a larger MINUTES resumes the run from its
`resume.pt`, weights, EMA, optimizer and RNG states included):

    # conditional CFM, 3 h, scored as the mean of 4 samples at 8 NFE
    EVAL_ARGS="--nfe 8 --samples 4" METHOD=condcfm LABEL=ext_condcfm_s0 \
        MINUTES=180 SEED=0 sbatch -w dahlia --time=04:30:00 \
        scripts/flow/fm_run.sbatch --ckpt-minutes 15
    # OT-CFM (unpaired), read as mixture - background
    EVAL_DECODE=mixture METHOD=otcfm LABEL=pilot_otcfm_s0 MINUTES=180 \
        sbatch -w saturn --time=04:30:00 scripts/flow/fm_run.sbatch \
        --ckpt-minutes 15
    # SB-CFM (sigma 1, entropic plan): METHOD=sbcfm, EVAL_DECODE=mixture
    # regression with the baseline's idt-aa loss, 1 h
    METHOD=regress_l1 LABEL=ext_regress_l1_s0 MINUTES=60 \
        sbatch -w dahlia scripts/flow/fm_run.sbatch --ckpt-minutes 5

Direct use (after `. ./scripts/flow/env.sh`, on a GPU node):

    $PYTHON scripts/flow/fm_train.py --method condcfm --label NAME \
        --minutes 180 --ckpt-minutes 15 [--seed S --batch 256 --lr 2e-4]
    $PYTHON scripts/flow/fm_eval.py OUTDIR/sphenix/flow/NAME --truth val \
        --nets ema --nfe 8 --samples 1,4,16
    $PYTHON scripts/flow/fm_eval.py OUTDIR/sphenix/flow/NAME --truth jewel \
        --steps best --nets ema --nfe 8 --samples 16
    $PYTHON scripts/flow/fm_eval.py --latency RUN_DIR... --nfe 1,8,16,64
    $PYTHON scripts/flow/coupling_diag.py --batches 256,1024,2048

Compare with the baseline (tables, `compare_arms.csv`, figures):

    B=outdir/sphenix/base; P='model_m(uvcgan-s)_d(resnet)_g(vit-modnet)'
    $PYTHON scripts/flow/fm_compare.py --extra-samples 16 \
        --flow outdir/sphenix/flow/{ext_condcfm_s0,pilot_otcfm_s0,...} \
        --baseline "$B/${P}_base_b32_lr5e-5" "$B/${P}_base_b4_lr5e-5_s0" ... \
        --reference "outdir/sphenix/pretrained/${P}_sgn_bkg_sub" \
        --out outdir/sphenix/flow/compare

Outputs per run in `OUTDIR/sphenix/flow/<label>/`: `config.json`
(arguments, versions, GPU, commit), `history.csv`, `summary.json`
(steps/s, samples/s, coupling share, peak memory, start-up, load, scoring
and end-to-end times), `inline_eval.csv`, `evals/{val,jewel}_truth.csv`
(one row per checkpoint, network and inference setting) and the per-event
jet energies `evals/*_truth/*.npy`.

## Status (2026-09-25 13:10)

Running on dahlia (A6000), jobs 20120-20126, 120 min of training each,
checkpoint every 15 min, scored on val when training ends (~14:40):

- `otcfm_pieces` ("the true pieces in the batch", a different design, not
  an improvement): each batch makes its 256 mixtures by adding its own 256
  HIJING and 256 PYTHIA events, shuffles the pieces, and the minibatch OT
  pairs them back (logged as `plan_recovery`; 100% in the checks). Same
  two-panel start (mixture, empty) as `otcfm`. Seeds 0-2,
  `OUTDIR/sphenix/flow/ext_pieces_s{0,1,2}/`, selected on the signal panel
  read directly (midpoint 16 NFE).
- `otcfm1`, the single-channel unpaired OT-CFM: real mixtures -> real HIJING
  events, one panel, jet = mixture - background. Seeds 0-2,
  `ext_otcfm1_s{0,1,2}/`, selected at 4 Euler steps as `otcfm`.
- `otcfm1` in log(E + 1) instead of log(E + 0.1) (`--log-bias 1`), seed 0,
  `ext_otcfm1_bias1_s0/`: tests the claim that a gentler scale cleans the
  image at the source.

Then: JEWEL at the val-selected checkpoints, the image read-outs
(`readout_test.py`), substructure (`substructure.py --extra`), and
`fm_compare.py` with the earlier runs. All earlier jobs have ended.

- Open: the jet-level physics (jets found in the extracted image) for the
  flows; a one-network distillation of the conditional-CFM posterior mean;
  why OT-CFM is better on JEWEL than on val.
