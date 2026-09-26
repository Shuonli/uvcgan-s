# Flow matching against UVCGAN-S on the sPHENIX decomposition

Started 2026-09-24. Branch `ddp` of `github.com/Shuonli/uvcgan-s`. The
scaling notes (`SCALING_NOTES.md`) hold the baseline's training and its
held-out scores; this file holds the flow-matching comparison. A short,
plain-language summary of both is `FLOW_SUMMARY.md`.

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

### Like-for-like: the same read-out rules for UVCGAN-S (2026-09-25)

The comparisons above set the cleaned OT-CFM (seeds + tower threshold)
against UVCGAN-S's raw output, and OT-CFM's mixture-consistent background
against UVCGAN-S's separately drawn background channel. Both are unfair:
the rules work on any extracted image. `readout_test.py
--clean-references` applies them to UVCGAN-S's jet image as well
(`docs/flow/readout_fair.csv`; seed 0 of each OT-CFM, the published
UVCGAN-S, EMA network):

| same rule for both | UVCGAN-S val / JEWEL `jer_cal` | UVCGAN-S MAE / off-jet GeV per event (val) | OT-CFM 1-panel val / JEWEL | OT-CFM MAE / off-jet (val) |
| :--- | ---: | ---: | ---: | ---: |
| raw output | 3.59 / 3.99 | 0.033 / 40 | 3.69 / 3.58 | 0.154 / 223 |
| 0.5 GeV tower threshold | 3.60 / 3.87 | 0.031 / 36 | 3.61 / 3.50 | 0.065 / 86 |
| 0.7 GeV tower threshold | 3.65 / 3.82 | 0.030 / 35 | 3.63 / 3.47 | 0.050 / 63 |
| 8 GeV seeds + 0.5 GeV | 3.61 / 3.97* | 0.029 / 34 | 3.61 / 3.50 | 0.053 / 68 |
| 10 GeV seeds + 0.7 GeV | 3.66 / 3.96* | 0.029 / 33 | 3.63 / 3.49 | 0.038 / 46 |

(*) the seeds miss 4-5% of UVCGAN-S's JEWEL jets (its extracted JEWEL
jets are softer); the two-panel OT-CFM is within 0.01-0.03 GeV of the
one-panel one throughout.

- **Image: UVCGAN-S is cleaner under every common rule** -- 4.5x on the
  raw outputs, ~25-30% in MAE and off-jet energy with the same clean-up.
- **Background image, same rule** (background = mixture - jet image, so
  both add up to the mixture): UVCGAN-S 0.033 raw / 0.029 cleaned against
  OT-CFM 0.154 / 0.038 -- UVCGAN-S better here too. The "2x better
  background" quoted earlier compared different rules and is withdrawn.
- **Val jet resolution: a tie once both use a threshold** (3.60-3.66 for
  both); raw, UVCGAN-S is better by ~0.1 GeV.
- **JEWEL jet resolution: OT-CFM better by 0.35-0.5 GeV under every
  common rule**; the threshold helps UVCGAN-S there too (3.99 -> 3.82).
- Training time is the other difference: OT-CFM reaches these numbers in
  15-75 minutes, UVCGAN-S in 8-17 hours.

### Two more designs: true pieces in the batch, and one panel (2026-09-25)

Seven 2-hour runs on dahlia (jobs 20120-20126), three seeds each and one
for log(E + 1); benchmark jobs 20127-20137. `docs/flow/compare*.csv`,
`compare_report.png` (val `jer_cal`, MAE, MSE against hours, JEWEL against
val), `readout_v2.csv`, `substructure_v2*.csv`.

Raw outputs at each model's selection setting, EMA network, val / JEWEL:

| model | best val `jer_cal` | JEWEL | MAE / MSE per tower (val) | hours to 4.00 / 3.70 GeV | ms per event |
| :--- | ---: | ---: | ---: | ---: | ---: |
| UVCGAN-S batch 32 / batch 4, 3 seeds each | 3.60-3.65 / 3.62-3.66 | 4.00-4.15 / 3.91-4.03 | 0.033 / 0.024 | 5.2-6.7 / 8.4-16.7 | 0.27 |
| two-panel unpaired OT-CFM, 4 steps | 3.67-3.68 | 3.55-3.56 | 0.151 / 0.068 | 0.33-0.5 / 0.75-1.25 | 2.0 |
| **one-panel unpaired OT-CFM**, 4 steps | 3.69-3.70 | 3.58-3.61 | 0.155 / 0.071 | 0.25 / 0.75 (1 of 3 seeds; the others hover at 3.70-3.73) | 2.0 |
| one panel, log(E + 1), 1 seed | 3.79 | 3.57 | 0.144 / 0.058 | 0.25 / - | 2.0 |
| **true pieces in the batch**, jet panel | 5.08-5.26 | 4.91-5.58 | 0.033 / 0.031 | - / - | 8.1 |
| true pieces, mixture - background (seed 0) | 4.25-4.64 | 4.83-5.03 | 0.033-0.037 / 0.024-0.026 | - / - | 8.1 |

With the same clean-up for every model (10 GeV seeds + 0.7 GeV tower
threshold), val / JEWEL `jer_cal` and val MAE: UVCGAN-S 3.66 / 3.96 /
0.029; two-panel OT-CFM 3.64 / 3.48 / 0.037; one panel 3.63 / 3.49 /
0.038; log(E + 1) 3.75 / 3.50 / 0.033; true pieces (m - b, 8 GeV + 0.5)
4.14 / 4.22 / 0.029.

- **True pieces in the batch** (a different design): the matching finds
  every mixture's pieces (plan recovery 1.000 throughout training), so it
  is supervised training on synthetic mixtures with a fixed start. It
  gives the cleanest flow image (MAE 0.033, MSE 0.031: UVCGAN-S's level)
  but poor, unstable jet energies (5.1-5.3 GeV, and 5.26 -> 6.59 between
  two checkpoints of seed 0); more solver steps make it worse (4.6-4.8 at
  4-8 steps, 5.3-5.7 at 16-32). From a fixed start with known targets the
  flow heads for an average jet, like the log-space regression; its
  JEWEL-minus-PYTHIA jet energy difference even has the wrong sign (-0.29 of
  the truth): it pulls jets toward PYTHIA's. Not competitive on jet energy.
- **One panel vs two**: the one-panel model gets to ~3.70 GeV within 15
  minutes (two panels: 45-75), with 18% less GPU memory (36 against 44
  GB, no PYTHIA pool on the GPU), but ends 0.015-0.04 GeV worse on val and
  JEWEL; images and substructure are the same (JEWEL-PYTHIA shape
  difference kept: p_T^D 0.88, z_lead 0.93, R_g 0.80, mass 0.41, against
  0.88 / 0.93 / 0.79 / 0.39). Dropping the dead panel speeds up training,
  it does not improve the result.
- **log(E + 1)**: MAE -7%, MSE -18%, val jet resolution +0.1 GeV. A trade,
  not the fix claimed earlier.
- More solver steps never help the jets (c.f. above): OT-CFM is best at 4
  Euler steps; the image is better cleaned afterwards.

## Step 2: a posterior sampler for single-event fidelity (pre-registered 2026-09-25, before training)

The request: a model built on newer methods that best preserves each
event's jet substructure and energy (single-event fidelity), with stable
training. What the results so far say:

- For each event, the best estimate of any jet property is its average
  over the decompositions consistent with the mixture (the posterior
  mean). A posterior sampler gives it for every observable at once
  (average the observable over K samples), plus a per-event uncertainty
  (their spread). Conditional CFM showed this for the cone energy: 3.60 GeV
  with K = 16 and about 3.48 as K grows, against 3.59 for the published
  UVCGAN-S.
- Flow matching trains as a regression: three seeds agree within
  +-0.01 GeV at every matched time, while the GAN's time to 3.70 GeV spans
  8-15 h.
- The sampler's weak point is JEWEL: 4.27 against 3.99 (UVCGAN-S) and 3.55
  (OT-CFM). Every model that learned the PYTHIA signals loses 0.4-0.7 GeV
  there; OT-CFM, which never models the signal, does not. The sampler has
  learned PYTHIA's jets as its prior.

Design, `postflow` (`fm_common.py`):

1. Only the jet image is generated (x1 = psi(s), noise start, conditioned
   on the mixture). Each sample is clipped to 0 <= s <= m, and the
   background is m - s, so background and jet add up to the measured energy
   in every tower (conditional CFM's two channels did not).
2. Randomised jets (`--augment jets`, `JetShapes`): half of the training
   jets are replaced by transformed ones:
   - harder or softer fragmentation at fixed energy (s^g, log g in
     [-0.38, 0.52]);
   - up to 40% of each tower's energy spread to its 8 neighbours;
   - tower-level fluctuations (log-normal, sigma up to 0.3);
   - the energy scaled by up to +-25%.

   The mixture is made from the transformed jet, so the pairs stay exact.
   The ranges were set on 8k PYTHIA training signals only. The transformed
   signals keep PYTHIA's average leading-tower share (+2.5%), p_T^D (-1.1%)
   and width (+2.5%; spreading alone would soften them by 8-17%), and their
   event-to-event spread in those grows 1.5x, 1.7x and 1.1x. Nothing was set
   on JEWEL.
3. Read-outs:
   - the K-sample mean energy, for the jet energy (as conditional CFM);
   - the per-jet mean of each observable over the samples, for
     substructure;
   - a single sample, for a realistic image;
   - the spread of the samples, as the per-event uncertainty.

Same U-Net (21.6M parameters), batch 256, lr 2e-4, EMA 0.9999, checkpoints
every 15 min. Checkpoints are selected on the 4-sample mean at 8 NFE, as
for conditional CFM.

`regress_mse` gives the posterior mean in one evaluation: s = sigmoid(net) m
per tower, trained by mean squared error in GeV on the randomised jets.
That is the estimator a K-sample mean converges to, at the cost of one pass.

Runs, on A6000 dahlia, 2 h of training each unless noted:

| arm | seeds | what it isolates |
| :--- | :--- | :--- |
| postflow [jets] | 0, 1, 2 | the proposed model; stability across seeds |
| postflow | 0 | the randomised jets (against [jets] seed 0); generating the jet alone (against conditional CFM at 2 h) |
| regress_mse [jets] | 0, 1 h | whether a one-pass mean reaches the sampler's energy resolution |

Budget: 7 GPU-h of training plus about 3 of scoring, at most 6 GPUs at once.

Metrics, fixed now:
- Primary: val `jer_cal` of the 16-sample mean at the val-selected
  checkpoint.
- JEWEL `jer_cal` at that checkpoint.
- Per-jet resolution of the posterior-mean observables (mass, girth,
  p_T^D, core, z_lead, n1) on val and JEWEL (`posterior_substructure.py`),
  against the published UVCGAN-S and the other models.
- The JEWEL-PYTHIA modification fractions.
- Per-tower `l1` and `mse`.
- Seed spread at matched training times.
- Inference ms/event.

Hypotheses and decision rules:
- H1: without randomisation, postflow's val `jer_cal` at 2 h is within
  +-0.03 GeV of conditional CFM's at 2 h (3.62 GeV with 16 samples).
- H2: the randomised jets lower JEWEL `jer_cal` by >= 0.15 GeV (postflow
  [jets] against postflow, seed 0) at a val cost of <= 0.05 GeV.
- H3: `regress_mse` [jets] reaches <= 3.65 GeV on val in one evaluation.
- postflow [jets] is recommended over UVCGAN-S if, with 16 samples, it
  reaches <= 3.62 GeV on val and <= 3.99 on JEWEL (the published model's),
  and its per-jet substructure resolution is better for most shape
  observables on both sets.

JEWEL is scored once per run, at the val-selected checkpoint, and is not
used for any choice.

### Step 2 results (jobs 20144-20157, 2026-09-25; kept as references)

Energy scores are val / JEWEL `jer_cal`, 20k events, with 16 samples at
8 NFE. The frozen-calibration resolution and the EMD come from
`jet_fidelity.py`, on 10k events each, for the mean image.

| model | `jer_cal` val / JEWEL | frozen-calibration resolution val / JEWEL | EMD per jet val / JEWEL, GeV |
| :--- | ---: | ---: | ---: |
| postflow, randomised jets, 3 seeds | 3.64 / 3.64 / 3.64 ; 4.27 / 4.30 / 4.29 | 3.64 / 4.56-4.62 | 4.75-4.78 / 4.57-4.58 |
| postflow, no randomisation, seed 0 | 3.61 / 4.28 | 3.60 / 4.54 | 4.74 / 4.52 |
| conditional CFM (`ext_condcfm_s0`) | 3.62 at 2 h, 3.60 at 3 h / 4.27 | 3.60 / 4.48 | 4.71 / 4.48 |
| `regress_mse`, randomised jets, one pass (0.5 ms) | **3.56** (3.58 after 5 min) / 4.12 | **3.56** / 4.45 | **4.54 / 4.43** |
| UVCGAN-S published | 3.59 / 3.99 | 3.57 / 4.16 | 4.75 / 4.54 |

- **H1 holds.** Generating the jet alone, with background = mixture - jet,
  costs nothing: 3.61 against 3.62 at 2 h.
- **H2 fails.** The randomised jets do not help on JEWEL (4.27-4.30 against
  4.28 without; 4.56-4.62 against 4.54 with the frozen calibration) and cost
  0.03 GeV on val.
- **H3 holds.** The one-pass posterior-mean network reaches 3.56 GeV on
  val, the best in-distribution jet energy and EMD of any model here,
  within an hour.
- **The decision rule rejects postflow** over UVCGAN-S (val 3.64 > 3.62,
  JEWEL 4.27-4.30 > 3.99).
- **Every model trained on PYTHIA signals stays at 4.1-4.3 GeV on JEWEL.**

## Backbone ablation: the UVCGAN-S generator as the OT-CFM velocity network (2026-09-25)

**Goal of the study** (restated 2026-09-25): a simple, credible OT flow-matching
method for mainly unpaired vacuum -> medium jet transformation. The
decomposition is the controlled benchmark, with event-level truth and a
strong UVCGAN-S reference. Unpaired OT-CFM already matches UVCGAN-S's
calibrated jet energy resolution in under an hour of training, against
8-17 h. The open question is event-level spatial fidelity: two jets of equal
energy can place it at different angles, and so differ in mass,
fragmentation and grooming.

**Hypothesis.** Part of the fidelity gap comes from the velocity network,
not from flow matching. All flows so far used TorchCFM's ADM U-Net, while
UVCGAN-S uses a generator already shown to work on these images.

**The two networks** (`fm_common.construct_net`, `UVCGANVelocity`):

| | ADM U-Net (all flows so far) | UVCGAN-S generator (ViT-ModNet, published sPHENIX configuration) |
| :--- | :--- | :--- |
| parameters | 21.6M | 32.1M, plus 0.2M for the time input added here |
| scales | 24 x 64 -> 12 x 32 -> 6 x 16 -> 3 x 8; 96, 192, 192, 192 channels | the same scales; 96, 192, 384 features |
| blocks | 2 residual blocks per scale (3 in the decoder) | one plain two-conv block per scale (encoder); one block of modulated, demodulated convs per scale (decoder) |
| normalisation | GroupNorm (32 groups) everywhere | none in the encoder; LayerNorm in the transformer; weight demodulation in the decoder |
| global context | self-attention (4 heads) at 6 x 16 and 3 x 8 | 12-block transformer (384 features, 6 heads) over the 24 bottleneck positions, plus an extra token whose output is a style vector that modulates every decoder conv |
| skips | every block's output, concatenated | one concatenated skip per scale; the path from below gated by a ReZero scale starting at 0 |
| time input | sinusoidal embedding -> MLP -> scale and shift of every GroupNorm | none (a GAN generator). Added here, the one change: the same kind of embedding, through a 2-layer MLP, added to the extra token, so the style carries t |
| initialisation | TorchCFM's | UVCGAN-S's (Kaiming) |

**Held fixed:**
- the one-panel unpaired OT-CFM (`otcfm1`: real mixtures -> HIJING events,
  the general source -> target form);
- exact minibatch OT (POT) on the squared L2 of standardised log(E + 0.1);
- straight path (sigma 0) and the CFM velocity loss;
- batch 256, Adam 2e-4 with 1000 warm-up steps, gradient clip 1, EMA
  0.9999;
- 2 h of training on one A6000 (dahlia), seed 0, checkpoints every 15 min;
- read-out: 4 Euler steps, jet = mixture - background in GeV;
- selection on val `jer_cal`.

Baseline: the saved U-Net runs `ext_otcfm1_s0..s2`; new run
`bb_uvcgan_otcfm1_s0`, job 20160. Only the network changed.

**Evaluation** (`jet_fidelity.py`, rules fixed before scoring):
- **Events:** the first 10k val events are the development set; val events
  10k-20k are the calibration set; the first 10k JEWEL events are the
  frozen test.
- **Energy:** the raw response, and a linear calibration fitted on the
  calibration set and frozen, with its bias, resolution and RMSE.
- **Substructure:** per-jet bias and RMSE in units of the true spread; the
  energy mover's distance and its shape part.
- **Read-outs:** raw outputs, and the same 0.5 GeV tower threshold for every
  model, against the equally thresholded truth.
- **Tables:** `docs/flow/jet_fidelity_backbone*.csv`; figure
  `docs/flow/jet_fidelity_backbone.png`.

| | U-Net, 3 seeds | UVCGAN-S generator, seed 0 | UVCGAN-S (the GAN), reference |
| :--- | ---: | ---: | ---: |
| updates in 2 h (steps/s), peak memory | 18.2-18.3k (2.53), 35.6 GB | 32.3k (4.48), 19.1 GB | |
| inference, 4 Euler steps | 2.0 ms/event (0.50 per evaluation) | 1.1 ms/event (0.27) | 0.27 ms (one pass) |
| val `jer_cal`, 20k events: 15 min / 30 min / 2 h / best | 3.70-3.73 / 3.70-3.71 / 3.70-3.73 / 3.685-3.696 | 3.82 / 3.70 / 3.665 / **3.664** | 3.59 |
| JEWEL `jer_cal`, 20k events, selected checkpoint | 3.58 / 3.61 / 3.58 | 3.55 | 3.99 |
| frozen calibration: resolution, val / JEWEL, GeV | 3.66-3.67 / 3.58-3.62 | 3.65 / 3.58 | 3.57 / 4.16 |
| frozen calibration: bias on JEWEL, GeV | +0.97 to +1.01 | +0.99 | +0.46 |
| raw response `jes`, val / JEWEL | 0.94 / 0.99 | 0.92 / 0.96 | 0.94 / 0.95 |
| EMD per jet, val / JEWEL, GeV | 5.15-5.17 / 4.74-4.75 | 5.36 / 4.77 | 4.75 / 4.54 |
| EMD shape part, val / JEWEL | 3.17-3.18 / 3.11-3.12 | **3.12 / 3.06** | 3.12 / 2.83 |
| RMSE / sigma, raw, val: mass, girth, p_T^D, z_lead, z_g, R_g | 0.85, 0.33, 0.49, 0.33, 1.09, 0.93 | 0.82, 0.32, 0.46, 0.32, 1.09, 0.92 | 1.06, 0.28, 0.41, 0.32, 1.10, 0.89 |
| the same on JEWEL | 0.99-1.00, 0.40, 0.68, 0.49, 1.14, 0.94 | 0.93, 0.39, 0.65, 0.47, 1.15, 0.94 | 1.00, 0.34, 0.60, 0.51, 1.16, 0.90 |
| bias / sigma, raw, val: girth, p_T^D, R_g | +0.19, -0.38, +0.24 | +0.18, -0.35, +0.22 | -0.09, +0.19, -0.23 |
| RMSE / sigma, 0.5 GeV threshold, val: girth, p_T^D, z_lead | 0.27, 0.38, 0.29 | 0.27, 0.37, 0.30 | 0.29, 0.44, 0.35 |

The U-Net seeds differ by 0.002-0.01 in these RMSEs. The solver curve is
unchanged. On val, at the selected checkpoint, 1 / 2 / 4 / 8 / 16 Euler
steps give 4.94 / 3.93 / 3.66 / 3.74 / 3.86 GeV (U-Net: 5.08 / 4.01 / 3.68
/ 3.72 / 3.83), and 16 midpoint evaluations give 4.10 (both).

**Verdict: a small, consistent gain, not a clear improvement.**
- **Gains:**
  - The raw per-jet RMSEs of mass, girth, p_T^D and z_lead fall by 3-7% on
    val and 2-7% on JEWEL.
  - The shape part of the EMD falls by 1.5%, to UVCGAN-S's level on val
    (not on JEWEL).
  - Calibrated energy resolution is equal or 0.01-0.02 GeV better. The
    generator starts slower (3.82 GeV at 15 min against 3.70-3.73), passes
    3.70 at the same 30 min, and keeps improving to 3.66. The U-Net
    plateaus at 3.69-3.70 and drifts up to 3.70-3.73 by 2 h. At an equal
    number of updates (18k) it is 3.67 against 3.70.
- **Unchanged or worse:**
  - The raw response is lower (0.92 against 0.94), so the full EMD is worse
    (5.36 against 5.15 GeV).
  - The characteristic biases of the raw OT-CFM image are unchanged: girth
    +0.18 sigma, p_T^D -0.35 sigma, R_g +0.22 sigma, the noise floor around
    the jet.
  - Only 15-35% of the raw girth / p_T^D gap to UVCGAN-S closes.
  - With the 0.5 GeV threshold both backbones are identical.
- **Conclusion:** the backbone is at most a minor part of the fidelity gap.
  The rest points at the coupling or the objective: a few-step solve of an
  unpaired, averaged coupling.
- **No three-seed confirmation.** Given the tiny U-Net seed spread, the
  small gains are likely real, but confirming them would not change the
  conclusion.

**Compute trade-off.** More parameters (32.3M against 21.6M), yet 1.8x more
updates per hour, half the memory and 1.8x cheaper inference: at 24 x 64 the
generator's large transformer runs on only 24 positions. As a flow backbone
it costs nothing extra and helps a little.

**Next experiment supported by this result.** Change the coupling, not the
network: the known-modification closure test of the matching cost (next
section). The backbone is held fixed there at the ADM U-Net. It is the
validated choice (three seeds, the TorchCFM reference), and at the closure
test's 16 x 16 jet canvases the generator's transformer would see only 2 x 2
positions.

    METHOD=otcfm1 LABEL=bb_uvcgan_otcfm1_s0 MINUTES=120 SEED=0 \
        EVAL_DECODE=mixture EVAL_ARGS="--nfe 4 --solver euler" \
        sbatch -w dahlia --time=03:30:00 scripts/flow/fm_run.sbatch \
        --ckpt-minutes 15 --inline-events 1000 --inline-nfe 4 --backbone uvcgan
    $PYTHON scripts/flow/jet_fidelity.py \
        --models 'OT-CFM-uvcgan-s0=single:bb_uvcgan_otcfm1_s0:mixture'
    $PYTHON scripts/flow/jet_fidelity.py --report \
        --show OT-CFM-unet-s0,OT-CFM-uvcgan-s0,UVCGAN-S \
        --figure docs/flow/jet_fidelity_backbone.png

## Closure test: does a shape-plus-energy matching cost improve event-level fidelity? (set up 2026-09-25, before training)

**Why.** Competitive average energy resolution does not show that a
transport keeps the identity of each jet while modifying its substructure.
Two earlier observations point at the source-target coupling:
- full-image matching is driven by where the jet is in the event;
- jet-centred matching barely follows the energy (`vac_med_coupling.py`).

**Test.** Hold the backbone fixed and change only the matching cost, in an
unpaired experiment whose modification is known.

**Jets** (`closure_data.py`; manifest `OUTDIR/sphenix/flow/cache/closure_manifest.json`):
- **Source:** clean PYTHIA events from the signal training cache, 600k
  random events read.
- **Axis:** the leading R = 0.4 cone (as the jet scores).
- **Selection:** cone >= 10 GeV, with the 9 x 9 window around the axis
  inside the acceptance (axis rows 4-19). 525k jets pass (87%); median cone
  energy 32 GeV.
- **Frame and crop:** the jet image J is the 53 towers of the R = 0.4 cone,
  in GeV of tower E_T as the images hold them, centred on the axis. It sits
  on a 16 x 16 canvas (window at rows and columns 4-12) so that the U-Net can
  downsample three times; the rest of the canvas is zero.
- **Energy observable:** E = sum of J.
- **Preprocessing:** the flow's existing psi = log(E + 0.1), standardised
  with one mean and deviation fitted on the training canvases of both pools
  (-2.10, 0.643; `norm_closure.json`). Amplitudes stay physical up to that
  map; nothing is rescaled.

**Known modification.** T(J) = 0.8 [0.8 J + 0.2 K(J)].
- K moves each tower's energy to its in-cone neighbours (3 x 3 minus the
  centre), with weights exp(-dR^2 / 2 (0.1)^2) normalised over the in-cone
  neighbours of that tower, so K preserves the sum.
- Checks: no negative tower; nothing outside the cone; E(T(J)) / E(J) =
  0.8000000 +- 3e-7 (float32), 0.7997-0.8003 as stored in float16.
- It is a method-validation toy, not a quenching model.

**Splits**, disjoint by parent event, from one random permutation of the
selected jets:
- train source A, 200k;
- train target B, 200k, stored only as T(B);
- validation, 10k pairs (J, T(J));
- test, 20k pairs.

Training sees A and T(B) of different events, never a pair and never T's
parameters. Selection acts on J before the modification and nothing is
reselected after it. (Re-finding the leading jet in a modified full event
would pick the other jet of the dijet in 57% of events.)

**Costs** (`fm_common.ShapeEnergyCost`), each passed to the same exact OT
solver as training (TorchCFM's `pot.emd`, uniform weights, default
iterations), with pairs drawn from the plan with replacement by the same
`sample_map`:
- **Baseline, the existing jet-centred cost:** squared L2 of the
  standardised states, TorchCFM's own. With one normalisation for both
  pools this gives the same plan as `vac_med_coupling.py`'s "centred" cost,
  the squared L2 of log(E + 0.1) over the cone towers.
- **Candidate:** with Q = J / E (for shape features only),

      D_s(i, j) = |P_s Q_i - P_s Q'_j|^2   P_s: sum pooling in s x s blocks of the canvas
      D_E(i, j) = [log((E_i + 0.1 GeV) / (E'_j + 0.1 GeV))]^2
      C(i, j)   = mean over s in {1, 2, 4} of D_s / a_s  +  lambda D_E / a_E

  - lambda = 1 for the candidate; lambda = 0 as a matching-only
    diagnostic.
  - a_s and a_E are the medians of the positive source-target distances
    between the first 2048 jets of each training pool, frozen in
    `closure_cost_calib.json`: a_s = 0.210, 0.303, 0.306 and a_E = 0.0664.
    No scale was degenerate.
  - The pooling grid is fixed on the canvas. The axis tower (8, 8) sits at
    a block corner, the same for every jet.

**Matching audit** (`closure_matching.py`, before training; 4 repeats of
unpaired batches, `docs/flow/closure_matching.csv`, figure
`docs/flow/closure_matching.png`). Numbers are batch 256 / batch 1024:

| cost | energy rank correlation, source vs matched target | matched / source energy: median, IQR | normalised-shape EMD of matched pairs | axis distance in the full events |
| :--- | ---: | ---: | ---: | ---: |
| random | -0.03 / -0.03 | 0.80, 0.30 / 0.80, 0.31 | 0.44 / 0.44 | 1.70 / 1.69 |
| full image | 0.09 / 0.10 | 0.81, 0.30 / 0.80, 0.29 | 0.44 / 0.44 | **1.53 / 1.38** |
| jet-centred (baseline) | 0.10 / 0.16 | 0.81, 0.29 / 0.80, 0.28 | 0.21 / 0.19 | 1.72 / 1.72 |
| shape only (lambda 0) | 0.02 / 0.08 | 0.80, 0.30 / 0.80, 0.29 | **0.19 / 0.17** | 1.75 / 1.70 |
| shape + energy (lambda 1) | **0.95 / 0.98** | 0.80, **0.065** / 0.80, **0.047** | 0.23 / 0.20 | 1.71 / 1.69 |

- **What drives each cost:**
  - Full-image matching pairs jets by position, not shape.
  - The jet-centred cost pairs by the pattern of log tower energies and
    hardly by energy.
  - The candidate pairs almost monotonically in energy, at the ratio the
    known T implies. That costs a little shape similarity: 0.23 against
    0.21 at batch 256, equal at 1024.
- **Invariance:** independent periodic phi shifts of the full events,
  followed by re-finding the axis and re-centring, leave every centred cost
  matrix exactly unchanged. The full-image cost changes by up to 40%.
- **Caveat:** none of this validates a correspondence. A permutation keeps
  the target marginal whatever the cost, and the trained ODE need not
  follow the minibatch pairs.

**Training** (`closure_run.sbatch` -> `fm_train.py --method jetflow`):
- held identical for both costs: ADM U-Net (21.6M parameters), batch 256,
  Adam 2e-4, 1000 warm-up steps, gradient clip 1, EMA 0.9999, straight path
  (sigma 0), CFM velocity loss;
- seed 0, so the initialisation and the drawn batches are identical;
- 2 h on one A6000 (dahlia), checkpoints every 10 min;
- inference: 4 Euler steps (the OT-CFM setting), outputs clipped at 0.

**Selection metric, fixed now.** Mean per-jet EMD between F(J) and T(J)
(GeV, R = 0.4) on the first 5000 validation pairs, EMA network, 4 Euler
steps (`closure_eval.py --select`).

**Test metrics** (`closure_eval.py`, 20k test pairs), with the identity and
a random target jet as reference outputs:
- the raw energy response E(F(J)) / E(J), whose target is 0.8 (no
  calibration);
- bias and RMSE of E(F(J)) - E(T(J));
- normalised-shape EMD;
- per observable: bias and RMSE of O(F(J)) - O(T(J)), the mean modification
  recovered (mean Delta_pred / mean Delta_true), and the per-jet correlation
  of Delta_pred with Delta_true;
- marginal W1 / sigma, and the largest difference between the correlation
  matrices of (log E, observables);
- the same with a better-resolved solve (32 midpoint evaluations) on 2000
  test pairs, for the direction of any difference.

**Decision rule.** The candidate improves clearly if, at the selection
setting and in the same direction with the resolved solve, it lowers:
- the shape EMD;
- the RMSE of O(F(J)) - O(T(J)) for most of mass, girth, p_T^D and z_lead;

without a worse energy response or RMSE and without worse marginals. Then
seeds 1 and 2 of both costs follow. Otherwise the notes report what got
worse.

### Closure test results (seed 0, jobs 20170-20172, 2026-09-26)

- **Training:** 2 h each on dahlia.
  - Existing cost: 84.0k updates (11.7 steps/s).
  - Shape + energy: 81.5k updates (11.3 steps/s; the cost adds ~3% to the
    step).
- **Selection** (validation EMD, as fixed):
  - The existing cost's best checkpoint was its first (10 min, 7.0k
    updates). Its validation EMD then rose from 4.60 to 4.66-4.75 GeV and its
    shape EMD from 0.081 to 0.084.
  - The candidate improved throughout: 4.41 -> 3.99 GeV, selected at 110
    min.
- **Inference:** 0.37 ms per jet with 4 Euler steps, 2.9 ms with 32
  midpoint evaluations.
- **Files:** `docs/flow/closure_eval_s0*.csv`, `docs/flow/runs/closure_*`,
  figure `docs/flow/closure_jets.png`.

Test set: 20k held-out pairs (J, T(J)); the resolved solve, 32 midpoint
evaluations, on the first 2000 pairs.

| | identity | random target | existing cost, 4 Euler | shape + energy, 4 Euler | existing, 32 midpoint | shape + energy, 32 midpoint |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| response E(F(J)) / E(J) (truth 0.8) | 1 | 0.83 +- 0.24 | 0.70 +- 0.08 | 0.71 +- 0.04 | 0.79 +- 0.09 | **0.81 +- 0.04** |
| E(F(J)) - E(T(J)): bias / RMSE, GeV | +6.5 / 6.6 | 0 / 7.4 | -3.4 / 4.6 | -2.9 / 3.2 | -0.6 / 3.0 | **+0.2 / 1.3** |
| log energy ratio RMSE | 0.22 | 0.28 | 0.18 | 0.13 | 0.12 | **0.05** |
| EMD to T(J), GeV | 7.0 | 14.5 | 4.6 | 4.0 | 3.7 | **2.9** |
| normalised-shape EMD to T(J) | **0.032** | 0.43 | 0.082 | 0.078 | 0.091 | 0.092 |
| RMSE / sigma: mass | 0.96 | 1.41 | 0.64 | 0.67 | 0.59 | 0.52 |
| RMSE / sigma: girth | **0.10** | 1.42 | 0.30 | 0.31 | 0.32 | 0.33 |
| RMSE / sigma: p_T^D | 0.73 | 1.41 | 0.86 | 0.62 | 0.64 | 0.54 |
| RMSE / sigma: z_lead | 0.55 | 1.41 | 0.80 | 0.50 | 0.69 | 0.50 |
| RMSE / sigma: z_g / R_g | 0.82 / **0.44** | 1.41 / 1.42 | 1.03 / 0.77 | 1.02 / 0.78 | 1.03 / 0.85 | 1.09 / 0.85 |
| mean change recovered, Delta_pred / Delta_true: E, mass, p_T^D, z_lead | 0 | 1 | 1.53, 1.42, 1.96, 2.08 | 1.44, 1.47, 1.69, 1.64 | 1.09, 1.15, 1.19, 1.21 | 0.97, 0.86, 1.39, 1.39 |
| marginal W1 / sigma: E, p_T^D, z_lead, girth | 1.24, 0.67, 0.49, 0.08 | 0 | 0.66, 0.64, 0.53, 0.13 | 0.55, 0.46, 0.31, 0.15 | 0.11, 0.13, 0.10, 0.06 | **0.04**, 0.26, 0.19, 0.11 |
| largest difference of the (log E, observables) correlation matrices | 0.24 | 0 | 0.23 | 0.45 | 0.09 | 0.10 |

(The per-jet correlation of Delta_pred with Delta_true, also in the CSV, is
not informative: both contain -O(J), so even the random target reaches
0.66-0.78.)

**Against the rule set before training: not a clear improvement.**
- **Better, in both solves:** the per-jet energy correspondence. With the
  resolved solve the energy RMSE drops from 3.0 to 1.3 GeV and the
  response is 0.81 ± 0.04 against the true 0.8, with less spread; in log
  terms the RMSE falls 0.12 -> 0.05. p_T^D and z_lead are better too.
- **Not better, or worse:**
  - The normalised-shape EMD shows no gain with the resolved solve (0.092
    against 0.091).
  - Girth is slightly worse (0.33 against 0.32).
  - With the resolved solve the shape marginals are further from the target
    (W1 of p_T^D 0.26 against 0.13, z_lead 0.19 against 0.10, girth 0.11
    against 0.06). The candidate over-does the softening (p_T^D and z_lead
    changes recovered 1.39x).
  - With 4 steps its (log E, observables) correlation structure is worse
    (0.45 against 0.23).
- Seeds 1-2 were therefore not run.

**Representative failure, common to both costs.** Neither transport keeps
each jet's fine structure:
- The output shape is 2.5-3x further from the true modified jet than the
  unmodified input is (shape EMD 0.078-0.092 against 0.032). The per-jet
  girth and R_g errors are 3x and 2x the identity's.
- `docs/flow/closure_jets.png` shows what happens. The coarse layout of each
  jet (where its prongs and core are) survives. The tower-level structure is
  smeared into a diffuse halo: the flows put 2.6-4.4% of the jet energy in
  towers that T(J) leaves (almost) empty, against 1.3% for T(J) itself, and
  flatten the leading tower (its share 0.20-0.245 against 0.256).
  *Corrected 2026-09-26:* "almost empty" meant T below 0.05 GeV. Those
  towers are occupied, and the exactly empty towers hold nothing. The error
  is a flattening of each jet's core; see "Locating the closure-test
  failure".
- It is the noise floor of the decomposition flows, and no cost change here
  removes it. The minibatch coupling explains why: its matched targets are
  5-7x further in shape from a source jet than the jet's own T(J) is (audit:
  0.17-0.23 against 0.032). No training pair carries the fine correspondence,
  and the flow averages over coarse ones.
- The four-step solve, the decomposition's choice, is biased here for both
  costs (response 0.70-0.71 against the true 0.8). The resolved solve
  corrects the energy, so a jet -> jet flow should be read with an accurate
  solve.

**Conclusion.**
- **What the coupling change does:** it improves event-level fidelity in the
  energy (the soft log-energy term is enough for the unpaired transport to
  recover each jet's energy change to 5%), and in p_T^D and z_lead.
- **What it does not do:** improve the event-level shape, which both costs
  lose to the same smearing.
- **For PYTHIA -> JEWEL:** the evidence supports the method for per-jet
  energy loss under this modification, not yet for per-jet substructure
  modifications, so it does not yet justify a PYTHIA -> JEWEL substructure
  study.
- **Suggested next experiment:** make the transport keep a jet's own fine
  structure, and test it in the same closure test. For example:
  - larger minibatches (the audit's shape distance of matched pairs falls
    only from 0.21 to 0.19 between 256 and 1024);
  - a representation or objective that does not average near-empty towers
    upward (the halo is also the decomposition's noise floor). *Superseded
    by the paired positive control below: the pipeline is not the cause.*
- **Caveat:** passing such a test would still support the method only under
  the tested modification; it would not validate the physical
  correspondence of real PYTHIA and JEWEL jets.

    python scripts/flow/closure_data.py                      # pools, T, norm
    python scripts/flow/closure_matching.py                  # audit, cost scales
    LABEL=closure_l2_s0 COST=l2 SEED=0 MINUTES=120 \
        sbatch -w dahlia scripts/flow/closure_run.sbatch     # + validation selection
    LABEL=closure_se1_s0 COST=shape_energy LAMBDA=1.0 SEED=0 MINUTES=120 \
        sbatch -w dahlia scripts/flow/closure_run.sbatch
    python scripts/flow/closure_eval.py closure_l2_s0 closure_se1_s0 \
        --out outdir/sphenix/flow/closure_eval_s0            # test pairs
    python scripts/flow/closure_eval.py closure_l2_s0 closure_se1_s0 \
        --figure docs/flow/closure_jets.png

## Locating the closure-test failure: pipeline checks, paired positive control, null test (2026-09-26)

The plan: inexpensive pipeline checks, then one paired flow-matching
positive control. A null test and a larger matching pool only if the
control succeeds.

### Pipeline checks (`closure_checks.py`, no training; the first 512-2000 test pairs)

- **Normalisation round trip** (psi = log(E + 0.1), standardised):
  - largest per-tower error 1.5e-5 GeV in float32, 2.7e-5 GeV from the
    float16 training cache;
  - zero towers stay exactly zero;
  - total energy is reproduced to 4e-7.
- **A velocity field that is exactly zero**, through the inference path
  (4 Euler steps and 32 midpoint evaluations, with and without the clip at
  0), returns J to 1.5e-5 GeV.
- **Probability path.** TorchCFM with sigma 0: x_t = t x1 + (1 - t) x0,
  u_t = x1 - x0, t ~ U(0, 1). The path is exactly x0 at t = 0 and x1 at
  t = 1, with no noise at either end. Inference starts at z(J), which is
  training's x0.
- **Clipping.** Raw outputs are >= -0.1 GeV by construction. The clip at 0
  changes the cone energy by < 0.1% and the energy of the empty towers not
  at all.
- **Halo, corrected.**
  - T(J) has only 1.3 exactly empty cone towers per jet: PYTHIA jets fill
    the cone, and the broadening fills the rest.
  - The unpaired flows put 0.0002-0.0006 GeV per jet there (32 midpoint
    evaluations).
  - The earlier "2.6-4.4% in towers T(J) leaves almost empty" counted
    towers below 0.05 GeV. Those are occupied, so there is no empty-tower
    halo.
- **What the error is.** Signed per-tower error by true tower energy, on
  20k test pairs with 32 midpoint evaluations:
  - The unpaired flows lower each jet's hardest towers (T > 5 GeV) by 0.87
    and 1.82 GeV on average, with 2.2-3.4 GeV rms.
  - They spread that energy over the ~32 softest towers, +0.03-0.04 GeV
    each.
  - So the core of each jet is flattened.
- **No implementation bug was found.** No earlier comparison needs updating
  beyond the halo wording (corrected in place above).

### Paired positive control (`closure_paired_s0`, job 20182)

**Configuration:** identical to the unpaired closure runs:
- the ADM U-Net (21.6M), the backbone of the saved unpaired runs, so that
  only the pairing changes;
- standardised psi, straight path (sigma 0), CFM velocity loss;
- Adam 2e-4, warm-up 1000, gradient clip 1, EMA 0.9999, batch 256;
- 2 h, seed 0, the 200k training source jets A.

The one difference: each source jet is paired with its own T(J), computed
on the fly (`--pairing paired`), with no matching.

Checkpoints are selected on the validation EMD with the accurate solve
(32 midpoint evaluations). The unpaired runs were reselected the same way
for this comparison (at 80 and 110 min). They had used the 4-step solve
before.

**Training against validation:**
- The paired run's EMD (val / train) is 0.048 / 0.048 GeV at 10 min,
  0.032 / 0.031 at 20, 0.019 / 0.019 at 50 and 0.015 / 0.015 at 120. Its
  shape EMD falls from 0.0006 to 0.00016.
- The unpaired runs sit at a shape EMD of 0.089-0.093 on both val and
  train, flat from the first checkpoint to the last. No fitting or
  generalisation gap is involved.

**Test** (20k held-out pairs, 32 midpoint evaluations; every output clipped
at 0; E_in = E(J), E_true = E(T(J)); `docs/flow/closure_control_m32.csv`,
event displays `docs/flow/closure_control_jets.png`):

| | identity F(J) = J | paired control | unpaired, jet-centred cost | unpaired, shape + energy cost |
| :--- | ---: | ---: | ---: | ---: |
| shape EMD to T(J), mean / median | 0.032 / 0.032 | **0.0002 / 0.0001** | 0.090 / 0.077 | 0.091 / 0.083 |
| EMD to T(J), GeV | 7.03 | **0.015** | 3.56 | 2.87 |
| E_out / E_in (truth 0.8) | 1 | **0.7997 +- 0.0004** | 0.785 +- 0.079 | 0.808 +- 0.038 |
| E_out / E_true (truth 1) | 1.25 | 0.9996 | 0.981 | 1.010 |
| E_out - E_true: bias / RMSE, GeV | +6.5 / 6.6 | **-0.011 / 0.021** | -0.68 / 2.83 | +0.22 / 1.32 |
| mass change O(F) - O(J) against O(T) - O(J) (true -0.96 +- 0.31 GeV): bias / RMSE, GeV | +0.96 / 1.01 | **-0.002 / 0.003** | +0.07 / 0.55 | +0.14 / 0.54 |
| girth change (true +0.0032 +- 0.0030): bias / RMSE | -0.0032 / 0.0044 | **0.0000 / 0.00004** | +0.0004 / 0.0144 | -0.0023 / 0.0146 |
| energy in the truth-empty towers (1.3 per jet), before / after the clip, GeV | 0 | 0 / 0 | 0.0004 / 0.0006 | -0.0001 / 0.0002 |
| hardest towers (T > 5 GeV): error mean / rms, GeV | +4.05 / 4.45 | -0.006 / 0.014 | -1.82 / 3.44 | -0.87 / 2.16 |
| updates in 2 h | | 89.8k (12.5 /s) | 84.0k | 81.5k |

**The positive control succeeds.** With the correct pairing, the same
representation, path, loss, network and solve reproduce T(J) nearly
exactly:
- the shape error is 200x below the identity's and 450x below the unpaired
  runs';
- every tower class is within 0.014 GeV rms;
- the mass and girth changes are recovered to 1% of their true spread;
- all of this within 10 minutes of training, with no gap between training
  and validation pairs.

So neither the MSE velocity objective, nor the log representation, nor the
solve blurs fine structure here. The unpaired failure lies in the endpoint
assignment, the coupling. This is a diagnostic control, not evidence for
the unpaired claim.

### Unpaired null test (`closure_null_se1_s0`, job 20186)

**Configuration:**
- **Source:** A.
- **Target:** an independent pool of unmodified PYTHIA jets with the same
  selection: B before T, rebuilt from its parent events by
  `closure_data.py --null`. T of the rebuilt jets matches the stored T(B) to
  4e-4.
- **Cost:** shape + energy (lambda 1, its frozen scales); everything else as
  in the unpaired closure runs.
- **Scoring:** F(J) against J, on the same held-out jets;
  `docs/flow/closure_null_m32.csv`, event displays
  `docs/flow/closure_null_jets.png`.
- **Validation and training agree** (EMD 0.22 / 0.22 GeV); the checkpoint
  selected is at 80 min.

| | the toy modification, J -> T(J) | null test: F(J) against J |
| :--- | ---: | ---: |
| shape EMD, mean / median | 0.032 / 0.032 | 0.0033 / 0.0028 (10% of the toy's) |
| EMD, GeV | 7.03 | 0.22 |
| energy ratio | 0.8 | 0.999 +- 0.007 (RMSE 0.24 GeV) |
| mass change: mean, RMSE | -0.96 +- 0.31 GeV | -0.001, 0.035 GeV (4% of the toy's mean change) |
| girth change: mean, RMSE | +0.0032 +- 0.0030 | +0.0001, 0.0005 (16% of the toy's) |
| hardest towers (> 5 GeV) | J above T(J) by 4.05 GeV | -0.05 / 0.19 GeV rms |

**The null test passes.**
- Without a domain shift, the unpaired transport is nearly the identity.
  Its spurious changes are 4-16% of the toy modification's, so it does not
  deform jets on its own.
- It loses fine structure only when it has to move the jets. The loss comes
  from learning a shift through the unpaired coupling, not from the
  pipeline.

### Larger matching pool (`closure_se1_pool1024_s0`, job 20188)

**Configuration:**
- a pool of 1024 jets per domain;
- each step, 256 complete pairs drawn from the 1024 x 1024 exact plan
  (`--ot-pool 1024`);
- the same shape + energy cost and scales, solver, network, loss, batch and
  2 h;
- run on the modification closure, the clearest failure (the null test's
  deformation is small).

**Cost:** the matching takes 80% of each step, 2.4 steps/s against 11.3 at
pool 256, so 17.2k updates in 2 h against 81.5k.

**Validation:** shape EMD 0.075-0.077 from the first checkpoint to the last,
train equal to val.

Test (20k pairs, 32 midpoint evaluations; `docs/flow/closure_pool_m32.csv`):

| | identity | paired control | unpaired, pool 256 | unpaired, pool 1024 |
| :--- | ---: | ---: | ---: | ---: |
| shape EMD, mean / median | 0.032 / 0.032 | 0.0002 / 0.0001 | 0.091 / 0.083 | 0.076 / 0.070 |
| EMD, GeV | 7.03 | 0.015 | 2.87 | 2.37 |
| E_out / E_in (0.8) | 1 | 0.7997 | 0.808 +- 0.038 | 0.802 +- 0.032 |
| E_out - E_true: bias / RMSE, GeV | +6.5 / 6.6 | -0.011 / 0.021 | +0.22 / 1.32 | +0.01 / 1.10 |
| mass change: bias / RMSE, GeV | +0.96 / 1.01 | -0.002 / 0.003 | +0.14 / 0.54 | +0.01 / 0.47 |
| girth change: bias / RMSE | -0.0032 / 0.0044 | 0.0000 / 0.00004 | -0.0023 / 0.0146 | -0.0019 / 0.0113 |
| hardest towers: error mean / rms, GeV | +4.05 / 4.45 | -0.006 / 0.014 | -0.87 / 2.16 | -0.50 / 1.61 |

The 4x larger pool lowers every fine-structure error by 15-25% and the
energy error by 17%, despite 4.7x fewer updates. But the shape error is
still 2.4x the identity's and ~400x the paired control's, and the per-jet
girth error is 2.6x the identity's. This is a real improvement, not a fix.
As planned, no further pool sizes or costs were tried.

### What the controls establish, and the next justified change

**Established** (one seed each; the earlier U-Net seeds differed little):
- **The pipeline is sound.** Given the correct endpoint pairs, this flow
  matching pipeline reproduces each jet's modification almost exactly:
  energy, tower pattern, mass and girth. The representation, straight
  path, MSE velocity loss, network, normalisation, clipping and solver are
  therefore not what loses fine structure.
- **The coupling is where it is lost.** With the same pipeline and a real
  shift to learn, the unpaired minibatch-OT coupling loses each jet's fine
  structure. Train equals val, and the error is flat from the first
  checkpoint on, so the objective itself settles there.
- **Without a shift, the unpaired map is nearly the identity.** It does not
  deform jets unprompted.
- **What the coupling controls:**
  - the energy term of the cost fixes the per-jet energy correspondence;
  - a 4x larger matching pool improves the fine structure modestly, at 80%
    matching overhead;
  - the audit had shown the same trend: the shape distance of matched pairs
    falls only from 0.21 to 0.19 between pools of 256 and 1024.

**Uncertain:**
- whether much larger pools or other couplings would converge to T; exact
  OT cannot scale much further, and even the population OT map of this
  cost is T only approximately, since sum pooling and K do not commute
  exactly;
- how this toy's behaviour carries over to a realistic, more complex
  modification;
- the basic limit: unpaired PYTHIA and JEWEL samples do not define a unique
  per-jet correspondence.

**The next justified change concerns the coupling, not the flow matching
training or representation:**
- Scaling the minibatch pool is an inefficient route, with a 4x pool for
  -17%.
- The single next experiment supported here is a semi-paired closure test:
  the same unpaired training plus a small fraction (for example 1% and 5%)
  of true (J, T(J)) pairs, measured by how much event-level fidelity each
  fraction recovers.
- For vacuum -> medium, JEWEL can provide such pairs in simulation: vacuum
  and medium showers of the same hard scattering.
- Until then, unpaired OT flow matching supports per-jet energy-loss
  statements under this toy, not per-jet substructure ones.

    python scripts/flow/closure_checks.py --n 2000
    LABEL=closure_paired_s0 COST=l2 SEED=0 MINUTES=120 \
        EVAL_ARGS="--setting 32:midpoint --train-n 5000" \
        sbatch -w dahlia scripts/flow/closure_run.sbatch --pairing paired
    python scripts/flow/closure_data.py --null
    LABEL=closure_null_se1_s0 COST=shape_energy LAMBDA=1.0 SEED=0 MINUTES=120 \
        EVAL_ARGS="--setting 32:midpoint --train-n 5000 --truth identity" \
        sbatch -w dahlia scripts/flow/closure_run.sbatch \
        --target-domain closure_null_tgt
    LABEL=closure_se1_pool1024_s0 COST=shape_energy LAMBDA=1.0 SEED=0 MINUTES=120 \
        EVAL_ARGS="--setting 32:midpoint --train-n 5000" \
        sbatch -w dahlia scripts/flow/closure_run.sbatch --ot-pool 1024
    python scripts/flow/closure_eval.py closure_l2_s0 closure_se1_s0 --select \
        --setting 32:midpoint --train-n 5000          # reselect, accurate solve
    python scripts/flow/closure_eval.py closure_paired_s0 closure_l2_s0 \
        closure_se1_s0 --control --setting 32:midpoint \
        --out outdir/sphenix/flow/closure_control_m32
    python scripts/flow/closure_eval.py closure_null_se1_s0 --control \
        --setting 32:midpoint --truth identity --out outdir/sphenix/flow/closure_null_m32

(`sbatch` failed on 2026-09-26 with "I/O error writing script/environment
to file" for two submissions; those ran the same script through a detached
`srun`.)

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

## Status (2026-09-26 08:20)

All runs and benchmark jobs have ended; nothing of this study is running.

- **Done 2026-09-25/26:**
  - the posterior-sampler study (step 2), kept as a reference;
  - the backbone ablation (the backbone is a minor part of the fidelity
    gap);
  - the jet -> jet closure test with two matching costs;
  - locating its failure (pipeline checks, paired positive control, null
    test, larger matching pool).
  - **Result:** the unpaired coupling, not the flow-matching pipeline, loses
    each jet's fine structure. The energy term fixes per-jet energy; a 4x
    larger pool helps 17%.
- **Next, as the controls suggest:** a semi-paired closure test (a small
  fraction of true pairs added to unpaired training). For vacuum ->
  medium, JEWEL vacuum/medium pairs of the same hard scattering would
  supply them.
- **Also open:**
  - the jet-level physics (jets found in the extracted image);
  - why OT-CFM is better on JEWEL than on val;
  - the dependence of any unpaired correspondence on the chosen cost.
