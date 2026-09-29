# Flow matching against UVCGAN-S on the sPHENIX decomposition

Started 2026-09-24. Branch `ddp` of `github.com/Shuonli/uvcgan-s`. The
scaling notes (`SCALING_NOTES.md`) hold the baseline's training and its
held-out scores; this file holds the flow-matching comparison. A short,
plain-language summary of both is `FLOW_SUMMARY.md`.

Questions: can a flow-based model reach the current UVCGAN-S physics
performance in fewer GPU-hours; is it more stable across seeds; how do final
quality and inference cost compare. Treated as hypotheses.

**Latest (2026-09-28, 23:00):** the section "PYTHIA -> JEWEL translation
pilot" tests the intended application directly: unpaired OT-CFM and online
alpha-DSBM between clean PYTHIA and clean JEWEL jets, with its own data set
and appendix `docs/flow/translation/slides/translation_appendix.pdf`.
The subtraction study closes with "Consolidated benchmark" (background-only
against joint subtraction, reconstructed-jet analysis of the paper, 3 seeds
per arm) and its deck `docs/flow/bench/slides/bench_deck.pdf`.

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
| `*.sbatch` | SLURM wrappers: `fm_smoke`, `fm_run` (train + score), `make_cache`, `bench_post` (benchmark scoring and images of a run) |
| `make_pairs.py` | the training mixtures with their own signals (by index key), for the paired arms |
| `bench_images.py`, `bench_jets.py`, `bench_report.py`, `bench_diag.py`, `bench_cost.py`, `bench_loss_t.py`, `bench_tables.py`, `bench_all.sh` | the consolidated benchmark: images, FastJet analysis, paper figures and tables, tower diagnostics and displays, cost, loss against t, deck tables, the whole CPU chain |

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
| global context | self-attention (4 heads) at 6 x 16 and 3 x 8 | 12-block transformer (384 features, 6 heads) over the 24 bottleneck positions, plus an extra token whose output is a style vector that modulates every decoder conv. Inherited quirk (upstream `ViTInput`): the Fourier position code is built on a `meshgrid(x, y)` in (W, H) order while tokens are flattened in (H, W) order, so 22 of the 24 tokens get another position's code; each token's code is still unique and fixed |
| skips | every block's output, concatenated | one concatenated skip per scale; the upsampled path from below enters ungated (the decoder ReZero option, `modnet_rezero`, is off in the published configuration; corrected 2026-09-28, an earlier version of this table said it was gated at 0) |
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

## Semi-paired closure test (set up 2026-09-26, before any scoring)

**Question.** The controls placed the loss of fine structure in the
unpaired coupling. How much event-level fidelity do a few true pairs restore
in otherwise unpaired training?

**Data.** The closure pools, unchanged. "Paired share" means that the
first K training source jets A[:K], from training parents only, come with
their own T(J):
- K = 2000, 1% of the 200k source jets;
- K = 10000, 5%.

The unpaired pools, A and T(B), are unchanged.

**Runs** (seed 0, 2 h, checkpoints every 10 min; everything else as in the
other closure runs: U-Net, Adam 2e-4, EMA 0.9999, sigma 0 straight path,
CFM velocity loss, batch 256):
- **semi-paired, K = 2000 and K = 10000** (`--pairing semi --paired-n K
  --paired-share 0.5`). Each batch holds 128 true pairs drawn from A[:K],
  plus 128 pairs drawn from the exact OT plan of 256 unpaired sources and
  256 unpaired targets. That plan uses the shape + energy cost (lambda 1,
  frozen scales), so the matching pool is the unpaired run's.
- **paired-only control, the same K** (`--pairing paired --paired-n K`):
  the same pairs, with no unpaired data.

**Why half of each batch.**
- For one source jet, a true pair and an OT pair point to different
  targets, and flow matching learns roughly their frequency-weighted
  average.
- At the natural 1-5% share, the pairs would move the result only 1-5% of
  the way from the unpaired answer.
- So the pairs are a small share of the data but get half the weight.

**Evaluation:**
- checkpoints selected on the validation EMD (32 midpoint evaluations);
- the control report on the 20k test pairs;
- to check for memorisation: the seen pairs A[:2000] against training jets
  whose pairs were never shown, A[K:K+2000];
- compared with 0% (the unpaired shape + energy run), 100% (the full paired
  control) and the identity.

**How the results are read.**
- The pairs restore the detail if the per-jet shape EMD falls below the
  identity's 0.032, toward the full paired control's 0.0002.
- The unpaired data helps if the semi-paired run beats the paired-only run
  with the same K on the test set.

    LABEL=closure_semi1_s0 COST=shape_energy LAMBDA=1.0 SEED=0 MINUTES=120 \
        EVAL_ARGS="--setting 32:midpoint --train-n 2000" \
        sbatch -w dahlia scripts/flow/closure_run.sbatch \
        --pairing semi --paired-n 2000 --paired-share 0.5
    LABEL=closure_pk1_s0 COST=l2 SEED=0 MINUTES=120 \
        EVAL_ARGS="--setting 32:midpoint --train-n 2000" \
        sbatch -w dahlia scripts/flow/closure_run.sbatch \
        --pairing paired --paired-n 2000
    (closure_semi5_s0, closure_pk5_s0: the same with --paired-n 10000)

### Semi-paired results (jobs 20192-20195, 2026-09-26)

**Validation during training** (shape EMD, held-out validation / the seen
pairs A[:2000], 32 midpoint evaluations):
- **Semi-paired, 2k pairs:** flat at 0.086-0.091 on validation all run
  long, the unpaired level, while the seen pairs fall from 0.0030 to 0.0012.
- **Semi-paired, 10k pairs:** 0.022-0.024 over the first 30 min, then
  worse to 0.047 by 2 h, as the unpaired data takes over.
- **Paired only, 2k pairs:** best at 10 min (0.0009); it then overfits
  slowly, to 0.0025.
- **Paired only, 10k pairs:** 0.0003, stable.

Selected checkpoints (lowest validation EMD): 120, 20, 10 and 50 min.
Updates in 2 h: 80.3k and 80.4k for the semi-paired runs (the matching
takes 10% of the step), 88.2k and 88.4k for the paired-only ones.

**Test** (20k held-out pairs, 32 midpoint evaluations;
`docs/flow/closure_semi_m32.csv`, event displays
`docs/flow/closure_semi_jets.png`):

| | shape EMD, mean / median | E_out / E_in (0.8) | E_out - E_true, RMSE, GeV | mass change: bias / RMSE, GeV | girth change, RMSE | hardest towers: mean / rms, GeV |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 0.032 / 0.032 | 1 | 6.61 | +0.96 / 1.01 | 0.0044 | +4.05 / 4.45 |
| unpaired (0% pairs) | 0.091 / 0.083 | 0.808 +- 0.038 | 1.32 | +0.14 / 0.54 | 0.0146 | -0.87 / 2.16 |
| semi-paired, 2k pairs (1%) | 0.086 / 0.077 | 0.810 +- 0.037 | 1.32 | +0.18 / 0.51 | 0.0139 | -0.90 / 2.09 |
| semi-paired, 10k pairs (5%) | 0.022 / 0.006 | 0.818 +- 0.032 | 1.15 | +0.08 / 0.27 | 0.0053 | +0.11 / 0.78 |
| paired only, 2k pairs | **0.0009 / 0.0008** | 0.7999 +- 0.0033 | 0.13 | -0.001 / 0.020 | 0.0002 | -0.005 / 0.062 |
| paired only, 10k pairs | **0.0003 / 0.0003** | 0.7999 +- 0.0010 | 0.04 | -0.001 / 0.006 | 0.0001 | -0.004 / 0.023 |
| all 200k pairs | 0.0002 / 0.0001 | 0.7997 +- 0.0004 | 0.02 | -0.002 / 0.003 | 0.00004 | -0.006 / 0.014 |

**Seen against unseen** (shape EMD on 2000 training jets whose pair was
shown, against 2000 whose pair was not):

| | seen pairs | unseen training jets |
| :--- | ---: | ---: |
| semi-paired, 2k | 0.0012 | 0.087 |
| paired only, 2k | 0.0005 | 0.0009 |
| semi-paired, 10k | 0.0072 | 0.022 |
| paired only, 10k | 0.0002 | 0.0003 |

**Reading:**
- **A few true pairs alone restore the detail for this modification.**
  With 2000 pairs (1%) and no unpaired data, the per-jet shape error is 35x
  below the identity's, the energy is right to 0.13 GeV RMSE, and the mass
  and girth changes are right to 6% of their spread. Unseen jets are almost
  as good as the seen ones. 10k pairs nearly match the fully paired model.
  T is smooth and simple, and a few thousand examples pin it down.
- **Mixing those pairs with unpaired minibatch-OT pairs makes things worse,
  not better.**
  - With 2k pairs, jets whose pair was not shown come out as badly as fully
    unpaired (0.087): the network memorises the 2000 pairs (0.0012) and
    applies the blurring unpaired map everywhere else.
  - With 10k pairs, the best checkpoint (20 min) reaches 0.022, better than
    the identity. But the result is bimodal (median 0.006 against mean
    0.022; one of the four displayed jets is badly wrong), it keeps getting
    worse with training, and its energy runs 2% high.
- **Why.** For a jet, the unpaired OT targets (other, similar jets) and the
  true target conflict. The network cannot generalise the true
  correspondence while the unpaired data keeps pulling toward the blurred
  map.
- **So the unpaired data, used this way, adds nothing here and costs a
  lot.** Few pairs have to shape the coupling, not compete with it.

**Uncertain:**
- one seed;
- a deterministic, smooth toy. A real medium modification is random even
  for the same hard scattering, so a JEWEL vacuum/medium "pair" defines a
  distribution of modified jets, not one, and would need a stochastic
  (conditional) flow;
- the paired share of the batch (50%) is one choice. The natural 1-5% share
  was not run: the averaging argument predicts almost no effect;
- selection matters: the paired-only 2k run overfits after 10 min, and the
  semi-paired 10k run degrades after 20.

**Next justified step:** use the pairs to shape the unpaired coupling
rather than add them as competing targets. The first, simplest check is
already in hand: the paired-only model. The next one:
- train on the few pairs first;
- match the unpaired targets to that model's predictions F(J) by OT
  (pseudo-pairs consistent with the paired map);
- retrain;
- test in the same closure with 1% pairs.

For vacuum -> medium, the first question is whether paired JEWEL
vacuum/medium simulation is available, and how random its per-event
modification is.

## Teacher-guided coupling: can a few-pair teacher make unpaired data useful? (2026-09-26)

**Question.** Can the few pairs improve the unpaired coupling enough for the
unpaired data to add something? The baseline to beat is now the 2k-pair
model (shape EMD ~0.0009), not unpaired-only (0.091).

**Frozen teacher** (`closure_teacher.py`):
- `closure_pk1_s0`, the paired-only model trained on the first 2000
  source jets and their T(J) (1% of the 200k training jets), at its
  validation-selected checkpoint (update 7350), EMA network, frozen
  throughout;
- its endpoints F_teacher(x) come from the accurate solve (32 midpoint
  evaluations), in GeV and clipped at 0 like every closure output;
- they are cached once for all 200k training sources (588 s on one A6000,
  2.9 ms per jet) and for the validation pairs;
- no pairs beyond these 2000 are used anywhere. Hidden T(x) outside them
  serves only to score, and to build the audit's oracle.

**The guided coupling:**
- Only the reference of the cost changes, from C_old(i, j) = d(x_i, y_j) to
  C_guided(i, j) = d(F_teacher(x_i), y_j).
- d is the existing shape + log-energy cost (lambda 1, frozen scales).
- The solver is the existing balanced exact OT (TorchCFM's `pot.emd`), on
  the baseline pool of 256 sources x 256 real targets from T(B).
- 256 complete pairs are drawn from the plan with TorchCFM's `sample_map`.
- The student still flows from the original x to the selected real y; the
  teacher only chooses y.

### Audit before training (`closure_guided_audit.py`)

1024 validation sources in 4 batches of 256, each matched against 4
independent batches of 256 real targets from T(B) (other events, so the
true counterpart is never present). Scored against the hidden T(x);
`docs/flow/closure_guided_audit.csv`, figure
`docs/flow/closure_guided_audit.png`.

| what the source is matched from | matched target's shape EMD to T(x): mean / median / p90 | energy RMSE, GeV | shape EMD between the targets chosen in different batches |
| :--- | ---: | ---: | ---: |
| (the teacher's prediction itself) | 0.0010 / 0.0008 / 0.0016 | 0.14 | |
| (identity x) | 0.032 / 0.032 / 0.039 | 6.6 | |
| original, d(x, y) | 0.214 / 0.200 / 0.312 | 1.86 | 0.249 |
| guided, d(F_teacher(x), y) | 0.214 / 0.199 / 0.309 | 1.80 | 0.247 |
| perfect teacher, d(T(x), y) | 0.213 / 0.199 / 0.308 | 1.79 | 0.247 |
| perfect teacher, nearest target without the one-to-one plan | 0.196 / 0.184 / 0.276 | 1.42 | 0.216 |

**The concern is confirmed.**
- Even a perfect teacher must be matched to a different real event. The
  best of 256 real targets is 7x further from T(x) than the unchanged jet,
  and 200x further than the teacher's own prediction.
- Guidance barely changes the targets chosen (0.214 against 0.2135). The
  limit is the pool, not the reference: in two of the three displayed
  cases, all four references pick the same jet.
- The selected target changes completely from one minibatch to the next.
- Matching error is not a lower bound on the trained flow's error, but the
  pilot was kept short, as planned.

### Continuation controls (`closure_continue.py`)

**Runs:**
- **A:** the frozen teacher.
- **B:** A's weights continued with paired-only flow matching.
- **C:** A's weights continued with paired plus guided-unpaired flow
  matching.

**Held identical between B and C:**
- the starting weights: the teacher's EMA weights, both as the network and
  as the starting EMA (its warm-up continues from update 7350);
- Adam 2e-4 (fresh, 1000 warm-up steps), gradient clip 1, EMA 0.9999;
- ADM U-Net, standardised psi, straight path sigma 0, accurate solve for
  evaluation;
- per update, the same paired term: 256 true pairs from the same 2000
  (A[:2000]), with the same pair-index and time sequences (their own
  generators), T computed on the fly, paired-loss coefficient 1.

**C's only addition:** weight 1 x the mean flow-matching loss of 256 pairs
from the guided plan (their own generators). Each term is its own mean, so
the paired gradient is not diluted.

**Supervision exposure:**
- the unique-pair budget is the same 2000 pairs (1%) for A, B and C;
- A saw 7350 x 256 = 1.88M paired examples;
- the selected B and C checkpoints each add 2000 x 256 = 0.51M paired
  examples;
- in C the pairs are 50% of the examples per update and 50% of the loss
  weight, and in B 100%.

**Budget:** 12000 updates each (pilot kept short after the audit),
checkpoints every 2000 updates, monitored on 1000 validation pairs. A run
stops if the monitor exceeds 3x the teacher at three checkpoints in a row.
Jobs 20200 and 20201.

**Cost:**
- B: 12000 updates in 16.3 min (12.2 /s).
- C: 6.0 updates/s. Per update, the paired term takes 26 ms, the teacher
  lookup 0.3 ms (cached), matching 6.4 ms, and the unpaired forward and
  backward 133 ms.
- Plus the 588 s teacher cache, once.

**Validation during training** (shape EMD, validation / the 2000 seen
pairs):
- B: 0.00093 / 0.00042 at 2000 updates; 0.00119 / 0.00028 at 12000.
  It slowly overfits the 2000 pairs.
- C: 0.078 / 0.022 at 2000 updates, 0.094 / 0.0035 at 4000, 0.098 / 0.0030
  at 6000. It was stopped by the rule.
- Both selected checkpoints are at 2000 updates, so the comparison is at an
  equal number of updates.

**Test** (20k held-out pairs, 32 midpoint evaluations, identical clip at 0,
no recalibration; `docs/flow/closure_teacher_abc_m32.csv`, event displays
`docs/flow/closure_teacher_abc_jets.png`):

| | identity | A: teacher | B: + paired-only continuation | C: + guided unpaired |
| :--- | ---: | ---: | ---: | ---: |
| shape EMD: mean / median / p90 / p99 | 0.032 / 0.032 / 0.039 / 0.047 | **0.0009 / 0.0008 / 0.0016 / 0.0034** | 0.0009 / 0.0008 / 0.0016 / 0.0035 | 0.079 / 0.072 / 0.120 / 0.192 |
| E_out / E_in (toy truth 0.8) | 1 | 0.7999 +- 0.0033 | 0.7999 +- 0.0035 | 0.787 +- 0.050 |
| E_out - E_true: bias / RMSE, GeV | +6.5 / 6.6 | -0.010 / 0.13 | -0.012 / 0.14 | -0.57 / 1.87 |
| mass change: bias / RMSE, GeV (truth -0.96 +- 0.31) | +0.96 / 1.01 | -0.001 / 0.020 | -0.001 / 0.020 | +0.35 / 0.49 |
| girth change: bias / RMSE (truth +0.0032 +- 0.0030) | -0.0032 / 0.0044 | 0.0000 / 0.0002 | 0.0000 / 0.0002 | +0.0047 / 0.0107 |
| marginals W1 / sigma: E, mass, girth, p_T^D, z_lead | 1.24, 0.92, 0.08, 0.67, 0.49 | 0.005, 0.003, 0.001, 0.002, 0.002 | 0.005, 0.003, 0.001, 0.002, 0.002 | 0.12, 0.33, 0.15, 0.46, 0.32 |
| towers T > 5 GeV: error mean / rms, GeV | +4.05 / 4.45 | -0.005 / 0.062 | -0.008 / 0.062 | **-1.55 / 2.20** |
| towers T < 0.2 GeV: error mean / rms, GeV | -0.023 / 0.048 | 0.000 / 0.001 | 0.000 / 0.001 | **+0.043 / 0.084** |

**Result: the guided unpaired data adds nothing beyond the few-pair teacher,
and it destroys the teacher's event-level fidelity.**
- The guided term brings back the core flattening of unpaired training
  (hard towers 1.55 GeV low, soft towers inflated) within 2000 updates.
- C recovers the 2000 seen pairs (0.003) and blurs every other jet, as the
  semi-paired runs did.
- Continuing paired-only adds nothing either: B equals A within noise, and
  slowly overfits.
- The reason is the audit: at the baseline pool, even a perfect teacher can
  only point to real jets that differ from T(x) in their fine structure. The
  guided coupling therefore teaches the same averaged, blurred map as the
  original one.
- **As agreed, this toy experiment stops here.** Only one guidance weight
  (1) was run; smaller weights would only interpolate between the teacher
  and C.

**What this establishes, for this toy only:**
- with few true pairs, the paired-only flow is the best event-level model;
- unpaired data, coupled by minibatch OT of real jets (guided or not), has
  no useful role in it;
- unpaired data could still help where the pairs do not cover the
  distribution, or through a mechanism other than endpoint coupling. That
  was not tested.

## Physical pairs: what JEWEL and HYBRID can provide (feasibility, 2026-09-26)

**Sources:**
- JEWEL 2.0.0, "Directions for use of JEWEL" (K. C. Zapp, EPJC 74 (2014)
  2762, arXiv:1311.0048), read in full;
- the hybrid strong/weak coupling model (Casalderrey-Solana, Gulhan,
  Milhano, Pablos, Rajagopal, JHEP 10 (2014) 019, arXiv:1405.3864;
  JHEP 03 (2017) 135);
- the EuCAIFCon 2026 contribution by Goncalves, Pablos, Flek and Schott,
  "The Low-Level Inverse Jet-Quenching Problem" (Indico event 1277,
  contribution 4223). It is a conference report and a lead, not a
  validated application.

**What is installed here: nothing.**
- No JEWEL or HYBRID code on this cluster.
- The JEWEL sample we use (Zenodo record 17594612, `jewel_jet30`) holds
  only one calorimeter image per event (ROOT TH2), with no generator
  record, seeds, version or parameters.
- The JEWEL website blocked automated access, so versions after 2.0.0 were
  not checked.

**Three levels of correspondence:**
1. the same hard scattering (the 2 -> 2 partons and the initial-state
   shower);
2. the same realised vacuum shower (the full parton cascade before
   hadronisation);
3. a medium modification of that same shower.

**JEWEL 2.0.0, as documented:**
- **Structure (section 3.2):**
  - JEWEL first sets the geometry (impact parameter, jet production point);
  - PYTHIA 6.4 generates the matrix element and the initial-state shower;
  - JEWEL generates "the final state parton shower including possible
    interactions in a medium";
  - strings are built, and PYTHIA hadronises.
  - Radiation and medium rescattering are interleaved in one evolution;
    vacuum runs are a separate executable (`jewel-*-vac`) linking a
    no-medium model.
- **Levels 2 and 3 do not exist in JEWEL.** A medium event is not a
  modification of a vacuum shower; it is a different stochastic evolution.
- **Level 1 is not available out of the box either:**
  - one job seed (`NJOB`) initialises the random numbers;
  - no per-event state saving and no reading of external hard scatterings
    is documented;
  - the geometry is set before the hard scattering, so identical seeds need
    not give identical hard scatterings in the vacuum and medium
    executables. This is an inference to check by running both.
- **The standard output cannot verify ancestry.** HepMC 2 keeps only the
  hadronic stage: intermediate particles are deleted before hadronisation
  (`COMPRESS`, `SHORTHEPMC`), and recoils are dropped unless `KEEPRECOILS`
  is set.
- **What level-1 pairs would take:**
  - a code change that saves the PYTHIA 6 state (or the hard-scattering
    record) after the matrix element and initial-state shower, and replays
    it into both final-state showers;
  - verification from an uncompressed parton record.
  - Even then, the vacuum and the medium showers are independent random
    evolutions of the same hard scattering. The pairing is conditional,
    and every shower fluctuation lies between the two jets.
- **Multiple medium realisations** of one hard scattering are possible with
  such a change.

**HYBRID:**
- **Structure:**
  - PYTHIA 8 generates the vacuum parton shower;
  - each parton gets a formation time (tau = 2E/Q^2) and a path through a
    hydrodynamic medium from a Glauber-sampled production point, and loses
    energy at the holographic strong-coupling rate;
  - later versions add medium response (Cooper-Frye) and elastic Moliere
    scatterings;
  - hadronisation is PYTHIA's.
- **Level 3 exists by construction:** the medium jet is the same realised
  vacuum shower (level 2) with per-parton energy loss. "Each quenched shower
  is obtained by modifying a known vacuum shower and therefore provides
  direct jet-by-jet supervision" (EuCAIFCon 2026).
- **What differs between the paired outputs:**
  - the production point and orientation in the medium;
  - the sampling of the medium response;
  - Moliere kicks, where enabled;
  - hadronisation, which is a separate string fragmentation of the
    modified partons.
- **Several medium realisations of one vacuum shower** come from
  re-sampling these. The target is then a conditional distribution
  p(medium jet | vacuum shower), not a deterministic map, and its width can
  be measured directly.
- **Parent showers must be disjoint** between training and evaluation.
- **Availability:** no public release of the HYBRID code was found. Access
  would go through the authors, for example D. Pablos, who co-authored the
  report. This is unconfirmed until the code, or a small paired sample, is
  obtained.

**Conclusions:**
- JEWEL, as documented, cannot supply shower-level pairs; at most
  hard-scattering-level pairs with a code change, and those leave the whole
  shower unpaired.
- HYBRID can supply shower-level pairs by design. With a stochastic medium,
  the realistic target is a conditional distribution.
- Nothing here validates the physics of such pairs. The claims in this file
  about pairs stay restricted to the toy until a paired physical sample
  exists.

    python scripts/flow/closure_teacher.py --run closure_pk1_s0     # cache, 588 s
    python scripts/flow/closure_guided_audit.py                      # audit
    python scripts/flow/closure_continue.py --label closure_cont_paired_s0 \
        --mode paired --updates 12000 --ckpt-updates 2000
    python scripts/flow/closure_continue.py --label closure_cont_guided_s0 \
        --mode guided --updates 12000 --ckpt-updates 2000             # stopped at 6000
    python scripts/flow/closure_eval.py closure_cont_paired_s0 closure_cont_guided_s0 \
        --select --setting 32:midpoint --train-n 2000
    python scripts/flow/closure_eval.py closure_pk1_s0 closure_cont_paired_s0 \
        closure_cont_guided_s0 --control --setting 32:midpoint \
        --out outdir/sphenix/flow/closure_teacher_abc_m32

## Consolidated benchmark: background-only against joint subtraction (2026-09-27/28)

**Question.** For M = S + B, is it better to predict the background B alone
and read the signal as S_hat = M - B_hat, or to predict both components?
What does training each mixture with its own background (simulation truth)
change? Judged on individual jets (core, substructure), not only on the
average energy resolution. This closes the study with one controlled
comparison and a short deck; no new model proposals.

**Deliverables:** `docs/flow/bench/slides/bench_deck.pdf` (8 slides + 2
backup; source `bench_deck.tex`, `body.tex`, generated `tables/*.tex`),
tables and figures in `docs/flow/bench/` (index: `docs/flow/bench/README.md`),
run manifest `bench_runs.csv` and configurations `configs/*.json`.

### Design

**Three flow arms, one network, one budget.**
- **Common to all three:**
  - the UVCGAN-S generator as velocity network (`--backbone uvcgan`, as in
    the backbone ablation: ViT-ModNet, 32.3M parameters, time added to the
    style token);
  - log(E + 0.1) standardised states; the straight path (sigma 0) and the
    CFM velocity loss;
  - Adam 2e-4 with 1000 warm-up steps, gradient clip 1, EMA 0.9999, batch
    256;
  - 2 h on one RTX A6000 (dahlia), checkpoints every 15 min, 3 seeds;
  - deterministic transports from the mixture, not posterior samplers.

| arm | method | source -> target | pairing | loss | readouts |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **unpaired FM** | `otcfm1` | M -> B | exact minibatch OT between the batch's real mixtures and independently drawn HIJING events | CFM MSE | S_hat = M - B_hat |
| **paired FM** | `otcfm1_paired` | M -> B | each mixture with its own background, B = M - S (bookkeeping) | CFM MSE | S_hat = M - B_hat |
| **joint FM** | `joint_paired` | (M, 0) -> (B, S) | each mixture with its own (B, S) | mean over B + mean over S, weight 1 each; both logged (`loss_bkg`, `loss_sig` in `history.csv`) | direct S_hat, and M - B_hat, reported separately |

- **Controls.**
  - Unpaired and paired FM differ only in the pairing: the same
    initialisation, preprocessing, path, loss, optimiser, batch and budget.
  - The joint arm is aligned with them. Its extra cost (a second output
    channel and its loss) is reported, not equalised: 961 more parameters,
    < 1% fewer updates per second, ~1% slower inference.
- **Training data.**
  - The same 633k real embedded mixtures (PYTHIA in HIJING, `embed` domain)
    for every arm.
  - The paired arms read each mixture's detector-level signal by its index
    key (`make_pairs.py`, `OUTDIR/sphenix/flow/cache/train_embed_pairs.npy`).
    B = M - S >= 0 exactly (max S - M = 0, min B = 0 in float32), and no
    training key is a val key.
  - The unpaired arm draws its targets from the 986k HIJING events.
- **Runs.**
  - `bb_uvcgan_otcfm1_s0` (backbone ablation) is seed 0 of the unpaired
    arm: same code path, configuration and data.
  - New runs: `bb_uvcgan_otcfm1_s1,s2`, `bench_paired_bkg_s0-s2`,
    `bench_joint_s0-s2` (jobs 20205-20211, 20213). They ran from the working
    tree of commit 544c373 with the changes committed with this section.
- **The legacy "true pieces" run (`otcfm_pieces`, `ext_pieces_s*`) is not
  reused.**
  - It is a joint two-panel flow (mixture, 0) -> (B, S) whose OT batch
    recovers the true pieces, so effectively paired.
  - But: ADM U-Net backbone, synthetic mixtures B + S, the loss averaged
    over both channels (half weight each), selected at 16 midpoint
    evaluations on the direct readout.
  - Its outcome is only a prior indication: a clean image, poor and
    unstable jet energies (5.1-5.3 GeV).
- **Paired training is a supervised simulation reference.** It is not an
  unpaired result and not a guaranteed upper bound: it needs the true
  background of every training mixture, which exists only in simulation.
  UVCGAN-S's `idt-aa` term uses the same kind of information (known additive
  mixing of simulated components).

**Selection and solvers.**
- **Checkpoint:** one per run, the lowest val `jer_cal` of the EMA network at
  4 Euler steps with the M - B_hat readout (the selection readout fixed for
  `otcfm1` before this study). Every observable uses that checkpoint; JEWEL
  is scored only there.
- **Solves:** 4 Euler steps (the selection readout) and the accurately
  resolved midpoint solve with 32 network evaluations, checked against 64 on
  val. Both are reported for every arm and readout.
- **Main figures:** each readout at its val-preferred solve (val `jer_cal`
  at the selected checkpoint): 4 Euler steps for every readout except the
  joint arm's direct S_hat (32 NFE: 4.50-4.66 against 5.61-6.11 GeV). The
  backup slide has the full common-solver comparison.

**References on the same events.**
- **UVCGAN-S, published checkpoint** (EMA generator, one pass):
  - batch 4, lr 5e-5, 800k updates (~105 h on an A6000);
  - trained on unpaired HIJING and PYTHIA images and real mixtures, with
    `idt-aa` (the generator decomposes synthetic sums of an independent
    background and signal, L1 to the pieces: supervised on synthetic
    mixtures), `idt-bb`, cycle consistency and three adversarial losses;
  - retrained from scratch (3 seeds each at batch 4 and batch 32,
    `SCALING_NOTES.md`) for the training-time comparison.
- **Area** (FastJet 3.5.1):
  - anti-kT jets of the mixture with active area, pT - rho A;
  - rho is the median of kT (R 0.4) jets in |y| < 0.7 without the two
    hardest;
  - no substructure.
- ICS needs fjcontrib, not available here.

**Events.**
- **Development:** the 20k val PYTHIA+HIJING mixtures of every earlier score
  (`eval_val_truth.load_pairs`, seed-0 draw).
- **Frozen test:** the 20k JEWEL+HIJING test mixtures of every earlier JEWEL
  score, scored only after all choices were fixed.
- **Frozen calibration:** fitted per run on the matched jets (or cones) of
  val events 10000-19999 and scored on events 0-9999 of val and of JEWEL. On
  JEWEL it therefore carries the domain shift.

**Analysis (paper arXiv:2510.23717v2, Figs. 3-7).** No analysis code is
public (LS4GAN/uvcgan-s holds training code only), so `bench_jets.py`
implements the paper's definitions minimally:
- FastJet anti-kT, R = 0.2 / 0.4 / 0.5;
- towers as massless constituents at their centres (E_T = tower value,
  towers <= 0 dropped); jets > 5 GeV, |eta| < 0.6;
- truth jets from the detector-level signal image;
- one-to-one matching, closest first, dR < 0.75 R, all jets;
- soft drop z_cut 0.1, beta 0 on C/A; substructure at R = 0.4, 20-30 GeV;
- our choices where the paper is silent: the tower threshold, greedy
  matching, the fake rate binned in pT_sub, C/A reclustering for soft drop.

Figures: `bench_fig3/4/5/6_{val,jewel}.png`, `bench_fig5_allR_*.png`,
`bench_fig7.png` (the paper's figure for truth and UVCGAN-S plus the
JEWEL/PYTHIA ratio of every method); `mid32/` has the same figures with the
accurate solve.

**Diagnostics beyond the paper.**
- per matched jet: pT response, bias and RMSE in GeV; per-jet bias and RMSE
  of z_g, r_g, girth, mass, z_lead and p_T^D;
- distribution agreement (W1 against all truth jets, with a truth-vs-truth
  floor from even/odd events), kept apart from per-jet agreement;
- per tower: MAE, RMSE, event energy bias; background error B_hat - B by the
  true signal energy of the tower (`bench_diag.py`);
- joint arm: B_hat + S_hat - M before any clean-up;
- event displays: fixed events (the first two of each set with a 25-35 GeV
  leading truth jet), common scales;
- where the paired flows' training signal lies: the CFM loss against t on
  held-out pairs (`bench_loss_t.py`);
- true-axis cone scores (`jet_fidelity.py`, images kind), supplementary:
  frozen-calibration energy resolution, per-jet observables, EMD and its
  shape part;
- cost: val `jer_cal` against training hours, time to T_acc, throughput,
  peak memory, inference latency on one A6000 (`bench_cost.py`).

**Uncertainties.** Statistical errors of one run on these events (binomial
or standard errors, `_se` columns) are kept apart from the seed spread (half
range of 3 seeds). Figures show seed 0 with its statistical errors and the
seed range as a band.

**Clean-up.** Raw outputs are primary. The only clean-ups, labelled and
reported separately, are a 0.5 GeV tower threshold on every output
(`[thr0.5]`, truth unchanged) and the frozen linear calibration (`_cal`).

### Results (jobs 20205-20213 training, 20220-20228 scoring and images, 20229 latency, 20230-20237 analysis)

**Selection metric, val `jer_cal` (GeV) at the selected checkpoints**
(seeds 0 / 1 / 2; UVCGAN-S published 3.59, JEWEL 3.99):

| arm, readout | selected at | val, 4 Euler | val, 32 NFE (64, seed 0) | JEWEL, 4 Euler | JEWEL, 32 NFE |
| :--- | :--- | ---: | ---: | ---: | ---: |
| unpaired, M - B_hat | 90 / 105 / 120 min | 3.66 / 3.68 / 3.67 | 4.10 / 4.11 / 4.09 (4.10) | 3.55 / 3.56 / 3.54 | 3.64 / 3.66 / 3.65 |
| paired, M - B_hat | 30 / 30 / 45 min | 3.95 / 3.95 / 3.95 | 4.10 / 4.11 / 4.12 (4.10) | 3.60 / 3.61 / 3.61 | 3.65 / 3.66 / 3.67 |
| joint, M - B_hat | 105 / 120 / 120 min | 4.14 / 4.12 / 4.19 | 4.33 / 4.23 / 4.32 (4.34) | 4.15 / 4.16 / 4.05 | 4.39 / 4.36 / 4.22 |
| joint, direct S_hat | (same) | 5.61 / 5.68 / 6.11 | 4.62 / 4.50 / 4.66 (4.52) | 5.06 / 5.07 / 5.44 | 4.48 / 4.40 / 4.31 |

- **The accurate solve is resolved.** 64 NFE matches 32 to 0.01 GeV,
  except for the joint direct readout (0.1 GeV).
- **Only the unpaired arm reaches T_acc (3.70 GeV)**, after 0.5 h (all
  seeds).
- **The paired arm is best after 30-45 min and then worsens** (4.01-4.02 at
  2 h) while its per-tower MAE keeps improving.
- **The joint arm is still improving at 2 h.**

**Jets** (`bench_summary.csv`, R = 0.4, 20 < pT_real < 30 GeV, seed means;
seed half ranges: background-only flows <= 0.002 in scale and <= 0.05 GeV in
RMSE, joint arm up to 0.023 and 0.4 GeV):

| | val scale | val res. (frozen cal.) | val RMSE, GeV | val eff. 14-20 | val fake 14-20 | JEWEL scale | JEWEL scale (val cal.) | JEWEL res. (val cal.) | JEWEL fake 14-20 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UVCGAN-S | 0.982 | 0.153 | 3.59 | 0.972 | 0.030 | 1.022 | 1.046 | 0.151 | 0.080 |
| Area | 0.881 | 0.261 | 7.37 | 0.865 | 0.192 | 0.904 | 1.016 | 0.269 | 0.231 |
| unpaired FM | 0.993 | 0.150 | 3.33 | 0.988 | 0.146 | 1.033 | 1.040 | 0.151 | 0.189 |
| paired FM | 0.828 | 0.154 | 5.51 | 0.968 | 0.005 | 0.863 | 1.040 | 0.158 | 0.015 |
| joint FM, M - B_hat | 0.912 | 0.161 | 4.41 | 0.881 | 0.029 | 0.981 | 1.072 | 0.154 | 0.075 |
| joint FM, direct S_hat (32 NFE) | 0.884 | 0.164 | 5.02 | 0.846 | 0.020 | 0.963 | 1.079 | 0.161 | 0.062 |

- **The radius separates the failure modes.** Scale at R = 0.2 / 0.4 / 0.5:

  | | R = 0.2 | R = 0.4 | R = 0.5 |
  | :--- | ---: | ---: | ---: |
  | UVCGAN-S | 0.99 | 0.98 | 0.98 |
  | unpaired FM | 0.88 | 0.99 | 1.10 |
  | paired FM | 0.82 | 0.83 | 0.85 |
  | joint FM, M - B_hat | 0.99 | 0.91 | 0.89 |

  - Unpaired FM's floor fills larger cones: its fake rate at 14-20 GeV is
    67% at R = 0.5 and 82% at 14-16 GeV.
  - The joint arm keeps the core and loses the periphery: its efficiency at
    14-20 GeV is 0.81 at R = 0.5.
- **The accurate solve:**
  - lowers both background-only scales to 0.74-0.75 (val);
  - removes the unpaired floor (fakes 0.146 -> 0.002);
  - leaves the calibrated resolution about the same (0.155-0.157).
- **The 0.5 GeV threshold** removes fakes but lowers every scale by a
  further 11-27% (UVCGAN-S 0.87, paired 0.60). It restores no core.

**Substructure** (val, R = 0.4, 20-30 GeV; in units of the spread of all
truth jets; per-jet: matched pairs):

| | per-jet RMSE / sigma: g, m, z_lead, p_T^D, z_g, r_g | distribution W1 / sigma: g, m, z_lead, p_T^D, z_g, r_g |
| :--- | :--- | :--- |
| UVCGAN-S | 0.60, 1.16, 0.38, 0.46, 1.12, 0.94 | 0.07, 0.21, 0.04, 0.06, 0.01, 0.02 |
| unpaired FM | 0.79, 1.31, 0.48, 0.72, 1.12, 0.97 | 0.59, 0.68, 0.35, 0.60, 0.22, 0.38 |
| paired FM | **0.55**, 1.17, **0.34**, 0.51, 1.12, **0.85** | 0.22, 0.53, 0.14, 0.36, 0.13, 0.13 |
| joint FM, M - B_hat | 0.68, 1.31, 0.44, 0.47, 1.18, 1.04 | 0.39, 0.76, 0.22, 0.17, 0.02, 0.50 |
| joint FM, direct S_hat (32 NFE) | 0.99, 1.67, 0.81, 0.87, 1.22, 1.23 | 0.79, 1.30, 0.64, 0.68, 0.05, 0.77 |
| truth vs truth (floor) | | 0.03, 0.03, 0.02, 0.02, 0.01, 0.03 |

- **UVCGAN-S has the closest distributions**, 2-7x closer than the best
  flow for every observable.
- **Paired FM has the smallest per-jet errors** of girth, z_lead and r_g.
  Its distributions are shifted: low mass, high xi (the lost core lowers the
  jet pT).
- **Per-jet z_g is not reproduced by any model** (RMSE ~1.1 sigma).
- **Seed half ranges:** <= 0.01 sigma for UVCGAN-S and the background-only
  flows, up to 0.14 for the joint arm.

**JEWEL** (the same selection; the JEWEL - PYTHIA shift of the mean as a
fraction of the truth's, stat. error 0.02-0.04):

| | girth | z_lead | mass | p_T^D | r_g |
| :--- | ---: | ---: | ---: | ---: | ---: |
| UVCGAN-S | 0.76 | 0.80 | 0.55 | 0.76 | 0.70 |
| unpaired FM | 0.63 | 0.71 | 0.36 | 0.66 | 0.46 |
| paired FM | 0.72 | 0.80 | 0.36 | 0.79 | 0.69 |
| joint FM, M - B_hat | 0.63 | 0.73 | 0.23 | 0.72 | 0.71 |
| joint FM, direct S_hat (32 NFE) | 0.63 | 0.80 | 0.22 | 0.80 | 0.71 |

- **The frozen PYTHIA calibration carries over** with a +4% scale shift for
  UVCGAN-S and the background-only flows and +7-8% for the joint arm. The
  calibrated resolutions stay at 0.151-0.161.
- **The joint M - B_hat readout does better on JEWEL than on PYTHIA:** scale
  0.98, the smallest per-jet girth error (0.57 sigma; UVCGAN-S 0.63). This
  is consistent with its keeping hard towers, since JEWEL jets are narrower.

**Towers and where the background error goes** (val, seed means; shares of
the true signal energy of the towers put into B_hat = M - S_hat):

| | MAE / tower, GeV | S = 0: B_hat - B, GeV | S 0.5-2 | S 5-10 | S > 10 | sum(S_hat - S), GeV / event |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| UVCGAN-S | 0.033 | -0.009 | 48% | 2% | 1% | -7 |
| unpaired FM | 0.150 | -0.090 | 38% | 17% | 12% | +112 |
| unpaired FM, 32 NFE | 0.051 | -0.026 | 66% | 33% | 24% | +1 |
| paired FM | 0.050 | -0.026 | 61% | 24% | 16% | +8 |
| paired FM, 32 NFE | 0.051 | -0.026 | 65% | 31% | 23% | +2 |
| joint FM, M - B_hat | 0.031 | -0.006 | 61% | 3% | -1% | -19 |
| joint FM, direct S_hat (32 NFE) | 0.031 | -0.003 | 66% | -4% | -14% | -25 |

- **Background-only prediction puts part of the jet core into B_hat**,
  paired or not. Joint prediction and UVCGAN-S (also a two-output
  generator) keep it; the joint direct readout overshoots the hardest
  towers.
- **Every model misses 20-60% of the signal in towers with S < 2 GeV.**
- **Joint consistency, B_hat + S_hat - M before clean-up (val):**
  - 4 Euler steps: MAE 0.009 GeV per tower, -7.7 GeV per event;
  - 32 NFE: 0.003 GeV per tower, +1.3 GeV per event.
- The JEWEL shares are within 1-3 points of these (`bench_towers_summary.csv`).

**Where the joint arm's training signal lies** (`bench_loss_t.csv`,
selected checkpoints, 2000 val pairs):
- The joint flow's signal-channel loss is 0.51 at t = 0 and 0.0015 at
  t = 0.02; its background-channel loss is 0.042 -> 0.003.
- The paired one-panel flow's loss rises from 0.040 at t = 0 to 0.081 at
  t = 0.95.
- With the source (M, 0) the signal channel of x_t = (1 - t) z(0) + t z(S)
  exposes S, and with it B, for any t > 0. The joint flow is therefore
  decided in its first step, and almost all of its training samples carry
  no loss.
- This explains its 10-40x smaller training loss. It is consistent with
  its slower convergence and larger seed spread; that link was not tested
  separately.

**Cost** (one RTX A6000; `bench_cost*.csv`, `bench_cost.png`):

| | best val `jer_cal` | at 2 h | h to 3.70 | updates / s | peak GB | ms / event, 4 NFE | 32 NFE |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| unpaired FM | 3.67 | 3.67 | 0.5 / 0.5 / 0.5 | 4.45 | 19.1 | 1.08 | 8.8 |
| paired FM | 3.95 | 4.02 | never | 4.55 | 18.1 | 1.10 | 8.9 |
| joint FM | 4.15 | 4.19 | never | 4.52 | 18.1 | 1.11 | 8.9 |
| UVCGAN-S retrained, batch 4 | 3.64 | 7.4 | 14.7 / 16.5 / 16.7 | | | | |
| UVCGAN-S retrained, batch 32 | 3.62 | 8.3 | 8.4 / 8.6 / 15.2 | | | | |
| UVCGAN-S published (~105 h) | 3.59 | | | | | 0.27 (1 pass) | |

**Supplementary true-axis cone scores** (`bench_cone*.csv`,
`bench_cone_s0.png`; 10k events per set, frozen calibration from val). Mean
EMD to the true jet, GeV, val / JEWEL (shape part in brackets):

| | EMD, val | EMD, JEWEL |
| :--- | ---: | ---: |
| UVCGAN-S | 4.75 (3.12) | 4.54 (2.83) |
| unpaired FM | 5.36 (3.13) | 4.77 (3.07) |
| paired FM | 8.24 (**2.93**) | 6.83 (**2.59**) |
| joint FM, M - B_hat | 5.69 (3.56) | 4.80 (2.86) |
| joint FM, direct S_hat, 32 NFE | 6.49 (4.33) | 5.46 (3.40) |

- Paired FM has the best shape and the worst energy.
- The frozen-calibration cone resolution on val is UVCGAN-S 3.57, unpaired
  3.65, paired 3.95, joint 4.14 GeV. On JEWEL it is 4.16 / 3.58 / 3.76 /
  4.32.

### Verdict

- **No flow arm is as faithful as UVCGAN-S** in jet scale, fakes and
  substructure distributions. After a frozen calibration, UVCGAN-S and the
  background-only flows have the same jet energy resolution (0.150-0.154 at
  20-30 GeV); the joint flow is 5-7% worse.
- **Predicting B alone loses part of the jet core, paired or not.**
  12-25% of the energy of towers with S > 10 GeV goes into B_hat.
  - Paired FM: scale 0.83.
  - Unpaired FM (4 steps) looks unbiased only because a soft floor adds
    +112 GeV per event: 15% fake jets at 14-20 GeV and the worst
    distributions.
  - Good average resolution coexists with this loss.
- **Predicting B and S keeps the hard core** but drops soft signal and low-pT
  jets. In this set-up it also varies more between seeds, and it is decided
  in its first step.
- **Neither dominates.**
  - Background-only (paired) gives the best per-jet substructure and the
    fewest fakes, with a low scale.
  - Joint gives a better scale at small R and on JEWEL, with lower
    efficiency.
- **Pairing improves the tower accuracy (3x), fakes (30x) and per-jet
  substructure. It does not fix the core loss.** It needs per-mixture
  simulation truth.
- **UVCGAN-S's fidelity comes after ~105 h and with its own supervised
  term.** The flows train in 2 h and match its calibrated resolution, not
  its scale or distributions.

**Missing or limited:**
- the ICS baseline (fjcontrib);
- the paper's analysis code (rebuilt from the text, so no paper numbers to
  compare);
- the val events' signal images are among UVCGAN-S's unpaired training
  signals;
- HIJING parents of the embedded mixtures cannot be checked for train/val
  overlap (the mixture keys are disjoint);
- the JEWEL sample (Zenodo 17594612) has no generator version or settings
  recorded (the paper states JEWEL 2.3.0 + Geant4);
- one network and a 2-h budget for the flows;
- the joint arm is still improving at 2 h.

**Commands** (also `docs/flow/bench/README.md`):

    $PYTHON scripts/flow/make_pairs.py              # training pairs (once)
    sub() { METHOD=$1 LABEL=$2 MINUTES=120 SEED=$3 EVAL_DECODE=mixture \
        EVAL_ARGS="--nfe 4 --solver euler" sbatch -w dahlia --time=03:30:00 \
        -J $2 scripts/flow/fm_run.sbatch --ckpt-minutes 15 --inline-events 1000 \
        --inline-nfe 4 --backbone uvcgan; }
    sub otcfm1 bb_uvcgan_otcfm1_s1 1; sub otcfm1 bb_uvcgan_otcfm1_s2 2
    for s in 0 1 2; do sub otcfm1_paired bench_paired_bkg_s$s $s; done
    for s in 0 1 2; do sub joint_paired bench_joint_s$s $s; done
    # per run: cone scores (val, JEWEL; 4 Euler, 32 NFE; 64 on val for
    # seed 0) and benchmark images
    LABEL=bench_joint_s0 JOINT=1 CONVERGENCE=1 sbatch -w dahlia \
        --dependency=afterok:JOB scripts/flow/bench_post.sbatch
    $PYTHON scripts/flow/bench_images.py --uvcgan         # UVCGAN-S images
    $PYTHON scripts/flow/fm_eval.py --latency RUN_DIRS --nfe 4,32 --solver euler,midpoint
    sbatch -p a6k -w saturn -c 64 --mem=160G scripts/flow/bench_all.sh
    $PYTHON scripts/flow/bench_tables.py
    cd docs/flow/bench/slides && ~/pyext/tectonic_env/bin/tectonic bench_deck.tex

**Tools installed for this benchmark (outside the env):**
- FastJet 3.5.1 Python bindings: `pip --target ~/pyext/jets`, its numpy,
  awkward and other dependencies removed, so the env's numpy 2.3.1 is used.
- Tectonic 0.17.0 and poppler in the conda environment
  `~/pyext/tectonic_env`.

## alpha-DSBM closure test: does a learned (Schrodinger bridge) coupling keep each jet's fine structure? (set up 2026-09-28, before any closure run)

**Why.** The closure test showed that paired training reproduces each jet's
modification, while unpaired minibatch OT-CFM flattens the hard cores. A 4x
matching pool helped modestly; even a perfect teacher could not find a close
enough real target among 256; teacher-guided matching damaged a good paired
model. These observations motivate removing the real-target assignment
step. They do not prove that the candidate distance bounds the learned
output error, nor that every decomposition failure comes from the coupling
(Part 9 also found core leakage in *paired* background-only flows).

**Hypothesis.** An online bridge algorithm whose training endpoints are
generated by the current model, not selected from other observed jets, can
improve event correspondence. This tests the chosen Schrodinger-bridge
coupling (entropic OT of |x - y|^2 / 2 in the model coordinates, with
regularisation eps). It is not a claim that marginal matching identifies
the physical modification or the background posterior.

**Method: online alpha-DSBM** (De Bortoli et al., NeurIPS 2024,
arXiv:2409.09347, Algorithm 1, Section 4, Appendices J and K; the 2023 DSBM
code, github.com/yuyang-shi/dsbm-pytorch, as a reference for conventions
only). `scripts/flow/dsbm.py`:
- **Reference process** sqrt(eps) B; bridge Interp_t(x0, x1, z) = (1 - t) x0
  + t x1 + sqrt(eps t (1 - t)) z; the same eps in the SDE sampler.
- **One network, two drifts:** the UVCGAN-S velocity backbone of the flow
  study (`fm_common.UVCGANVelocity`, time added to its extra/style token)
  plus a direction input s (1 forward, 0 backward): sinusoidal features and a
  2-layer MLP like the time, the two outputs concatenated and projected to
  the token (Appendix K). Markov inputs only: state, time, direction; no
  conditioning on the original event.
- **Clocks and targets:** each direction runs on its own clock u from its
  start to its end. Forward u = t, target the endpoint X1, drift (E[X1 | X_t] -
  X_t) / (1 - t); backward u = 1 - t (Algorithm 1's reverse time), target X0,
  drift (E[X0 | X_t] - X_t) / t. These are the targets (X1 - X_t)/(1 - t) and
  (X0 - X_t)/t of Algorithm 1's losses (13).
- **Parameterisation and weighting:** endpoint = c_skip x + c_out nn(c_in x)
  with Appendix J's coefficients, generalised (as Appendix J notes) to the
  endpoint second moments v_start, v_end of each direction: D = (1-u)^2
  v_start + u^2 v_end + eps u (1-u), c_in = D^-1/2, c_skip = u v_end / D,
  c_out = (v_end (1 - u^2 v_end / D))^1/2. The network regresses the
  unit-variance target (X_end - c_skip x) / c_out with unit weight (Karras
  et al.'s choice, derived in Appendix J): the same per-time minimiser as
  (13), without its singular weight near the end; t ~ U[1e-4, 1 - 1e-4] as
  in the reference. For the closure, v_0 = 1.048 and v_1 = 0.955 (the pools'
  E[z^2]), nearly the unit case. Two earlier versions failed the Gaussian
  check (below): without the skip, and with unit variances for unequal
  endpoint scales.
- **Stages:** (1) pretraining, bridge matching on independent real pairs,
  half of each batch per direction; (2) online refinement, per update b =
  128 real sources X0 and 128 real targets X1: X1_hat from the forward SDE
  started at X0 and X0_hat from the backward SDE started at X1, EMA
  parameters, no gradient; the forward drift is trained on (X0_hat, X1), the
  backward drift on (X0, X1_hat), loss (l_fwd + l_bwd)/2 (Algorithm 1, lines
  9-14). Generated endpoints stay in model coordinates.
- **Optimisation:** Adam 2e-4 (betas 0.9, 0.999), 1000 warm-up updates,
  gradient norm clipped at 1, batch 256; EMA decay 0.999 (the paper's image
  models), rollouts and evaluation with the EMA parameters (the paper's
  default); a fresh optimiser at every stage start (Appendix K); the
  pretrained checkpoint is kept and both later stages start from its raw
  and EMA weights.
- **Sampler:** Euler-Maruyama, N = 30 equal steps (the paper's MNIST
  setting) for the rollouts and for evaluation, increments sqrt(eps h) z;
  the last step returns the endpoint prediction without the final noise
  (the reference's endpoint handling for transfer tasks).
- **Not used anywhere in training or sampling:** true pairs, OT assignment,
  adversarial or shape losses, teachers.

**Noise level, fixed before any closure score.** In model coordinates the
closure canvas has unit variance (the cone's 53 towers vary with a per-pixel
spread of 1.4-1.7, the rest is the constant -0.315 of an empty tower; random
source-target pairs differ by 2.2 per cone tower). The paper sets the
bridge-midpoint noise at ~0.75-0.8x the per-pixel data spread (MNIST eps =
1; AFHQ-64 eps = 0.75^2); scaled to our canvas that gives eps ~2.6, or ~0.85
with its resolution shift (16 vs 28 px). **eps = 1**, the paper's MNIST
value: midpoint noise sd sqrt(eps)/2 = 0.5 (0.32 in log E). At most one more
setting, eps = 0.25, only if the first result is limited by the entropic
spread.

**Gaussian check first** (`dsbm_gauss.py`): pi0 = N(0, I_16), pi1 = N(0, 4
I_16), eps = 1, the same code and sampler. Analytic SB cross-covariance c* =
(sqrt(4 s0^2 s1^2 + eps^2) - eps)/2 = 1.5616 per dimension (OT 2,
independent 0). Pass: both directions' coupling within 5% of c* and both
generated variances within 5%. Network-free reference: the same procedure
with exact linear projections and the same 30-step sampler converges to
1.573 / 1.565 (variances 3.984 / 0.984); its pretrained stage gives 1.372
(12% short of c*), so a pretrained-only sampler must fail the check.
Status when the closure runs were launched (details in the results below):
pretraining matches the network-free values to <= 1%; refinement with a
model linear in x reaches the 5% band; the paper's concatenation MLP settles
~10% high in the coupling and ~20% high in the variances (the online loop
amplifies its approximation error).

**Closure runs** (seed 0, eps = 1, UVCGAN backbone, the closure pools,
normalisation and splits of `closure_data.py`, unchanged):

| arm | what | budget, GPU-h (one A6000) |
| :--- | :--- | ---: |
| pretrained bridge | stage 1 only | 0.5 |
| **alpha-DSBM** | stage 1 + 1.5 h online refinement, rollouts included | 2.0 |
| continued bridge | stage 1 + 1.5 h more independent bridge matching | 2.0 |
| OT-CFM, UVCGAN backbone | the matched comparator (existing runs used the U-Net): exact minibatch OT, squared L2 in the same coordinates, pool 256, 2 h; EMA 0.9999 as the historical OT-CFM runs | 2.0 |

Total allocation: 0.5 + 1.5 + 1.5 + 2.0 = **5.5 GPU-hours** of training,
plus ~0.5 GPU-h of scoring. Only if needed: eps = 0.25 (+3.5), a second
seed of a clearly promising result (+3.5), the M -> B pilot (+2 and the
benchmark). The Gaussian check and its diagnostics used ~2 GPU-hours.

References reused, labelled as historical (U-Net backbone, ODE): the
identity, the paired control, the unpaired OT-CFM runs.

**Selection and scoring.** Checkpoints every 10 min, selected on the first
5000 validation pairs by the closure test's metric (mean full EMD, GeV), one
SDE sample per jet from a fixed seed (the OT-CFM comparator: 32 midpoint
evaluations, as the control). Test: 20k pairs, one sample per jet, fixed
seed; the closure report (`closure_eval.control_scores`): normalised-shape
EMD (dimensionless, unit-energy jets: mean, median, p90, p99), EMD in GeV,
energy response and RMSE, hardest-tower error, mass and girth change, p_T^D
and z_lead, marginals W1/sigma, error by true tower energy. Also: N = 30
against 60 steps with coupled Brownian increments against two independent
samples (1000 validation pairs); 8 samples per jet on 1000 test pairs
(per-jet spread, shape EMD between samples, the average image reported
separately); event displays of preselected jets; updates, memory, NFE,
latency. This toy has one right answer per jet, so conditional spread is
error here.

**Decision gate (fixed now).** Promising if alpha-DSBM's single-sample test
shape EMD is clearly below OT-CFM's and below the identity's, with an energy
response near 0.8, marginals no worse than OT-CFM's and no hard-core loss.
Then a second seed; if robust, one unpaired M -> B pilot through the Part 9
benchmark. Otherwise stop after the closure test and report the limitation.

**When eps = 0.25 is run (fixed 11:58, before any score of a refined
model).** Only if alpha-DSBM fails the gate *and* its error is dominated by
the sampling spread: on the 1000 validation pairs of the step check, the
shape EMD between two independent samples of the same jet is at least its
single-sample shape EMD to T(J). (For a Euclidean distance, the spread then
makes at least half of the squared error.) Otherwise the error is mostly
systematic, not the noise level's, and the test stops at eps = 1.

**Triggered (13:57).** eps = 1 failed the gate on test (single-sample shape
EMD 0.097 against the identity's 0.032), and on the step check's 1000
validation pairs two samples of a jet differ by 0.116 against 0.099 to
T(J) (ratio 1.17). The eps = 0.25 arms (`dsbm_*_e025_s0`, jobs 20314-20316)
were launched with everything else unchanged (midpoint noise sd 0.25 in z,
0.16 in log E).

### Gaussian check results (jobs 20238-20270, 2026-09-28)

`dsbm_gauss.py`, pi0 = N(0, I_16), pi1 = N(0, s1^2 I_16), eps = 1, 30
Euler-Maruyama steps, 10k pretraining updates (Adam 1e-4, batch 256), then
online refinement; 10k samples per measurement. Values relative to the
analytic SB answer (coupling c* per dimension; variance s^2), after
refinement; `docs/flow/dsbm/gauss_check.{png,csv}`, per-run histories
`docs/flow/dsbm/diag_*.{csv,json}`:

| variant (s1 = 2 unless noted) | network | coupling fwd / bwd | Var X1_hat | Var X0_hat | passes (5%) |
| :--- | :--- | ---: | ---: | ---: | :---: |
| pretrained only (every variant), network-free value 0.878 / 0.94 / 0.93 | | 0.86-0.89 | 0.93-0.95 | 0.92-0.95 | no |
| no preconditioning (endpoint regression, job 20238; stopped) | MLP | 1.30 / 0.79 | 1.60 | 0.66 | no |
| unit-variance preconditioning (jobs 20245-20258; lr 2e-5 shown) | MLP | 0.75 / 1.51 | 0.63 | 2.03 | no |
| unit-variance preconditioning | **linear in x** | 1.002 / 1.008 | 0.98 | 0.99 | **yes** |
| per-direction variances (as the closure) | **linear in x** | 1.004 / 1.031 | 0.99 | 1.03 | **yes** |
| same, lr 3e-5, batch 512, 20k updates | **linear in x** | 0.991 / 1.019 | 0.97 | 1.01 | **yes** |
| per-direction variances | MLP (paper K.2) | 1.065 / 1.043 | 1.13 | 1.08 | no |
| same, lr 3e-5, batch 512, 20k updates | MLP | 1.030 / 1.032 | 1.05 | 1.05 | no (still falling) |
| s1 = 1 (symmetric) | MLP | 1.047 / 1.044 | 1.07 | 1.07 | no |
| s1 = 1, lr 3e-5, batch 512, 20k updates | MLP | 1.045 / 1.042 | 1.07 | 1.06 | no (still falling) |

- **The conventions are right.** Pretraining reproduces the network-free
  iteration (`imf_exact.py`: exact linear projections, the same 30-step
  sampler) to <= 1% in the preconditioned MLP runs. With a network that can
  represent the exact drift (linear in x, any function of time and
  direction), online refinement converges to the analytic coupling and
  marginals within 1-3%, and stays there at 20k updates. The discretised
  exact fixed point is 1.007 / 1.002 (variances 0.996 / 0.984), so the
  30-step sampler itself costs < 1%.
- **The check discriminates.** The pretrained-only sampler matches both
  marginals within 6-8% but its coupling is 12% short; a sampler with the
  wrong-sign coupling (c = -2.00) fails; 30 against 60 steps with the same
  Brownian path changes c by <= 1% (rms difference of the outputs 0.13).
- **Two parameterisations failed and were replaced before the closure
  runs:** plain endpoint regression (no skip; the pretrained forward
  variance is already 17% high and refinement runs away), and Appendix J's
  unit-variance preconditioning with unequal endpoint scales (a wrong fixed
  point in every variant tried: warm-up, online-network rollouts, 100 steps,
  lr 2e-5).
- **The online loop amplifies function-approximation error.** The paper's
  concatenation MLP overshoots: the coupling 3-7% and the variances 5-13%
  high, shrinking only slowly with a 3x smaller learning rate and 2.5x more
  updates (M3, S3 above). This is a property of the online procedure with a
  network that can't represent the drift exactly (Appendix I discusses
  error accumulation), not of the conventions. The UVCGAN backbone is not
  linear either, so the closure's marginals are checked against the target.
- Cost of the check and its diagnostics: 15 jobs, 2.1 GPU-hours on one
  A6000 each (sacct).

### Closure results at eps = 1 (jobs 20271-20274 training, 20311-20313 scoring)

**Training** (one A6000 each, run side by side on dahlia; `summary.json`):

| arm | GPU h | updates | updates / s | time in rollouts (OT: matching) | peak GB | inference, ms / jet |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| bridge, pretrained | 0.50 | 26.9k | 14.9 | | 3.6 | SDE 30 steps, 1.51 |
| alpha-DSBM | 0.50 + 1.50 | 26.9k + 9.0k | 1.67 | 89% (7680 NFE / update) | 3.8 | SDE 30 steps, 1.51 |
| bridge, continued | 0.50 + 1.50 | 26.9k + 80.6k | 14.9 | | 3.8 | SDE 30 steps, 1.51 |
| OT-CFM, UVCGAN, L2 | 2.00 | 122.8k | 17.1 | 10% | 3.5 | ODE, 32 midpoint NFE, 1.61 |

Latency: batch 2000, one A6000, in the same scoring job. The bridge net has
32.8M parameters (the OT-CFM net 32.3M plus the direction embedding).

**Validation** (`docs/flow/dsbm/closure_curves.{png,csv}`; 5000 pairs, one
output per jet):
- **alpha-DSBM:** shape EMD 0.195 (the pretrained bridge) -> 0.132 after 1k
  refinement updates, then 0.110, 0.104, 0.101, 0.099, 0.098, 0.098, 0.097,
  0.097. At the end it still falls by ~0.0005 per 1000 updates. Response
  0.84-0.85 early in the refinement, back to 0.82.
- **The continued bridge** stays at 0.195-0.200 and **OT-CFM** (UVCGAN) at
  0.087-0.093, from their first checkpoints on.
- Train (2000 or 5000 training pairs) and val agree within 0.001 for every
  run. Selected: the last checkpoint for alpha-DSBM and the pretrained
  bridge; 100 min for OT-CFM; 70 of 90 min of continuation for the
  continued bridge.

**Test** (`docs/flow/dsbm/closure_test.csv`, `_observables.csv`; 20k
held-out pairs, one output per jet, fixed seed; `+` = historical U-Net
reference, ODE):

| | shape EMD: mean / median / p90 / p99 | EMD, GeV | E_out / E_in (0.8) | E RMSE, GeV | towers > 5 GeV: mean / rms, GeV | mass change RMSE, GeV | girth change RMSE |
| :--- | :--- | ---: | :--- | ---: | :--- | ---: | ---: |
| identity | 0.032 / 0.032 / 0.039 / 0.047 | 7.03 | 1 | 6.61 | +4.05 / 4.45 | 1.01 | 0.0044 |
| paired control + | 0.0002 / 0.0001 / 0.0002 / 0.0005 | 0.015 | 0.7997 +- 0.0004 | 0.02 | -0.006 / 0.014 | 0.003 | 0.0000 |
| OT-CFM U-Net, L2 + | 0.090 / 0.077 / 0.151 / 0.274 | 3.56 | 0.785 +- 0.079 | 2.83 | -1.82 / 3.44 | 0.55 | 0.0144 |
| OT-CFM U-Net, shape + E + | 0.091 / 0.083 / 0.143 / 0.240 | 2.87 | 0.808 +- 0.038 | 1.32 | -0.87 / 2.16 | 0.54 | 0.0146 |
| same, pool 1024 + | 0.076 / 0.070 / 0.117 / 0.188 | 2.37 | 0.802 +- 0.032 | 1.10 | -0.50 / 1.61 | 0.47 | 0.0113 |
| **OT-CFM, UVCGAN, L2** (matched) | 0.092 / 0.079 / 0.152 / 0.284 | 4.02 | 0.778 +- 0.102 | 3.65 | -1.49 / 3.60 | 0.69 | 0.0145 |
| bridge, pretrained | 0.194 / 0.151 / 0.360 / 0.731 | 7.62 | 0.799 +- 0.169 | 5.49 | -2.97 / 5.36 | 0.88 | 0.0289 |
| bridge, continued | 0.195 / 0.152 / 0.363 / 0.729 | 7.56 | 0.788 +- 0.164 | 5.38 | -3.11 / 5.35 | 0.88 | 0.0292 |
| **alpha-DSBM** | 0.097 / 0.090 / 0.142 / 0.224 | 5.75 | 0.818 +- 0.170 | 5.49 | -1.45 / 4.18 | 0.79 | 0.0152 |

Distributions against T(J) (W1 / sigma: E, mass, girth, p_T^D, z_lead):
identity 1.24, 0.92, 0.08, 0.67, 0.49; OT-CFM UVCGAN 0.18, 0.12, 0.03,
0.15, 0.12; OT-CFM U-Net 0.02-0.13, 0.07-0.14, 0.06-0.10, 0.14-0.36,
0.10-0.28; pretrained bridge 0.12, 0.09, 0.04, 0.08, 0.06; **alpha-DSBM
0.02, 0.06, 0.03, 0.04, 0.03**. Per-jet RMSE / sigma (p_T^D, z_lead):
alpha-DSBM 0.73, 0.84; OT-CFM UVCGAN 0.68, 0.72; identity 0.73, 0.55.
Energy in the truth-empty towers: alpha-DSBM 0.014 GeV per jet, the bridges
0.03, OT-CFM <= 0.001. Clipping: raw outputs are >= -0.045 GeV (the
readout's floor is -0.1); every score uses the output clipped at 0, as
before; in the truth-empty towers the clip changes the energy by < 0.001
GeV per jet.

**Sampling spread and integration** (the step check on 1000 val pairs,
`closure_test_steps.csv`; 8 samples of each of 1000 test jets,
`closure_test_multi.csv`):

| | shape EMD to T(J) | two samples of one jet | 30 vs 60 steps, same noise | mean of 8 samples to T(J) | spread of E / sigma(E), girth, mass |
| :--- | ---: | ---: | ---: | ---: | :--- |
| bridge, pretrained | 0.192 | 0.222 | 0.016 | 0.130 | 0.71, 0.44, 0.58 |
| bridge, continued | 0.193 | 0.224 | 0.016 | 0.132 | 0.67, 0.45, 0.56 |
| alpha-DSBM | 0.099 | 0.116 | 0.013 | 0.058 | 0.70, 0.25, 0.51 |

- **30 steps resolve the SDE.** Doubling the steps with the same Brownian
  path moves a jet by about a tenth of the distance between two samples;
  the error to T(J) is 0.099 against 0.100 at 60 steps.
- **The spread is most of the error.** Two samples of a jet differ by more
  than either differs from T(J). The energy of a jet varies between samples
  by 70% of the whole population's spread: its E_out / E_in has sd 0.17,
  against 0.03-0.10 for OT-CFM.
- **Beyond the spread there is a systematic part.** Even the mean of 8
  samples (reported separately, not a result) is 0.058, 1.8x the identity.
- Single-sample scores vary by 0.001 between seeds of the sampler.

**Decision (pre-registered gate): not promising at eps = 1.**
- The single-sample shape EMD (0.097) is not below OT-CFM's with the same
  backbone and budget (0.092), and 3x the identity's (0.032).
- The energy response (0.82) is near 0.8 and the marginals are better than
  OT-CFM's, but the hard core is not kept better (-1.45 / 4.18 GeV against
  -1.49 / 3.60) and the per-jet energy error is worse (5.5 against 3.6 GeV).

**What the refinement did (measured).**
- At the same 2 GPU hours, learning the coupling online halves the error of
  continuing the independent-pair bridge: shape 0.097 against 0.195,
  girth-change RMSE 0.015 against 0.029, EMD 5.75 against 7.56 GeV.
- It also gives the closest output distributions of every unpaired run:
  W1 <= 0.06 sigma in the five observables, OT-CFM 0.03-0.18.
- So the procedure works as an algorithm: it moves the pretrained coupling
  towards a tighter one and keeps the marginals. That is not a correct
  physical correspondence: the Schrodinger bridge at eps = 1 in these
  coordinates is still a random map, its conditional spread comparable to
  the jets' own variation.

### Closure results at eps = 0.25, and the verdict (jobs 20314-20316 training, 20322 scoring)

Triggered by the pre-registered rule (above); everything else unchanged.
Training: pretraining 27.0k updates in 30 min (selected at 10 min on val,
0.207; the later stages start from its 30-min checkpoint, as at eps = 1);
alpha-DSBM 8.7k refinement updates (1.61/s, 89% in rollouts); the continued
bridge 81.2k. Validation: alpha-DSBM 0.217 -> 0.154 after 1k refinement
updates, 0.124, 0.109, 0.098, 0.091, 0.087, 0.083, 0.080, 0.077 at 90 min,
**still falling by 0.0025 per 10 min**; the continued bridge flat at 0.218.

Test (20k pairs, one sample per jet; the full table, both noise levels and
every reference: `docs/flow/dsbm/closure_test.csv`, slides A2):

| | shape EMD: mean / median / p90 / p99 | EMD, GeV | E_out / E_in | E RMSE, GeV | towers > 5 GeV, GeV | mass / girth change RMSE |
| :--- | :--- | ---: | :--- | ---: | :--- | :--- |
| identity | 0.032 / 0.032 / 0.039 / 0.047 | 7.03 | 1 | 6.61 | +4.05 / 4.45 | 1.01 / 0.0044 |
| OT-CFM, UVCGAN, L2 (matched) | 0.092 / 0.079 / 0.152 / 0.284 | 4.02 | 0.778 +- 0.102 | 3.65 | -1.49 / 3.60 | 0.69 / 0.0145 |
| OT-CFM U-Net, shape + E, pool 1024 + | 0.076 / 0.070 / 0.117 / 0.188 | 2.37 | 0.802 +- 0.032 | 1.10 | -0.50 / 1.61 | 0.47 / 0.0113 |
| alpha-DSBM, eps = 1 | 0.097 / 0.090 / 0.142 / 0.224 | 5.75 | 0.818 +- 0.170 | 5.49 | -1.45 / 4.18 | 0.79 / 0.0152 |
| bridge, continued, eps = 0.25 | 0.217 / 0.165 / 0.445 / 0.757 | 7.53 | 0.781 +- 0.145 | 4.82 | -3.07 / 5.19 | 0.76 / 0.0320 |
| **alpha-DSBM, eps = 0.25** | 0.077 / 0.068 / 0.124 / 0.237 | 4.42 | 0.796 +- 0.129 | 4.34 | -0.44 / 3.38 | 0.67 / 0.0122 |

Distributions (W1 / sigma: E, mass, girth, p_T^D, z_lead): alpha-DSBM eps =
0.25: 0.08, **0.35**, 0.08, 0.24, 0.20; eps = 1: 0.02, 0.06, 0.03, 0.04,
0.03; OT-CFM UVCGAN: 0.18, 0.12, 0.03, 0.15, 0.12. On average the eps = 0.25
model changes the jet mass by 1.38x the true change and the girth by 0.10x
(truth 1, 1; eps = 1: 1.06, 0.60).

Spread (step check, 1000 val pairs; 8 samples of 1000 test jets): two
samples of a jet differ by 0.062 against 0.079 to T(J) (ratio 0.79; eps =
1: 1.17); the mean of 8 samples scores 0.064 against 0.077 for one sample
(eps = 1: 0.058 against 0.096); 30 against 60 steps with the same noise:
0.006. The per-jet energy spread falls from 0.70 to 0.45 of the population
spread.

**Verdict (pre-registered gate): not promising at either noise level; the
test stops here.**
- eps = 0.25 has the lowest single-sample shape EMD of the UVCGAN-backbone
  runs (0.077; the matched OT-CFM 0.092) and keeps the hard core best
  (-0.44 GeV mean error; OT-CFM -1.49). That is at the historical 4x-pool
  OT-CFM's level (0.076), not "clearly below" OT-CFM, and it is 2.4x the
  identity. Its distributions are worse than OT-CFM's (mass, p_T^D, z_lead),
  so the gate fails on two of its conditions.
- eps = 1 fails as reported above.
- No second seed and no M -> B pilot (both were conditional on the gate).
  The eps = 0.25 curve had not flattened at 2 GPU h, so its limit at a
  larger budget is **inconclusive**. Even at its final, slowing rate
  (-0.0025 per 10 min; -0.031 over the last hour) it would need at least 3
  more GPU hours to reach the identity's 0.032.

**Measured, and what it means.**
- Algorithmic: at both noise levels the online refinement halves the error
  of continued independent-pair training at the same cost (0.097 against
  0.195; 0.077 against 0.217). A smaller eps makes the single samples less
  random (spread 0.12 -> 0.06).
- Its error then becomes systematic: the mass change overshoots and the
  broadening is missing. The Gaussian check showed that the online loop
  amplifies the network's approximation error (MLP: +3-7% coupling, +5-13%
  variance); the same mechanism here is a hypothesis, not tested.
- Physical correspondence is a separate question: the Schrodinger bridge of
  |x - y|^2 / 2 in log-energy coordinates is a mathematical choice of
  coupling. T(J) is not that coupling's answer (the identity's own error,
  0.032, is well below what any unpaired coupling reached), so even a
  converged bridge would not be evidence of the physical modification.

## Paired noisy-interpolant pilot: does a noisy training path reduce the paired flow's core leakage? (set up 2026-09-28 13:40, before training)

**Why.** In the consolidated benchmark the *paired* background-only flow
(`otcfm1_paired`, M -> B with each mixture's own background) still puts
part of the jet core into the background: jet scale 0.828 with 4 Euler steps
and 0.751 with the accurate solve; 16% / 23% of the true signal energy of
towers with S > 10 GeV goes into B_hat (24% / 31% at S 5-10 GeV). Correct
training pairs alone did not fix it. Sign: with S_hat = M - B_hat, lost
signal means B_hat is too *high*; the flow stops short of the true
background in the core, it does not subtract too much.

**Hypothesis (not an established explanation).** Training only on
noiseless straight interpolants supervises the velocity field only on the
segments between (z(M), z(B)) pairs; the states the ODE visits at
inference drift off them, where the field is extrapolated. A noisy
interpolant that returns exactly to the endpoints trains the field in a
neighbourhood of the paths. Caveats: even an exactly learned marginal
transport need not be the best event-specific estimator; noisy training
paths do not make this a posterior sampler or a solved Schrodinger bridge.

**Path** (stochastic interpolants, Albergo, Boffi and Vanden-Eijnden, JMLR
26, 2025; flow matching, Lipman et al., arXiv:2210.02747), with a =
z_bkg(M), b = z_bkg(B_true), t ~ U[0, 1), eps ~ N(0, I) of a's shape:
gamma(t) = eta sin(pi t); x_t = (1 - t) a + t b + gamma(t) eps; u_t = (b -
a) + eta pi cos(pi t) eps (the same eps); loss mean((v(t, x_t) - u_t)^2).
eta is the largest noise sd, at t = 1/2, in the standardised log-energy
coordinates (not GeV); gamma(0) = gamma(1) = 0. `fm_train.py --path sine
--eta ETA` adds gamma eps and gamma' eps to the straight path of the
existing sigma = 0 matcher (`fm_common.Method.sine_path`); eps comes from
its own generator, so initial weights, mixture order and sampled times
are those of eta = 0. No clipping of the noisy states.

**Checks before training** (`sine_path_check.py`, CPU, 64 real training
pairs; `docs/flow/bench/noisy/path_check.json`): eta = 0 reproduces the
straight path's times, states and targets bit for bit and leaves the global
RNG state unchanged; eta = 0.1 keeps the times and the RNG stream; x_0 = a
and x_1 = b exactly; u_t equals the central difference of x_t (float64, at
fixed a, b, eps) to 3e-10 (|u| up to 6.7); the added noise has sd
gamma(t) (ratio 0.999).

**Arms** (seed 0, one A6000 each, run side by side on dahlia; only the path
differs): control eta = 0 (`--path sine --eta 0`, identical to the
existing straight path), treatment eta = 0.1. Everything else as the
benchmark's paired arm: `otcfm1_paired`, the UVCGAN-S velocity backbone,
log(E + 0.1) standardised, the same 633k training mixtures and val events,
Adam 2e-4, warm-up 1000, clip 1, EMA 0.9999, batch 256. **Budget: exactly
32,640 updates** (8 x 4080; the benchmark's seed 0 made 32,620 in its 2 h),
checkpoints every 4080 updates (the benchmark run's own checkpoint steps,
so the fresh control can be checked against it), inline val curves (1000
events, 4 Euler) and the per-checkpoint val cone scores as in the
benchmark. Wall time recorded. No new conditioning, matching,
discriminator, endpoint, shape or joint loss.

**Evaluation** (PYTHIA val only; JEWEL frozen until the comparison and
choices are fixed, and then only if the pilot merits a follow-up):
- **Primary: the final update (32,640) of both arms**, EMA, deterministic
  ODE from z_bkg(M), S_hat = M - B_hat, the accurate solve (midpoint, 32
  network evaluations = 16 steps). Checked against 64 evaluations on the
  same 1000 val events for both arms; a finer solve only if the 32/64
  difference is material next to the treatment effect. 4 Euler steps
  reported separately as the fast readout; a gain only at 4 steps is not
  evidence of a better continuous flow. Any secondary checkpoint choice
  uses one declared rule for both arms (none planned).
- **The benchmark's analysis, unchanged:** background error B_hat - B by
  true signal energy of the tower with the energy-weighted leakage share
  (S 5-10 and > 10 GeV bins the key ones); raw jet scale, calibrated
  resolution (frozen val calibration), efficiency and fake rate; per-jet
  bias / RMSE and distribution W1 of the substructure observables, kept
  apart; per-tower error, the off-signal (S = 0) residual and the event
  energy bias; the fixed event displays. References: the published
  UVCGAN-S (its ~105 h budget and the val-signal overlap caveat) and the
  benchmark's paired arm (selected checkpoint; context only, not the
  comparison).
- **Trajectory diagnostic** (`noisy_traj.py`, the first 256 val events,
  out of training and selection): (1) the ordinary ODE path in hard towers
  (S > 5 GeV): does the decoded background B(x_t) ever fall below the true
  B, or stay above it? (2) at t0 = 0.25, 0.5, 0.75, start from the clean
  reference interpolant (1 - t0) a + t0 b, unperturbed and with +/- a fixed
  small perturbation (identical for both models), integrate the rest
  accurately, and measure the endpoint response and its error against the
  event's true B. Contraction alone is not success (a field can be
  insensitive by smoothing away structure).

**Decision (fixed now).** Seed half ranges of the benchmark's paired arm
(32 NFE, val): leakage share 0.001 (S 5-10) and 0.0006 (> 10); scale
0.002; efficiency 0.001; per-jet substructure RMSE <= 0.008 in the
observable's units. A difference counts if it exceeds 3x the larger of
that half range and the statistical error. **Promising** only if, at the
accurate solve: the leakage share falls in both hard bins (S 5-10, > 10
GeV); at least two substructure observables improve (per-jet RMSE or W1)
and none worsens; and nothing offsets it (scale not lower, efficiency not
lower, fake rate not higher, no added off-signal or event energy). Then
seeds 1 and 2 of both arms before any claim. **Clearly fails** (no leakage
reduction at the accurate solve, or a gain only at 4 steps): stop; no
noise or path sweep. A negative result rules out only this setting.

### Pilot results (jobs 20309-20310 training, 20317-20321 scoring and analysis)

**Training.** Both arms stopped at exactly 32,640 updates: control 120.0
min of training, 120.7 end to end (4.53 updates/s); eta = 0.1 118.8 and
120.1 min (4.58/s); side by side on dahlia; peak memory as the benchmark's
paired arm (18.1 GB). Manifest and configs: `docs/flow/bench/noisy/`
`noisy_runs.csv`, `configs/`. The fresh
control repeats the benchmark's `bench_paired_bkg_s0` loss history to four
significant digits (0.13810 against 0.13810 at step 100, 0.07586 against
0.07584 at 300): the same initialisation, mixture order and times. The
treatment's loss is 0.049 higher throughout, the irreducible part of the
noisy target (eta^2 pi^2 / 2 per element).

**Validation curves** (`docs/flow/bench/noisy/noisy_curves.{png,csv}`; 20k
val events, cone scores, EMA): see the figure; the arms stay within 0.1 GeV
of each other in cone resolution at 4 Euler steps at every checkpoint.

**Solver check** (`traj_solver.csv`, the first 1000 val events, final
update): midpoint 32 against 64 evaluations differs by 0.0004 GeV rms per
tower and by <= 0.0007 in every leakage share, for both arms. 32 is
resolved; no finer solve was needed. 4 Euler steps differ from it by 0.036
GeV per tower and -5.3 GeV per event (less background: the 4-step solve's
lower leakage is a discretisation effect, as in the benchmark).

| 1000 val events, accurate solve (mid64) | S 0.5-2 | S 2-5 | S 5-10 | S > 10 GeV | B_hat - B where S = 0, GeV |
| :--- | ---: | ---: | ---: | ---: | ---: |
| control eta 0: share of S in B_hat | 65.4% | 47.2% | 31.2% | 23.2% | -0.025 |
| eta 0.1 | 65.4% | 47.3% | 31.2% | 23.2% | -0.025 |
| control, 4 Euler steps | 61.0% | 39.7% | 23.6% | 16.0% | -0.026 |
| eta 0.1, 4 Euler steps | 61.1% | 39.9% | 23.6% | 16.0% | -0.025 |

**Trajectory diagnostic** (`traj.{png,json}`, `traj_paths.csv`,
`traj_perturb.csv`; the first 256 val events, 592 hard towers with S > 5
GeV, final update, EMA, midpoint):
- **The ODE never undershoots.** In no hard tower does the decoded
  background B(x_t) fall below the true B at any step, and every hard tower
  ends above it (both arms). The flow stops short in the core; it does not
  overshoot and come back.
- **It stalls, both arms alike.** The hard-tower error sum(B(x_t) - B) /
  sum(S) follows the clean interpolant (1 - t) a + t b only up to t ~ 0.2
  (0.60 against 0.57 at t = 0.19), is 0.31 against 0.21 at t = 0.5, and is
  frozen after t ~ 0.7 (0.28 -> 0.275 at t = 1) while the clean path goes
  to 0. eta = 0.1 differs from the control by <= 0.003 at every t.
- **The field, not the drift off the path, keeps the core in B_hat.**
  Started exactly on the clean interpolant, the remaining solve still ends
  with 26% (t0 = 0.25), 18% (0.5) and 7% (0.75) of the hard towers' signal
  in B_hat, against 27% from t = 0; eta = 0.1: 25%, 18%, 7%.
- **Displacements survive.** A +-0.05 shift of the hard towers at t0 is
  kept at the endpoint with factor 0.70 / 0.92 / 1.00 (t0 = 0.25 / 0.5 /
  0.75; eta = 0.1: 0.69 / 0.92 / 1.00), a random direction 0.74 / 0.95 /
  1.00. Neither field restores a displaced core. The endpoint error against
  the true B changes by the same amount in both arms.

**The benchmark at the final update** (`docs/flow/bench/noisy/`:
`bench_summary.csv`, `bench_jets.csv`, `bench_towers_summary.csv`,
`bench_fig3-6_val.png`, `bench_bkgerr.png`, `bench_displays*.png` (the
benchmark's fixed events); `noisy_compare.csv`: every decision metric with
its threshold; PYTHIA val, 20k events, R = 0.4; UVCGAN-S: published, ~105
h, its training signals include the val signal images):

| | UVCGAN-S | control, 32 NFE | eta 0.1, 32 NFE | control, 4 Euler | eta 0.1, 4 Euler |
| :--- | ---: | ---: | ---: | ---: | ---: |
| jet scale, 20-30 GeV | 0.982 | 0.753 | 0.753 | 0.832 | 0.831 |
| resolution, frozen calibration | 0.153 | 0.161 | 0.161 | 0.156 | 0.157 |
| jet pT RMSE, 20-30 GeV | 3.59 | 7.03 | 7.03 | 5.45 | 5.46 |
| efficiency, 14-20 GeV | 0.972 | 0.956 | 0.959 | 0.969 | 0.971 |
| fake rate, 14-20 GeV | 0.030 | 0.002 | 0.002 | 0.006 | 0.005 |
| share of S in B_hat, towers S 2-5 GeV | 12.1% | 47.0% | 47.0% | 39.4% | 39.6% |
| S 5-10 GeV | 2.2% | 31.4% | 31.4% | 23.6% | 23.7% |
| S > 10 GeV | 0.8% | 23.1% | 23.2% | 15.8% | 15.9% |
| B_hat - B where S = 0, GeV per tower | -0.010 | -0.025 | -0.025 | -0.025 | -0.025 |
| sum(S_hat - S), GeV per event | -7.1 | +0.7 | +0.7 | +6.1 | +6.0 |
| per-tower MAE, GeV | 0.033 | 0.050 | 0.050 | 0.049 | 0.049 |

Substructure (20-30 GeV, in units of the spread of all truth jets; g, m,
z_lead, p_T^D, z_g, r_g):

| | per-jet RMSE / sigma | distribution W1 / sigma |
| :--- | :--- | :--- |
| UVCGAN-S | 0.60, 1.16, 0.38, 0.47, 1.12, 0.94 | 0.07, 0.21, 0.04, 0.06, 0.01, 0.02 |
| control, 32 NFE | 0.54, 1.36, 0.34, 0.53, 1.12, 0.85 | 0.22, 0.89, 0.16, 0.40, 0.14, 0.14 |
| eta 0.1, 32 NFE | 0.54, 1.36, 0.34, 0.53, 1.12, 0.85 | 0.22, 0.90, 0.16, 0.40, 0.15, 0.14 |
| control, 4 Euler | 0.55, 1.16, 0.34, 0.51, 1.12, 0.85 | 0.22, 0.51, 0.15, 0.36, 0.13, 0.13 |
| eta 0.1, 4 Euler | 0.55, 1.16, 0.34, 0.52, 1.12, 0.86 | 0.22, 0.52, 0.15, 0.37, 0.13, 0.13 |

- **Every decision metric is unchanged** (`noisy_compare.csv`: all 58
  comparisons within their thresholds, at both solves). The largest
  leakage change is +0.0007 (S > 10 GeV, 32 NFE; threshold 0.0019). The
  fresh control after 2 h matches the benchmark's paired seed 0 at its
  selected 30-min checkpoint: scale 0.753 against 0.752, leakage 23.1% /
  31.4% against 23.1% / 31.4% (S > 10 / 5-10 GeV). The core loss does not
  change with more training either.
- The 4-step readout keeps its trade-off in both arms: less core in B_hat
  (15.8% against 23.1%) and a higher scale (0.83 against 0.75), bought with
  +6 GeV per event of spurious signal and larger distribution biases
  elsewhere (mass W1 0.51 against 0.89 is the one improvement). The noisy
  path changes neither side of it.

**Decision: the pilot clearly fails; stopped.** At eta = 0.1 the noisy
interpolant does not reduce the hard-core leakage at the accurate solve
(nor at 4 steps), and changes no other metric. Seeds 1 and 2 were not run,
no noise or path sweep followed, and JEWEL stays frozen (no follow-up
merits it). This rules out only this setting (eta = 0.1, sine schedule, 2
h, this network).

**What the diagnostic suggests (hypothesis, not tested further).** The core
stays in B_hat even when the solve starts exactly on the true path, so the
failure is not a lack of supervision *around* the paths, which is what
the noise adds. The learned velocity is the regression E[b - a | x_t] over
every training pair passing near x_t; in a hard tower the partly
subtracted state does not tell how much of it is signal, and the average
over plausible backgrounds is higher than the true one. That is a
property of the marginal (posterior-mean) field, which a noisy path of
this size does not change. Explaining it needs information the state
lacks, e.g. a signal prediction as in the joint arm, or a different
target, not a wider tube around the same regression.

## PYTHIA -> JEWEL translation pilot (set up 2026-09-28 20:00, before training)

**Question.** Can the existing unpaired methods (OT-CFM, online alpha-DSBM)
map clean PYTHIA jet images to outputs whose statistics match held-out
JEWEL jets, while the output still depends in a meaningful way on its input
jet? This is the intended application tested directly. Background
subtraction mixed transport with an inverse problem and the residual
readout S_hat = M - B_hat; here both ends are clean jet images. It is a
new, exploratory experiment:
- the failed toy closure stays a negative result; its gate is not
  reopened;
- nothing here can establish a true per-event medium modification. There
  is no matched JEWEL jet for a PYTHIA jet: any unpaired map is one choice
  among the many that reproduce the same marginals (OT-CFM: least squared
  log-energy change; alpha-DSBM: the Schrodinger bridge of |x - y|^2 / 2 at
  eps). No JEWEL jet, OT partner or nearest JEWEL jet is treated as an
  input's truth;
- PYTHIA -> JEWEL differences include generator and selection effects as
  well as medium effects (below);
- the published subtraction UVCGAN-S checkpoint is not a translation
  baseline and is not used.

**Data** (`translation_data.py`; its own caches, normalisation and runs
under `OUTDIR/sphenix/flow/translation/`; manifest and parent split lists
in `docs/flow/translation/`):
- **Parents:** a generated event, identified by (generator file, event
  number).
  - PYTHIA: the 2,635,582 events of `train/signal.h5`, whose index gives
    unique keys; 2636 files.
  - JEWEL: the 870,000 clean images of the 870 ROOT files of Zenodo
    record 17594612. 833,000 of them are embedded once each in the old
    test mixtures; mixtures are not distinct jets, the ROOT files are the
    parent list.
  - Duplicate images, by a content hash, are dropped (the first of a group
    is kept).
- **Jets** (the closure test's code, `closure_data.select` / `windows`,
  identical for both domains):
  - one jet per parent, the leading R = 0.4 cone (phi periodic);
  - axis rows 4-19, cone >= 10 GeV;
  - the 53 cone towers in GeV of tower E_T, clipped at 0 (neither sample
    has negative towers), on the 16 x 16 canvas;
  - JEWEL is rounded to float16, the precision of the PYTHIA h5.
- **Physical amplitudes:** no per-jet normalisation; neither spectrum is
  reweighted.
- **What the crop leaves out:**
  - all energy outside the cone around the leading axis: wide-angle and
    out-of-cone medium-induced radiation, recoil energy, the recoiling jet,
    the underlying event;
  - the window's towers beyond dR = 0.4;
  - everything beyond |eta| < 1.1.
  - Per jet, the event energy and the ring 0.4 < dR <= 0.8 are recorded.
- **Splits, by parent,** from one fixed permutation per domain:
  - PYTHIA train 200k, val 10k, test 20k;
  - JEWEL train 200k, val 10k, test 20k, and ref 20k (a second held-out
    JEWEL sample, only for the JEWEL-vs-JEWEL finite-sample reference);
  - the 20k JEWEL parents of the subtraction study's JEWEL evaluation are
    in no split, so that evaluation stays untouched by JEWEL training.
- **Normalisation:** psi = log(E + 0.1), standardised with one mean and sd
  over both training pools together (`translation/norm.json`; the closure's
  `norm_closure.json` is not used).
- **Known and unknown about the samples:**
  - PYTHIA: sPHENIX-style production naming (type 11, "jet30", run 19,
    no noise); its generator version, tune, the meaning of "jet30" and the
    detector simulation are not recorded.
  - JEWEL: no generator record, version, medium parameters, seeds or
    recoil treatment. The old mixed sample's name says `jet30_40_50`,
    meaning undocumented.
  - Both share the smallest tower value (1.1e-4 GeV) and the low quantiles
    of the tower spectrum, which suggests one tower pipeline (not
    documented).
  - Differences can come from the medium, the generators (JEWEL uses PYTHIA
    6 for the hard process; tunes, underlying event: JEWEL events carry
    about half of PYTHIA's total event energy, median 35 against 75 GeV),
    each sample's generator-level jet requirement, and the common 10 GeV
    selection acting on different spectra.

**Models** (seed 0, from scratch, the UVCGAN-S backbone, the same training
pools; each 2 A6000 GPU-hours of training time on dahlia):
- **OT-CFM** (`fm_train.py --method jetflow --cost l2`):
  - exact minibatch OT, TorchCFM's pair sampling, pool = batch 256;
  - one fixed cost, the squared L2 of the standardised states (jet-centred,
    the geometry of the closure's DSBM comparison);
  - straight path (sigma 0), MSE velocity loss, Adam 2e-4, warm-up 1000,
    clip 1, EMA 0.9999;
  - 120 min, checkpoints every 10 min. No cost sweep.
- **Online alpha-DSBM** (`dsbm.py`, the checked implementation):
  - bidirectional bridge, independent-pair pretraining (30 min), then
    online refinement from EMA rollouts (90 min, rollouts in the budget);
  - eps = 0.25 in the new standardised coordinates (midpoint noise sd
    0.25 in z);
  - endpoint preconditioning variances refitted on the translation pools;
  - batch 256, Adam 2e-4, warm-up 1000, clip 1, EMA 0.999, 30-step
    Euler-Maruyama rollouts. No noise sweep.
- No new backbone, conditioning, shape or conservation loss, or noisy
  path.
- **Primary comparison: the final checkpoint at the 2 GPU-h budget.**
  - Validation curves every 10 min (population scores and input
    correlations on PYTHIA val -> JEWEL val) are descriptive only.
  - Test data are used for nothing but the final scores.
  - A curve still improving at 2 h is reported as such, with no
    extrapolated time to success.

**Solver (fixed on validation, then frozen):** the final checkpoint on
2000 PYTHIA val inputs, against JEWEL val.
- OT-CFM: midpoint with 32 network evaluations is adequate if, against 64:
  - every W1 / sigma below changes by less than its bootstrap sd;
  - the per-jet cone energy moves by less than 1% of the rms energy change
    the model makes (E_out - E_in);
  - otherwise 64, checked against 128.
- alpha-DSBM: the SDE sampler with 30 steps is adequate if, against 60
  steps with coupled Brownian increments:
  - the per-jet energy difference is below 1/3 of that between two
    independent samples at 30;
  - every W1 moves by less than its bootstrap sd;
  - otherwise 60. Noise is never dropped.
- Four-step Euler for OT-CFM: a secondary speed readout only.
- NFE and latency (one A6000, batch 2000) are reported.

**Samples compared** (20k test each; outputs clipped at 0, no threshold or
clean-up for any method):
- **JEWEL test:** the target.
- **JEWEL ref:** the finite-sample floor.
- **Identity:** the PYTHIA test inputs.
- **Random JEWEL:** 20k jets of the JEWEL training pool, one per input,
  ignoring it. It shows that good target distributions alone do not make a
  translation.
- **OT-CFM:** one output per input.
- **alpha-DSBM:** one sample per input, fixed sampler seed. Eight samples
  per input on the first 1000 test inputs, to show variability. That
  spread is algorithmic, not a physical uncertainty.

**Metrics:**
- **Population, fixed crop:**
  - E, mass, girth, p_T^D, z_lead, z_g, R_g of the 53 cone towers
    (`closure_eval.Jets`, the closure's definitions);
  - W1 / sigma_JEWEL, each with a bootstrap sd (200 resamples of both
    samples);
  - the common final selection E >= 10 GeV is applied to outputs and to
    JEWEL alike, and the share of outputs passing is reported (JEWEL: 100%
    by construction), with the migration of E bins.
- **Joint:**
  - pT-girth and pT-z_lead densities and binned means;
  - W1 of girth and z_lead in pT bins (10-20, 20-30, 30-60 GeV);
  - the largest difference of the (log E, mass, girth, p_T^D, z_lead)
    correlation matrices.
- **Refound jets, kept apart from the fixed crop:** anti-kT R = 0.4
  (FastJet, `bench_jets.py`'s constituents and soft drop) on each canvas's
  towers. For the leading jet: pT, axis offset, mass, girth, p_T^D,
  z_lead, z_g, R_g, with the same final selection (pT >= 10 GeV) on both
  sides.
- **Dependence on the input** (no selection):
  - the input-output correlation of E, girth, mass, core fraction and
    leading-tower energy, with bootstrap sd;
  - the distributions of their changes, and the changes against input pT
    (a separate figure);
  - the own-input preference rate: how often an output is closer, in
    normalised-shape EMD, to its own input than to another test input
    (identity 1, random JEWEL 0.5).
  - These describe the proposed map, not errors against a truth; high
    correlation alone is no success (the identity has it).
- **Image fidelity:**
  - fixed displays: the first 1000 PYTHIA test jets nearest the 0.1, 0.3,
    0.5, 0.7 and 0.9 quantiles of their cone energy (fixed now), their
    outputs, 8 alpha-DSBM samples each, and JEWEL test jets at the same
    quantiles of the JEWEL spectrum as a visual reference, not a truth;
  - the cone tower spectrum;
  - occupancy (towers above 0, 0.01, 0.1, 1 GeV), energy in towers below
    0.5 GeV, leading-tower energy, core fraction.

**How the outcome is read (fixed now; successes and failures reported
separately, not as one number):**
- **Moves toward JEWEL** in an observable: its W1 / sigma is below the
  identity's by more than 3 combined bootstrap sd. **Consistent with
  JEWEL:** within 3 sd of the JEWEL-ref floor.
- **Input dependence beyond random JEWEL:**
  - the input correlation of E and of girth exceeds 0 by more than 3
    bootstrap sd;
  - the own-input preference rate exceeds 0.5 by more than 3 standard
    errors.
- **Artefact flags:**
  - *soft floor:* mean occupancy above 0.01 GeV, or mean energy in towers
    below 0.5 GeV, more than 10% above both PYTHIA's and JEWEL's;
  - *flattening:* mean z_lead or core fraction below both domains' by more
    than 3 sd.
- **Useful outcome:**
  - movement toward JEWEL in at least 4 of the 7 fixed-crop marginals and
    in both joint comparisons;
  - input dependence beyond random JEWEL;
  - no artefact flag.
  - Then one narrowly defined follow-up is recommended.
- **Otherwise:** the result is recorded as it is, with no new architecture
  or hyperparameter search.

### Data as built (job 20331, 20:17; `docs/flow/translation/manifest.json`)

| | PYTHIA | JEWEL |
| :--- | ---: | ---: |
| parents (unique keys) / files | 2,635,582 / 2636 | 870,000 / 870 |
| exact duplicate images dropped | 190,584 | 0 |
| leading axis outside rows 4-19 | 329,869 | 93,683 |
| leading cone below 10 GeV | 522 | 43,306 |
| selected (share) | 2,305,191 (87.5%) | 733,011 (84.3%) |
| eligible (selected, unique, not reserved) | 2,134,675 | 716,290 (16,721 reserved) |
| used: train / val / test / ref | 200k / 10k / 20k / - | 200k / 10k / 20k / 20k |
| cone E quantiles 5 / 50 / 95%, GeV | 23.0 / 31.9 / 44.0 | 12.7 / 26.4 / 44.2 |
| event energy, median, GeV | 74.3 | 51.3 |
| cone share of the event energy, median | 0.43 | 0.51 |
| ring 0.4 < dR <= 0.8, mean GeV (share of the cone) | 2.83 (9.3%) | 2.09 (9.2%) |

- **Duplicates:** 7.2% of the PYTHIA images are exact copies of an image in
  another file.
  - They come in pairs only, 295-496 per file across 955 files, never
    within one file, in shuffled order: 397 events of file 1954 reappear in
    file 2448 under other event numbers (the same number in only 194 of all
    pairs).
  - One of each pair is kept, so no image can sit in two splits.
  - No near-copy (same leading tower, total energy within 0.1%) was found
    among the other events of two such file pairs.
  - JEWEL has none, and no image is shared between the domains.
  - The parent key alone would not have caught these copies.
- **Spectra:** PYTHIA's cone energy is cut sharply near 20 GeV (5% below 23
  GeV: a generator-level jet requirement seen through the detector
  response). JEWEL reaches down to the 10 GeV selection (5% below 12.7 GeV;
  5% of JEWEL parents fail it, 0.02% of PYTHIA's).
- **The crop drops about half the event energy in both samples.** The ring
  just outside the cone holds 9% of the cone energy in both.
- **Normalisation:** psi mean -2.126, sd 0.629. Per-element E[z^2] (the
  DSBM endpoint variances): PYTHIA 1.088, JEWEL 0.911. eps = 0.25 is a
  midpoint noise sd of 0.25 in z, 0.157 in log(E + 0.1).
- **Checks:** float16 caches reproduce every canvas energy exactly; JEWEL's
  float16 rounding is at most 4.9e-4 relative; no negative tower in either
  sample; the 16,721 reserved JEWEL parents are in no split.
- The tower spectra of both samples have discrete peaks below ~0.01 GeV (a
  property of the simulation). log(E + 0.1) compresses that region
  (psi(0.01) - psi(0) = 0.1), so no model in these coordinates can resolve
  it; those towers carry < 0.1% of the energy.
- **Build:** the JEWEL ROOT files are read once into a flat cache (45 s,
  30 processes, each file read whole and parsed in memory).
  - uproot's default handler re-read ~0.7 GB per 1.5 MB file through NFS,
    and a memory map faulted page by page.
  - Under SLURM the forked workers came up pinned to one core; the pools
    now restore the parent's CPU mask.

### Pilot results (jobs 20333-20334 training, 20340-20343 solver checks and outputs, 20344-20347 report)

**Training** (one A6000 each, side by side on dahlia, 20:22-22:23;
`docs/flow/translation/cost.csv`, `configs/`):

| model | stage | GPU h | updates | updates / s | overhead | peak GB |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: |
| OT-CFM (32.3M parameters) | train | 2.00 | 122,720 | 17.0 | OT matching 10.4% | 3.5 |
| alpha-DSBM (32.8M) | bridge pretraining | 0.50 | 26,058 | 14.5 | | 3.6 |
| | online refinement | 1.50 | 8,761 | 1.62 | rollouts 89% (7680 NFE per update) | 3.8 |

**Validation curves** (`curves.csv`, `tr_curves.png`; 10k PYTHIA val inputs
against 10k JEWEL val, every 10 min; mean W1/sigma of the seven
observables; val references: identity 0.321, JEWEL-train draws 0.016):
- **OT-CFM at 32 NFE:**
  - 0.034 at 10 min, 0.024-0.025 from 40 min to 2 h (flat);
  - E 0.092 -> 0.041; mass 0.03-0.04 until 1.3 h, then 0.056;
  - input correlations: E 0.935 -> 0.947, girth 0.990-0.992.
  - With 4 Euler steps: 0.26-0.27 throughout (E 0.54-0.59).
- **alpha-DSBM:**
  - pretraining 0.132 -> 0.101 (30 min);
  - refinement first jumps to 0.162 (50 min), then declines to 0.137 at
    2 h, still falling (0.002-0.004 per 10 min);
  - over the last 70 min: E 0.355 -> 0.223 and mass 0.465 -> 0.327
    improve, while p_T^D 0.100 -> 0.138 and z_lead 0.082 -> 0.118 worsen;
  - the girth correlation jumps from 0.93 to 0.98 when refinement starts.
  - At the budget, its distributions are further from JEWEL than its own
    pretrained bridge's (0.137 against 0.101), with a tighter coupling.
  - Whether it recovers with more time is open.

**Solver, fixed on validation before any test output** (`solver_check.csv`,
`_ode64`, `_ode128`; 2000 val inputs, 200 bootstraps):

| check | per-jet cone E difference, rms GeV | relative to | largest W1/sigma shift (its sd) | reading |
| :--- | ---: | :--- | :--- | :--- |
| OT-CFM 32 vs 64 NFE | 0.52 | 8.4% of the E change (6.2 GeV rms) | mass 0.027 (0.013) | not adequate |
| OT-CFM 64 vs 128 | 0.10 | 1.6% | z_g 0.012 (0.015) | not adequate (energy) |
| OT-CFM 128 vs 256 | 0.035 | 0.54% | R_g 0.004 (0.021) | **adequate: 128 NFE frozen** |
| OT-CFM 4 Euler vs 256 | 6.38 | 98% | mass 0.66 | does not translate |
| alpha-DSBM 30 vs 60 steps, same noise | 0.55 | 0.09 of two independent samples (6.01) | z_g 0.004 (0.019) | **adequate: 30 steps frozen** |

The pre-registered rule named 64 NFE as the fallback, checked against 128.
64 failed the energy criterion narrowly, so the doubling was continued
to 128, which passes against 256. Latency on one A6000 at batch 2000:
- OT-CFM: 6.58 ms per jet at 128 NFE, 0.21 at 4 Euler steps;
- alpha-DSBM: 1.55 ms (30 steps).

The validation curves above were drawn at 32 NFE (descriptive). At 128 NFE
the final OT-CFM checkpoint scores E 0.044 and mass 0.030 on the check's
2000 validation inputs.

**Test: distributions of the fixed-crop observables** (`population.csv`,
`tr_marginals.png`; 20k each; W1/sigma to JEWEL test; every sample cut at
cone E >= 10 GeV; bootstrap sd 0.004-0.011):

| | pass | E | mass | girth | p_T^D | z_lead | z_g | R_g |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JEWEL ref (floor) | 1 | 0.011 | 0.014 | 0.009 | 0.018 | 0.019 | 0.012 | 0.008 |
| random JEWEL | 1 | 0.019 | 0.024 | 0.011 | 0.010 | 0.011 | 0.005 | 0.010 |
| identity | 1 | 0.546 | 0.881 | 0.105 | 0.264 | 0.232 | 0.034 | 0.158 |
| **OT-CFM** | 0.995 | 0.025 | 0.021 | 0.012 | 0.010 | 0.008 | 0.017 | 0.012 |
| OT-CFM, 4 Euler (speed readout) | 0.974 | 0.510 | 0.678 | 0.129 | 0.125 | 0.099 | 0.010 | 0.249 |
| **alpha-DSBM** | 0.999 | 0.256 | 0.325 | 0.056 | 0.112 | 0.099 | 0.036 | 0.073 |

- **OT-CFM:** every observable is within 3 sd of the JEWEL-vs-JEWEL floor.
  z_g was already consistent for the identity: PYTHIA and JEWEL barely
  differ in it here.
- **alpha-DSBM:** closes 51-64% of the identity's gap in the six others
  and stays 5-23x the floor.
- The E panel of `tr_marginals.png` also shows the outputs below 10 GeV,
  which the cut removes.

**Joint structure** (`joint.csv`, `tr_joint.png`; W1/sigma of girth and
z_lead in cone-energy bins):

| | girth: 10-20 / 20-30 / 30-60 GeV | z_lead: 10-20 / 20-30 / 30-60 | correlation matrix, max diff |
| :--- | :--- | :--- | ---: |
| JEWEL ref | 0.032 / 0.044 / 0.019 | 0.022 / 0.033 / 0.019 | 0.022 |
| identity | 0.158 / 0.156 / 0.329 | 0.474 / 0.401 / 0.369 | 0.317 |
| OT-CFM | 0.023 / 0.012 / 0.020 | 0.030 / 0.025 / 0.016 | 0.015 |
| alpha-DSBM | 0.095 / 0.071 / 0.106 | 0.129 / 0.161 / 0.184 | 0.128 |

At fixed energy JEWEL jets are narrower and harder than PYTHIA's (the
binned means, `tr_profiles.png`). OT-CFM reproduces both trends in every
bin; alpha-DSBM gets about halfway.

**Refound jets** (`refound.csv`, `tr_refound.png`; the leading anti-kT
R = 0.4 jet found in each canvas, pT >= 10 GeV on every sample):
- acceptance: JEWEL test 0.976, ref 0.974, OT-CFM 0.978, alpha-DSBM 0.996,
  identity 1;
- mean axis offset from the canvas centre: 0.127 in JEWEL and OT-CFM;
- OT-CFM: pT, mass, girth, p_T^D, z_lead, z_g, R_g all within the floor
  (W1/sigma 0.006-0.031);
- alpha-DSBM: 0.03-0.34 (identity 0.04-0.82).

**Dependence on the input** (`dependence.csv`, `tr_changes.png`,
`tr_migration.png`; 20k test jets, no selection):

| | corr. E | girth | mass | core | leading tower | own input preferred | shape EMD to its input / to another input | E_out/E_in |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| identity | 1 | 1 | 1 | 1 | 1 | 1 | 0 / 0.454 | 1 |
| random JEWEL | 0.006 | -0.004 | 0.007 | -0.000 | -0.001 | 0.498 | 0.466 / 0.466 | 0.88 +- 0.37 |
| OT-CFM | 0.944 | 0.989 | 0.912 | 0.991 | 0.917 | 1.000 | 0.041 / 0.465 | 0.82 +- 0.15 |
| OT-CFM, 4 Euler | 0.865 | 0.989 | 0.823 | 0.990 | 0.886 | 1.000 | 0.047 / 0.458 | 0.67 +- 0.12 |
| alpha-DSBM | 0.800 | 0.983 | 0.880 | 0.982 | 0.792 | 1.000 | 0.056 / 0.458 | 0.91 +- 0.15 |

- Bootstrap sd of a correlation <= 0.003; the Spearman values agree
  within 0.03.
- **OT-CFM's energy change is set by the two spectra.** Median E_out/E_in
  by input energy:
  - 0.57 at 10-20 GeV, 0.60 at 20-25, 0.73 at 25-30, 0.89 at 30-40, 1.00
    at 40-50, 1.04 at 50-70.
  - PYTHIA's cone energy starts near 20 GeV and JEWEL's reaches down to
    10. The near-monotone map sends PYTHIA's lower edge onto JEWEL's
    10-20 GeV tail and leaves the top almost unchanged.
  - That is a quantile-like map between two differently selected samples,
    not an energy-loss mechanism.
  - High-energy jets gain a harder core, as JEWEL's high-energy jets have
    one. The median leading-tower change is +2.2 GeV at 40-50 GeV and
    +3.7 at 50-70, against -1.9 at 20-25.
- **alpha-DSBM:** 0.78-0.94, flatter; hence it misses JEWEL's low-E tail.
- **Share of outputs with E >= 10 GeV** (`migration.csv`):
  - OT-CFM: 0.16 for 10-15 GeV inputs, 0.70 at 15-20, >= 0.996 above;
  - alpha-DSBM: 0.41, 0.98 and 1.
  - Only 279 of the 20k inputs are below 20 GeV.
- **alpha-DSBM's variability** (`variability.csv`, `tr_samples.png`; 8
  samples of each of the first 1000 test inputs):
  - the spread of one input's outputs, in units of JEWEL's population sd:
    E 0.37, mass 0.33, leading tower 0.40, z_lead 0.36, p_T^D 0.29, girth
    0.13, core 0.14;
  - two samples of a jet differ more (shape EMD 0.068) than a sample
    differs from its input (0.056);
  - the 8-sample mean follows the input better than one sample (E
    correlation 0.91 against 0.80). It is reported, never used as a
    result.
  - This is the sampler's variability, not a physical uncertainty.

**Image fidelity** (`fidelity.csv`, `tr_towers.png`, `tr_displays.png`; per
jet, the 53 cone towers):

| | towers > 0 | > 0.01 GeV | > 0.1 | > 1 | energy in towers < 0.5 GeV | leading tower, GeV | core fraction | z_lead |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JEWEL test | 32.9 | 28.6 | 18.4 | 5.85 | 2.71 | 9.68 | 0.435 | 0.342 |
| identity | 36.3 | 32.0 | 21.4 | 7.35 | 2.98 | 10.15 | 0.422 | 0.307 |
| OT-CFM | 42.8 | 28.7 | 18.5 | 5.86 | 2.72 | 9.68 | 0.434 | 0.341 |
| alpha-DSBM | 47.5 | 30.4 | 19.9 | 6.46 | 2.91 | 9.97 | 0.438 | 0.327 |

- **No soft floor or flattening flag.** OT-CFM matches JEWEL in every
  class above 0.01 GeV, and its tower-energy spectrum lies on JEWEL's.
- **Below 0.01 GeV it cannot make exact zeros.**
  - It has 10.2 empty cone towers per jet against JEWEL's 20.1, and 14.1
    towers of ~0.002 GeV against 4.4.
  - Those towers carry 0.024 GeV per jet, against 0.022 for JEWEL: 0.1% of
    the jet energy.
  - alpha-DSBM: 5.5 empty towers, 0.040 GeV.
  - This is the log(E + 0.1) representation (psi(0.01) - psi(0) = 0.1);
    the discrete sub-0.01 GeV peaks of the data are not reproduced
    either.
- **Displays:** each output keeps its input's prongs and layout. OT-CFM
  rescales the energy and sharpens the core of hard jets; the alpha-DSBM
  samples of one jet vary visibly (e.g. 36.8-49.2 GeV for a 40.4 GeV
  input).

**Reading, as fixed before training** (`verdict.csv`):
- **OT-CFM: a useful outcome on every pre-registered item.**
  - It moves toward JEWEL in 6 of 7 marginals; z_g was already within the
    floor.
  - All 7 are consistent with JEWEL, and so are both joint comparisons.
  - Its input dependence goes beyond random JEWEL, with no artefact flag.
- **alpha-DSBM: also meets the letter of the rule,** with 6 of 7
  marginals and both joints moved, input dependence, and no flag. But it
  stays 5-23x the floor.
- **The 4-Euler readout fails:** 4 of 7 marginals moved, and the girth
  joint did not.
- (A pT bin with fewer than 50 jets would have made a joint comparison
  undefined; none had.)

**What this establishes.** Trained unpaired for 2 GPU-hours, OT-CFM with the
UVCGAN-S backbone maps clean PYTHIA jets to outputs that are, within the
finite-sample precision of 20k jets, indistinguishable from held-out JEWEL
jets:
- in seven fixed-crop and seven refound-jet observables;
- in the energy dependence of girth and z_lead;
- while each output stays tied to its own input: correlations 0.91-0.99,
  outputs 11x closer in shape to their own input than to another.
- This holds for this crop, this pair of samples and one seed.
- alpha-DSBM at eps = 0.25, with the same budget, is clearly worse on the
  distributions and no better on dependence. Its refinement had not
  converged.

**What it does not establish:**
- **A per-jet medium modification.**
  - There is no truth to compare a jet with. The OT-CFM map is the
    least-change transport in standardised log E, a modelling choice.
  - Its high input correlation is partly by construction: the closure test
    showed that the cost decides the per-jet correspondence, and no
    unpaired coupling tried there reproduced a known per-jet modification.
- **Physics in the energy change.** It follows the two samples' spectra
  (generator-level jet requirements, generators), inside a crop that drops
  out-of-cone energy.
- **Robustness:** one seed. alpha-DSBM's limit at a larger budget is open.

**Recommended follow-up (one, narrow):** OT-CFM seeds 1 and 2 with the same
data, cost, budget and 128-NFE solve. Report the seed spread of every
metric above, and the per-jet differences between the three maps' outputs
for the same test jets. The question is whether the *per-jet* map, not only
its distributions, is reproducible. No new architecture, cost or
noise-level search follows from this pilot.

    sbatch -w saturn scripts/flow/translation_data.sbatch          # data set
    KIND=otcfm LABEL=tr_otcfm_s0 MINUTES=120 SEED=0 \
        sbatch -w dahlia -J tr_otcfm_s0 scripts/flow/translation_run.sbatch
    KIND=dsbm LABEL=tr_dsbm_e025_s0 PRE_MINUTES=30 MINUTES=90 EPS=0.25 SEED=0 \
        sbatch -w dahlia -J tr_dsbm_e025_s0 scripts/flow/translation_run.sbatch
    MODE=check sbatch -w dahlia scripts/flow/translation_post.sbatch     # 32 vs 64
    MODE=check ODE=64:midpoint RUNS=tr_otcfm_s0 LABELS=OT-CFM \
        CHECK_FILE=solver_check_ode64.csv sbatch -w dahlia scripts/flow/translation_post.sbatch
    (the same with ODE=128:midpoint, CHECK_FILE=solver_check_ode128.csv)
    MODE=generate ODE=128:midpoint STEPS=30 sbatch -w dahlia scripts/flow/translation_post.sbatch
    ODE=midpoint128 STEPS=30 sbatch -p a6k -w saturn -c 32 --mem=64G \
        scripts/flow/translation_report.sh                           # report + tables
    cd docs/flow/translation/slides && ~/pyext/tectonic_env/bin/tectonic translation_appendix.tex

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

alpha-DSBM closure test (`dsbm.py`, `dsbm_eval.py`; per run
`OUTDIR/sphenix/flow/<label>/`: `config.json` with eps, the midpoint noise
sd and the endpoint variances, `history.csv`, `summary.json` with rollout
time and NFE, `checkpoints/`, `evals/closure_{val,train}_sde30.csv`):

    $PYTHON scripts/flow/dsbm_gauss.py --out docs/flow/dsbm/diag_L3 --model linear \
        --precond-scales --refine 20000 --refine-lr 3e-5 --batch 512 --warmup 1000
    j=$(LABEL=dsbm_pre_e1_s0 STAGE=pretrain MINUTES=30 EPS=1.0 SEED=0 \
        sbatch -w dahlia scripts/flow/dsbm_run.sbatch | awk '{print $4}')
    LABEL=dsbm_ref_e1_s0 STAGE=refine INIT=dsbm_pre_e1_s0 MINUTES=90 EPS=1.0 \
        sbatch -w dahlia --dependency=afterok:$j scripts/flow/dsbm_run.sbatch
    LABEL=dsbm_cont_e1_s0 STAGE=continue INIT=dsbm_pre_e1_s0 MINUTES=90 EPS=1.0 \
        sbatch -w dahlia --dependency=afterok:$j scripts/flow/dsbm_run.sbatch
    # (eps 0.25: the same with EPS=0.25 and labels dsbm_*_e025_s0)
    LABEL=closure_uvcgan_l2_s0 COST=l2 SEED=0 MINUTES=120 \
        EVAL_ARGS="--setting 32:midpoint --train-n 5000" sbatch -w dahlia \
        scripts/flow/closure_run.sbatch --backbone uvcgan
    RUNS="closure_paired_s0 ... dsbm_ref_e025_s0" LABELS="..." \
        OUT=docs/flow/dsbm/closure_test FIG_RUNS="..." FIG_LABELS="..." \
        sbatch -w dahlia scripts/flow/dsbm_report.sbatch   # the job 20322 list
    $PYTHON scripts/flow/dsbm_tables.py; $PYTHON scripts/flow/dsbm_curves.py ...
    $PYTHON scripts/flow/dsbm_gauss_plot.py LABEL=docs/flow/dsbm/diag_X ...
    cd docs/flow/dsbm/slides && ~/pyext/tectonic_env/bin/tectonic dsbm_appendix.tex

Paired noisy-interpolant pilot: `docs/flow/bench/README.md` (end).

PYTHIA -> JEWEL translation pilot: `docs/flow/translation/README.md` and the
end of its section above.

## Status (2026-09-28 23:00)

All runs have ended; nothing is running.

- **Latest: the PYTHIA -> JEWEL translation pilot** (section above;
  appendix `docs/flow/translation/slides/translation_appendix.pdf`).
  - **Data:** its own set of clean jets, split by parent; 7.2% of the
    PYTHIA images were exact copies across files and were dropped.
  - **OT-CFM** (2 GPU h, 128-NFE ODE):
    - its outputs match held-out JEWEL within the JEWEL-vs-JEWEL
      finite-sample floor in 7 fixed-crop and 7 refound-jet observables,
      and in girth and z_lead against energy;
    - each output stays tied to its own input (correlations 0.91-0.99);
    - no soft floor above 0.01 GeV;
    - its energy change follows the two samples' spectra, not a medium
      mechanism.
  - **alpha-DSBM** (eps 0.25) gets about halfway; its refinement had not
    converged.
  - **Pre-registered reading:** useful. The one follow-up recommended is
    OT-CFM seeds 1-2 (is the per-jet map reproducible?). Nothing about a
    per-jet medium modification is established.
- **Earlier today, two single-question experiments, both negative and both
  stopped as pre-registered:**
  - **Online alpha-DSBM on the closure test** (appendix
    `docs/flow/dsbm/slides/dsbm_appendix.pdf`):
    - single samples are 0.097 (eps = 1) and 0.077 (eps = 0.25) in shape
      EMD, against 0.092 for OT-CFM with the same backbone and 0.032 for
      doing nothing;
    - gate failed.
  - **Paired noisy-interpolant pilot** (deck slides 11-13): eta = 0.1
    changes nothing measurable; the core is lost even from states on the
    true path.
- **Consolidated benchmark** (still the reference for subtraction; deck
  `docs/flow/bench/slides/bench_deck.pdf`):
  - no flow arm is as faithful as UVCGAN-S;
  - predicting the background alone loses 12-25% of the hardest towers'
    energy, paired or not; predicting both keeps it but drops soft signal
    and low-pT jets.
- **Missing:** ICS (fjcontrib), the paper's own analysis code.
- **Open:**
  - whether the OT-CFM translation's per-jet map is reproducible across
    seeds and how it depends on the cost (the closure test showed the cost
    decides the per-jet correspondence);
  - a paired HYBRID sample with several medium realisations per vacuum
    shower;
  - why the background-only flow's field stops short in the core.
