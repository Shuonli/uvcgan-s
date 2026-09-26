# Summary: speeding up UVCGAN-S, and flow matching for sPHENIX background subtraction

*As of 2026-09-26, 08:20. Branch `ddp` of github.com/Shuonli/uvcgan-s.
This file is the short version. The full logs, with job numbers and
commands, are `SCALING_NOTES.md` (UVCGAN-S training) and `FLOW_NOTES.md`
(flow matching).*

## The task

Each sPHENIX calorimeter event is a 24 x 64 grid of tower energies. A
mixed event is a PYTHIA jet event (the signal) added tower by tower to a
HIJING heavy-ion event (the background). A model takes the mixture and
returns the two pieces. UVCGAN-S, a CycleGAN variant, is the current
model.

Every model is scored the same way, on held-out events whose answer we
know:

- **Jet energy resolution (`jer_cal`, GeV, lower is better)**: the energy
  in an R = 0.4 cone around the true jet axis, extracted against true,
  after a linear calibration. This is the main number. A simple median-rho
  subtraction gives 5.28 GeV; the published UVCGAN-S gives 3.59.
- **Per-tower error of the extracted jet image**: MAE (`l1_sig`) and MSE,
  in GeV.
- **Jet substructure, jet by jet**: mass, girth, p_T^D, core fraction,
  leading-tower share (z_lead), number of towers above 1 GeV (n1), and
  soft-drop z_g and R_g.
- **Two test sets**:
  - **val**: 20k PYTHIA-in-HIJING mixtures, the same kind as the training
    data.
  - **JEWEL**: quenched jets in HIJING, which no model ever trained on. It
    tests whether a model still works when the jets look different from
    PYTHIA.

Before any flow was trained, three targets were fixed: 4.00, 3.70 and
3.65 GeV ("useful", "acceptable", "matches the baseline"). It was also
fixed that JEWEL is used only as a final test, never to choose settings.

Names used below:

- **Synthetic mixture**: a HIJING event plus a PYTHIA event, added. It
  gives unlimited exact training pairs. UVCGAN-S already uses these in its
  `idt-aa` loss.
- **OT-CFM** (optimal-transport flow matching), trained **unpaired**. Each
  batch pairs real mixtures with unrelated HIJING and PYTHIA events,
  choosing the pairs that change the images least. The model then learns
  to move each mixture toward its pair.
- **SB-CFM**: the same idea with a softer (entropic) pairing.
- **Conditional CFM**: learns to draw many plausible (background, jet)
  splits of a given mixture. It is trained on synthetic mixtures, and its
  estimates are averages over samples.

## Part 1: making UVCGAN-S train faster (`SCALING_NOTES.md`)

- **Train 150-180k updates at batch 4 (20-24 h on one A6000) instead of
  the configured 800k (~105 h).** This matches the published model on
  JEWEL (3.99 GeV) and comes within 1-2% on val (3.63-3.65 against 3.59),
  for 4.5-5x less compute.
- **Batch 32** is ahead on val early: it reaches 3.70 GeV in 8-15 h
  instead of 15-17 h. But it stalls on JEWEL at 4.1-4.2 GeV.
- **Step-speed tricks:**
  - Caching the gradient penalty makes steps 1.3x faster but the model
    worse.
  - `torch.compile` gives 1.07-1.15x.
  - Mixed precision, TF32 and fused Adam do not help: a step is limited by
    the number of small GPU kernels, not by arithmetic.
- **Multi-GPU:** DDP gives only ~1.12x for two GPUs, and two nodes are
  slower than one GPU. Extra GPUs are better spent on parallel
  experiments.
- **Seeds:** UVCGAN-S's final numbers are reproducible across seeds
  (+-0.02-0.03 GeV), but its path is not:
  - the time to reach 3.70 GeV varies from 8.4 to 15.2 h;
  - the raw network jumps by 0.1-0.3 GeV between checkpoints;
  - the averaged (EMA) network is unusable for the first ~5 h.

## Part 2: flow matching against UVCGAN-S (`FLOW_NOTES.md`)

The questions:
- Can a flow-matching model reach UVCGAN-S's jet resolution in fewer
  GPU-hours?
- Does it train more stably?
- At what inference cost?

All flows use the same 21.6M-parameter U-Net on one A6000.

### What we tried

"Time to 3.70" is the training time until val `jer_cal` first reaches
3.70 GeV, confirmed at the next evaluation.

| model | learns from | best val `jer_cal` | JEWEL | jet image MAE (val) | time to 3.70 | ms / event |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| UVCGAN-S published (800k updates, ~105 h) | unpaired + synthetic mixtures | 3.59 | 3.99 | 0.033 | | 0.27 |
| UVCGAN-S batch 4, 3 seeds (21-23 h) | same | 3.62-3.66 | 3.91-4.03 | 0.032-0.033 | 14.7-16.7 h | 0.27 |
| OT-CFM, unpaired, jet = mixture - background, 4 steps, 3 seeds | real mixtures, HIJING, PYTHIA; no pairs | 3.67-3.68 | **3.55-3.56** | 0.15 | 0.75-1.25 h | 2.0 |
| the same with one panel (mixture -> HIJING only), 3 seeds | real mixtures, HIJING | 3.69-3.70 | 3.58-3.61 | 0.155 | 0.75 h (1 of 3 seeds; the others hover at 3.70-3.73) | 2.0 |
| SB-CFM, 20-min pilot | as OT-CFM | 6.69 | | 0.060 | | 8.1 |
| conditional CFM, average of 16 samples, 3 seeds | synthetic mixtures | **3.60 / 3.60 / 3.60** | 4.25-4.28 | 0.036 | <= 1 h | 65 |
| regression, L1 in GeV (UVCGAN-S's `idt-aa` loss), 3 seeds | synthetic mixtures | 3.79-3.81 | 4.29-4.35 | **0.026** | not reached (plateau) | 0.5 |
| regression, squared error in log units | synthetic mixtures | 4.12 | 4.17 | 0.032 | | 0.5 |
| "true pieces in the batch" flow, 3 seeds | synthetic mixtures, pieces shuffled | 5.08-5.26 | 4.91-5.58 | 0.033 | | 8.1 |

### What we observed

1. **Unpaired matching cannot tell which jet belongs to which mixture.**
   - The PYTHIA event paired with a mixture has its jet in the right place
     only 3-4.5% of the time, which is what random pairing gives.
   - The reason is that the jet covers only ~50 of the 1536 towers, so the
     "least change" choice is decided almost entirely by the background.
   - As a result, the "jet" half of the model learns the average PYTHIA
     event, the same for every mixture, and is useless.
   - We checked this directly. With an empty starting image, the jet half
     adds the same cost to every mixture, so the pairing is the identical
     permutation with or without it (8 of 8 batches).

2. **The background half works, and that is enough.** It learns "the
   nearest HIJING-like background", which removes the jet. Reading the
   jet as mixture minus background gives 3.67-3.68 GeV on val. It gets
   there in under 1.5 h of training, with no paired data at all.

3. **That unpaired model is the most robust one.** On JEWEL it scores
   3.55 GeV: better than on val, and 0.35-0.45 GeV better than any other
   model. It never learned what a PYTHIA jet looks like, so a quenched jet
   cannot mislead it. It simply removes whatever does not look like
   background.

4. **Every model that learned from PYTHIA jets loses 0.3-0.7 GeV on
   JEWEL.** This covers UVCGAN-S, conditional CFM and both regressions.
   The more closely a model fits PYTHIA, the more it loses.

5. **Fewer solver steps give better jet energies; more steps give a
   cleaner image.**
   - With 4 coarse Euler steps, OT-CFM reaches 3.68 GeV but gives a noisy
     image (MAE 0.15).
   - Solved to convergence, the image is much cleaner (MAE 0.05), but 30%
     of the jet energy is lost (4.08 GeV).
   - The coarse steps act like an average over plausible backgrounds; the
     full solve picks one sharp background.

6. **The noisy image can be cleaned afterwards without hurting the jets.**
   Keep the extracted jet only near jet seeds (towers whose cone energy
   exceeds 8-10 GeV), and only in towers above 0.5-0.7 GeV. MAE drops from
   0.15 to 0.037, and the jet numbers get slightly better (val 3.64, JEWEL
   3.48). The seed threshold has to sit below the analysis' jet threshold:
   12 GeV seeds missed 2.4% of the softer JEWEL jets.

7. **The fair comparison, after you pointed out the unfair one.** Here the
   same clean-up is applied to UVCGAN-S's output as well:
   - **Image:** UVCGAN-S's is cleaner (MAE 0.029 against 0.037-0.038;
     off-jet error 33 against 44-46 GeV per event).
   - **Val jet resolution:** a tie (3.60-3.66 for both).
   - **JEWEL:** OT-CFM is better by 0.35-0.5 GeV (3.47-3.50 against
     3.82-3.97).
   - **Training time:** OT-CFM gets there in 15-75 minutes, UVCGAN-S in
     8-17 hours.

   The earlier "2x better background" claim compared different rules and
   has been withdrawn.

8. **Supervised training on synthetic mixtures is the fast route to
   in-distribution accuracy.**
   - A plain regression with UVCGAN-S's own L1 loss reaches 3.80 GeV in
     under an hour, with the smallest per-tower errors of all (MAE 0.026).
     Then it plateaus.
   - Conditional CFM reaches 3.60 GeV (the published model's level) on all
     three seeds in 2.5-3 h, when 16 samples are averaged.
   - The catches:
     - both lose on JEWEL because of the PYTHIA prior;
     - conditional CFM needs 128 network passes per event (65 ms, against
       0.27 ms for UVCGAN-S).

9. **Training stability: the flows are clearly better.** Every flow and
   regression agrees across three seeds to +-0.01-0.03 GeV at every
   matched training time, with smooth, monotone curves. For UVCGAN-S, see
   Part 1.

10. **"True pieces in the batch" is supervised learning in disguise, and
    a poor version of it.**
    - If each batch holds the actual HIJING and PYTHIA pieces of its
      mixtures, shuffled, the matching finds all of them (100%).
    - But a flow that starts from a fixed mixture with known targets heads
      for the average split.
    - The image is clean (MAE 0.033), but the jet energy is poor and
      unstable: 5.1-5.3 GeV, jumping to 6.6 between two checkpoints.
    - More solver steps make it worse.
    - On JEWEL it pulls jets toward PYTHIA: the JEWEL-minus-PYTHIA energy
      difference comes out with the wrong sign.

11. **Two variants that are trades, not fixes.**
    - **One panel instead of two:** dropping the useless jet panel gets to
      ~3.70 GeV in 15 minutes with 18% less memory, but ends 0.015-0.04 GeV
      worse.
    - **log(E+1) instead of log(E+0.1):** a slightly cleaner image
      (MAE -7%) for 0.1 GeV worse jets.

12. **Substructure, jet by jet.**
    - No model measures z_g, R_g or the jet mass per jet at this
      granularity and background: the per-jet resolution is 0.8-1.2 of the
      true spread. Their distributions and averages are measurable.
    - Every extraction dilutes the quenching signal. The table below gives
      the share of the true JEWEL-minus-PYTHIA difference that survives,
      for jets with 20-30 GeV true cone energy (1.0 = fully kept):

    | | p_T^D | z_lead | R_g | mass |
    | :--- | ---: | ---: | ---: | ---: |
    | OT-CFM, seeds + threshold | 0.82-0.88 | 0.86-0.93 | 0.68-0.80 | 0.39-0.41 |
    | UVCGAN-S published | 0.81 | 0.84 | 0.74 | 0.64 |
    | conditional CFM, average of 16 | 0.73 | 0.75 | 0.77 | 0.49 |
    | L1 regression | 0.77 | 0.81 | 0.79 | 0.50 |

13. **Vacuum -> medium with unpaired OT (the physics idea you raised).**
    - Matching PYTHIA jets to JEWEL jets pairs them by position on full
      images and by shape on jet-centred images. It pairs them only weakly
      by energy (rank correlation 0.1-0.6, depending on the
      representation).
    - So what an unpaired flow would call "the modification" is set by the
      choice of distance, not by the data, and unpaired data cannot test
      that choice.

14. **Which single answer is "best" depends on the question.**
    - For jet energy, the average over plausible splits wins.
    - For per-tower error, the per-tower median wins.
    - For a realistic-looking image, one sample wins.
    - UVCGAN-S (L1 + adversarial loss) gives one sharp answer that is a
      good compromise on all three. That is why it looks good everywhere,
      although it is not the best at any single one.

### Bottom line of Part 2

- **Speed and stability:** flows reach UVCGAN-S's jet resolution in far
  fewer GPU-hours (15 min to 3 h, against 8-17 h), and they train much
  more stably.
- **Unpaired OT-CFM** (jet = mixture - background, 4 steps, cleaned) is
  the robust choice: a tie on val, the best on JEWEL, and cheap (2 ms per
  event). Its image is somewhat noisier than UVCGAN-S's.
- **Conditional CFM** is the most accurate in distribution, but costly at
  inference and weak on JEWEL.
- **UVCGAN-S** still has the cleanest single image.

## Part 3: a model for the best single-event fidelity (done; kept as a reference)

Your request: newer technology, the best preservation of each event's jet
energy and high-dimensional substructure, and stable training.

### Step 1 (done): what the existing sampler already gives, jet by jet

We drew 16 samples per event from the trained conditional CFM, on 10k val
and 10k JEWEL events, and scored every jet against truth.

Each number is sigma(estimate - truth) / sigma(truth): 0 is perfect, and 1
means the estimate carries no information. "energy" here is the
uncalibrated cone energy; the calibrated `jer_cal` is in Part 2.

**Val** (PYTHIA, like training):

| | energy | mass | girth | p_T^D | core | z_lead | n1 | z_g | R_g |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sampler: observables of the average image of 16 | **0.51** | **0.78** | **0.25** | **0.31** | **0.21** | **0.26** | 0.83 | 1.15 | 0.86 |
| sampler: each observable averaged over the 16 samples | **0.51** | 0.79 | 0.25 | 0.32 | 0.22 | 0.27 | **0.72** | **0.87** | **0.77** |
| sampler: one sample | 0.69 | 1.03 | 0.36 | 0.48 | 0.31 | 0.40 | 0.97 | 1.19 | 0.95 |
| UVCGAN-S published | 0.53 | 0.83 | 0.27 | 0.36 | 0.23 | 0.29 | 0.83 | 1.10 | 0.86 |
| OT-CFM one panel, 4 steps | 0.53 | 0.82 | 0.26 | **0.31** | 0.23 | **0.26** | 0.85 | 1.07 | 0.89 |
| L1 regression | 0.54 | 0.87 | 0.34 | 0.40 | 0.29 | 0.35 | 0.83 | 1.16 | 0.91 |

**JEWEL** (quenched jets, never trained on):

| | energy | mass | girth | p_T^D | core | z_lead | n1 | z_g | R_g |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sampler: observables of the average image | 0.41 | 0.84 | **0.29** | 0.40 | **0.25** | 0.34 | 0.83 | 1.16 | 0.83 |
| sampler: each observable averaged | 0.41 | 0.84 | 0.33 | 0.55 | 0.27 | 0.49 | **0.77** | **0.90** | **0.75** |
| sampler: one sample | 0.53 | 1.07 | 0.49 | 0.78 | 0.39 | 0.70 | 1.01 | 1.20 | 0.95 |
| UVCGAN-S published | 0.41 | 0.89 | 0.33 | 0.58 | 0.27 | 0.50 | 0.87 | 1.16 | 0.88 |
| OT-CFM one panel, 4 steps | **0.35** | **0.81** | 0.31 | **0.40** | 0.28 | **0.34** | 0.82 | 1.13 | 0.88 |
| L1 regression | 0.43 | 0.86 | 0.41 | 0.63 | 0.34 | 0.56 | 0.85 | 1.19 | 0.90 |

What this says:

- **In distribution, averaging the sampler's 16 samples beats UVCGAN-S
  jet by jet on every observable:** energy 4% better, girth 7%, p_T^D 14%,
  z_lead 12%. n1, z_g and R_g are 11-21% better when the observable is
  averaged over the samples.
- **Two read-outs, each best for its own kind of observable:**
  - for energy and smooth shapes (girth, p_T^D, core, z_lead), compute the
    observable on the average image;
  - for counting and clustering observables (n1, z_g, R_g), compute it on
    each sample and average.

  A single sample is much worse than either, because it carries the full
  uncertainty.
- **The sampler gives per-jet error bars, and they carry real
  information.** The spread of the samples correlates with the actual
  error: 0.2-0.5 on val, and up to 0.7 on JEWEL for p_T^D and z_lead. But
  the error bars are too narrow: 44-65% of jets fall within +-1 sigma
  instead of 68%, so they are about 1.1-1.7x too small.
- **On JEWEL, the sampler's shapes are as good as OT-CFM's and better than
  UVCGAN-S's, but its jet energy is worse than OT-CFM's** (0.41 against
  0.35). That is the PYTHIA prior again.

### Step 2: the new model

`postflow` is a conditional flow-matching sampler with the physics built
in:

1. **It generates only the jet image; the background is mixture minus
   jet.** Every sample adds up to the measured energy in every tower, with
   0 <= jet <= mixture: no negative towers and no overshoot. The old
   sampler generated both images separately, and they did not add up.
2. **Randomised jet shapes in training (`--augment jets`), aimed at the
   JEWEL weakness.**
   - Half of the training jets are replaced by modified versions:
     - harder or softer fragmentation at the same energy;
     - up to 40% of each tower's energy spread to its neighbours;
     - tower-level fluctuations;
     - the energy scaled by up to +-25%.
   - The mixture is built from the modified jet, so the training pairs
     stay exact.
   - The ranges were set on PYTHIA training jets only. The modified jets
     keep PYTHIA's average shape, but their leading-tower share and p_T^D
     vary 1.5-1.7x more from event to event.
   - Nothing was tuned on JEWEL.
3. **Read-outs, as learned in step 1:**
   - the average image, for energy and smooth shapes;
   - the per-sample average, for n1, z_g and R_g;
   - the spread of the samples, as a per-jet error bar.

A cheap companion, **`regress_mse`**, is one network that predicts the
average split directly: each tower's jet share, trained with squared error
in GeV. In principle this is what averaging infinitely many samples
converges to, in 1 network pass (0.5 ms) instead of 128.

Runs on dahlia, started 17:35, each with 2 h of training and then
scoring:

| run | what it tells us |
| :--- | :--- |
| postflow with randomised jets, seeds 0, 1, 2 | the proposed model, and how stable it is across seeds |
| postflow without randomisation, seed 0 | what the randomisation buys on JEWEL and costs on val |
| `regress_mse` with randomised jets, 1 h | whether one pass reaches the sampler's energy resolution |

Written down before training (`FLOW_NOTES.md`, section "Step 2"):

- **H1:** generating the jet alone costs nothing: within +-0.03 GeV of
  conditional CFM at 2 h (3.62 GeV with 16 samples).
- **H2:** randomised jets improve JEWEL by >= 0.15 GeV, at a val cost of
  <= 0.05 GeV.
- **H3:** the one-pass `regress_mse` reaches <= 3.65 GeV on val.
- **Decision rule:** the new model is recommended over UVCGAN-S if, with
  16 samples, it reaches <= 3.62 GeV on val and <= 3.99 on JEWEL, and has
  better per-jet substructure on most shape observables on both sets.

**Outcome** (FLOW_NOTES.md, "Step 2 results"):
- **H1 holds:** generating only the jet costs nothing (3.61 against 3.62
  GeV at 2 h).
- **H2 fails:** the randomised jet shapes did not help on JEWEL
  (4.27-4.30 against 4.28 without them) and cost 0.03 GeV on val.
- **H3 holds, and it is the surprise:** the one-pass `regress_mse` network,
  which predicts the average split directly, reaches **3.56 GeV on val**
  after under an hour of training (3.58 after 5 minutes), at 0.5 ms per
  event. That is the best in-distribution jet energy and per-jet energy
  mover's distance of any model. On JEWEL it gets 4.12.
- **The new sampler is not recommended** over UVCGAN-S (val 3.64, JEWEL
  4.27-4.30).
- **Every model trained on PYTHIA jets stays at 4.1-4.3 GeV on JEWEL**,
  against 3.99 for UVCGAN-S and about 3.55 for the unpaired OT-CFM.

## Part 4: does a better network fix unpaired OT-CFM's blurry jets? (backbone ablation)

The research goal was restated: a simple, credible unpaired OT flow for
vacuum -> medium jets, with the decomposition as a benchmark. The open
issue is event-level spatial fidelity: the raw OT-CFM jet images carry a
faint noise floor, which makes jets look broader and softer than they are.

- **Test:** swap the flow's network for the UVCGAN-S generator, with the
  time added to its style input, and change nothing else.
- **Result:**
  - Energy resolution is the same or slightly better (val 3.66 against
    3.69-3.73 at 2 h; JEWEL 3.55 against 3.58-3.61).
  - Raw substructure errors fall 3-7%.
  - It trains 1.8x more updates per hour, uses half the memory, and runs
    1.8x faster.
  - The noise-floor biases (girth +0.18 sigma, p_T^D -0.35 sigma) are
    unchanged.
- **Conclusion:** the network is at most a minor part of the gap.

## Part 5: known-modification closure test of the matching cost

- **Question:** can an unpaired transport recover each jet's own
  modification, and does a better matching cost help?
- **Setup:**
  - PYTHIA jets, and the same jets modified by a known rule: energy x 0.8
    and a gentle broadening.
  - Training on different jets for source and target, so no pairs are
    seen; scoring each held-out jet against its true modified version.
- **Costs compared:** the existing jet-centred matching, against a new cost
  that compares unit-energy shapes at three scales plus a soft log-energy
  term.
- **Matching audit, before training:**
  - Full-image matching pairs jets by position.
  - The existing cost pairs by shape and ignores energy.
  - The new cost pairs jets almost exactly by energy.
- **Result:**
  - The new cost recovers each jet's energy change well. With an accurate
    solve the per-jet energy error is 1.3 GeV against 3.0, and the response
    is 0.81 ± 0.04 against the true 0.80.
  - p_T^D and z_lead are also better.
  - The per-jet shape does not improve.
  - **Neither cost keeps each jet's fine structure:** the output shapes are
    2.5-3x further from the true modified jet than the unmodified input
    jet is. The jets keep their coarse layout, but the tower-level detail is
    smeared into a diffuse halo, the same noise floor as in the
    decomposition.
  - The decomposition's 4-step solve is biased for jet -> jet flows; an
    accurate solve is needed.
- **Conclusion:**
  - The coupling change fixes per-jet energy correspondence, not per-jet
    shape.
  - This does not yet justify a PYTHIA -> JEWEL study of per-jet
    substructure.
  - The next step is making the transport keep a jet's own fine structure,
    tested in the same closure test.

## Part 6: where the closure test's fine structure is lost

A sequence of cheap diagnostics, then controls:

- **Pipeline checks, no training.** No bug.
  - The normalisation round trip is exact.
  - A zero velocity returns the input.
  - Clipping at 0 changes nothing that matters.
  - The probability path has no noise at either end, and inference starts
    where training does.
  - Correction to Part 5: there is no halo in empty towers. The true
    modified jets have only ~1 exactly empty tower each, and the flows put
    ~0 there. The error is a flattening of each jet's core: the hardest
    towers come out low, and that energy is spread thinly over soft towers.
- **Paired positive control:** the same flow, trained on the true pairs
  (J, T(J)).
  - It reproduces T(J) almost exactly: shape error 0.0002 against 0.032
    for doing nothing and 0.09 for the unpaired flows.
  - Energy response 0.7997 (truth 0.8); mass-change error 0.003 GeV.
  - It gets there within 10 minutes of training, equally on training and
    held-out jets.
  - So the flow-matching pipeline is fine.
- **Unpaired null test** (source and target both unmodified PYTHIA): the
  flow is nearly the identity. Its spurious changes are 4-16% of the toy
  modification. It doesn't deform jets on its own.
- **Larger matching pool** (1024 instead of 256 jets per domain, same batch
  of 256 pairs):
  - All fine-structure errors drop 15-25%: shape error 0.076 against 0.091.
  - The matching then takes 80% of each training step.
  - Still 2.4x worse than doing nothing on shape.
- **Conclusion:**
  - The unpaired coupling, not the flow-matching training or
    representation, loses each jet's fine structure.
  - Bigger minibatch matching helps only slowly.
  - The next justified experiment is semi-paired training: add a small
    fraction of true pairs to the unpaired data. For vacuum -> medium, JEWEL
    can simulate vacuum and medium versions of the same hard scattering to
    provide them.

## Where everything is

- `FLOW_NOTES.md`: the full log of the flow study (pre-registration,
  every result with its job numbers, commands).
- `SCALING_NOTES.md`: UVCGAN-S training speed and settings.
- `docs/flow/`:
  - tables: `compare*.csv`, `readout_*.csv`, `substructure*.csv`,
    `coupling*.csv`, `latency.csv`;
  - figures: `compare_report.png`, `compare_nfe.png`, `substructure.png`,
    `readout_events_val.png`;
  - each run's config and scores, under `docs/flow/runs/`.
- `scripts/flow/`: all code:
  - `fm_common.py`: the models;
  - `fm_train.py`, `fm_eval.py`, `fm_compare.py`: train, score, compare;
  - `readout_test.py`, `substructure.py`, `jet_fidelity.py`: read-outs and
    the jet-level benchmarks;
  - the coupling diagnostics.
- `outdir/sphenix/flow/<run>/`: checkpoints, training histories and
  per-event outputs (not in git).

## Still open

- A semi-paired closure test: how much event-level fidelity a small
  fraction of true pairs recovers in otherwise unpaired training. For
  vacuum -> medium, paired JEWEL vacuum/medium simulation would supply the
  pairs.
- Jet-level physics with the jets actually found in the extracted image.
  So far a jet is the cone at the true axis.
- Why OT-CFM does better on JEWEL than on val.
- Any unpaired correspondence depends on the chosen matching cost. That is
  a method dependence to study, which agreement of the output distributions
  cannot resolve.
