# Stochastic conditional FM pilot: several JEWEL-like outputs per PYTHIA jet

Noise-to-target conditional flow matching (Lipman et al. 2210.02747), with
the PYTHIA jet as an unchanged second input channel. The training pairs
(PYTHIA condition, JEWEL target) come from a minibatch optimal-transport plan
(Tong et al. 2302.00482): exact (the hard control) or entropic (the soft
treatment). Given a PYTHIA jet, each new Gaussian noise gives a new output.
It is an ODE sampler, not DSBM and not an SDE bridge. Its conditional
distribution is the one the chosen coupling defines. It is not a physical
medium response, and its spread is not a calibrated uncertainty.

Design, pre-registration, results and caveats: `FLOW_NOTES.md`, section
"Stochastic conditional FM pilot"; plain-language summary: `FLOW_SUMMARY.md`,
Part 13. Data, splits, normalisation and observables are those of the
translation pilot (`../README.md`), unchanged.

Result in brief (seed 0, 2 A6000 GPU-hours per arm, midpoint 128 NFE):
sharp single samples with JEWEL's substructure (cone pT and mass 2-3x
OT-FM's distance); diversity 2500x the solver error; but a jet's samples
spread over most of the JEWEL population, with output-input correlations
(E 0.19, girth 0.72) equal to those of the training pairs themselves (0.17,
0.71). The entropic plan changes nothing measurable. Pre-registered reading:
hard useful by the letter, soft not (mass 3.4 sd behind OT-FM).

## Files

| file | content |
| :--- | :--- |
| `coupling_audit.json`, `coupling_audit_grid.csv`, `coupling_audit_pairs.csv` | the training-only audit that fixed the entropic regularisation before training (reg 7: median row support 4.0 targets): support quantiles, row entropy, marginal residuals, Sinkhorn sweeps and time; the sampled pairs' energy and shape displacements, hard against soft against random pairs |
| `coupling_audit_corr.csv` | descriptive, after the choice: condition-target correlations of the audit pairs (E 0.17 hard, 0.16 soft, 0 random; girth 0.71, 0.70) |
| `checks.json`, `gauss_check.json` | pre-training checks: the path inside the loss, indexing after both plans, fixed-noise reproducibility; the Gaussian positive control y = c + 0.2 xi |
| `configs/*.json` | the two runs' configurations |
| `solver_check_{hard,soft}_ode128.csv` (`_ode256.csv` if needed), `solver_frozen_{hard,soft}.json` | validation: midpoint 128 against 256 NFE from the same noise, and two noises at 256; the frozen solve |
| `curves.csv`, `curves_refs.csv`, `tr_curves*.png` | validation curves every 10 min (midpoint 128; OT-FM's at 32) |
| `population.csv`, `joint.csv`, `refound.csv`, `dependence.csv`, `fidelity.csv`, `migration.csv`, `verdict.csv`, `tr_*.png` | the translation pilot's population benchmark (`translation_report.py`), one output per test input (sampler seed 0), next to OT-FM, alpha-DSBM, CycleGAN and the references |
| `cf_spread.csv`, `cf_spread_corr.csv` | 8 samples of the first 1000 test inputs: within-input sd against the sd across inputs and against the size of the change; correlations of the within-input fluctuations |
| `cf_distances.csv` | normalised-shape EMD: two samples of one input, sample to its input, to OT-FM's output, to an unrelated input; energy and tower rms differences of two samples |
| `cf_swap.csv` | the condition swap at fixed noise |
| `cf_tails.csv` | per-sample tails against JEWEL test's 5% / 95% quantiles (core fraction, z_lead, soft energy, occupancy) |
| `cf_seeds.csv` | the population W1 for a second sampler seed (sampling variability of the benchmark) |
| `cf_verdict.csv` | the pre-registered reading, item by item |
| `cf_pairs_info.csv` | the couplings' reference sets: targets per input used (64 of about 500) |
| `cf_samples_{hard,soft}.png`, `cf_samples_slide.png`, `cf_profiles.png`, `cf_spread.png` | every sample of five fixed inputs (no averaging), their radial profiles and leading/core towers; the spread and distance summary |
| `deck/` | pT spectra, substructure and shape-profile figures (`translation_deck_figs.py`) |
| `slides/condfm_appendix.{tex,pdf}`, `slides/tables/cf_*.tex` | the four-slide appendix |

Runs, checkpoints and outputs (not in git):
`OUTDIR/sphenix/flow/translation/condfm/{runs,outputs}`; the existing
outputs used for comparison are linked there, not copied or regenerated.

## Commands

From the repository root after `. ./scripts/flow/env.sh`:

    # before training: the coupling audit (fixes reg), the checks, the Gaussian control
    sbatch -p a6k -w dahlia --gres=gpu:1 -c 6 --wrap \
        '. ./scripts/flow/env.sh && $PYTHON -u scripts/flow/condfm_checks.py --audit --checks --gauss'
    # the two runs: seed 0, 2 A6000 GPU hours each, then their validation curves
    KIND=condjet COUPLING=exact LABEL=tr_condfm_hard_s0 MINUTES=120 SEED=0 \
        sbatch -w dahlia -J tr_condfm_hard_s0 scripts/flow/translation_run.sbatch
    KIND=condjet COUPLING=entropic OT_REG=7 LABEL=tr_condfm_soft_s0 MINUTES=120 SEED=0 \
        sbatch -w dahlia -J tr_condfm_soft_s0 scripts/flow/translation_run.sbatch
    # solver check on validation, the frozen solve, test outputs (8 samples, swap, repeat)
    RUN=tr_condfm_hard_s0 LABEL=CondFM-hard ARM=hard sbatch -w dahlia scripts/flow/condfm_post.sbatch
    RUN=tr_condfm_soft_s0 LABEL=CondFM-soft ARM=soft sbatch -w dahlia scripts/flow/condfm_post.sbatch
    # a second sampler seed of the single outputs (sampling variability of the benchmark)
    $PYTHON scripts/flow/translation_eval.py --generate OUTDIR/.../condfm/runs/tr_condfm_hard_s0 \
        --labels CondFM-hard-s1 --seed 1 --ode-setting 128:midpoint --primary-only   # same for soft
    # descriptive, after the choice of reg: condition-target correlations of the
    # audit pairs, and each plan's targets for the 1000 repeated-sample inputs (GPU)
    $PYTHON scripts/flow/condfm_checks.py --pair-corr
    $PYTHON scripts/flow/condfm_checks.py --partners
    # every metric, figure and table (CPU), then the appendix
    sbatch -p a6k -w saturn -c 32 --mem=64G scripts/flow/condfm_report.sh
    cd docs/flow/translation/condfm/slides && ~/pyext/tectonic_env/bin/tectonic condfm_appendix.tex
