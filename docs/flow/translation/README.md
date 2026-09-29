# Unpaired PYTHIA -> JEWEL jet translation pilot

Can OT-CFM and online alpha-DSBM, trained unpaired on clean jets, map a
PYTHIA jet image to an output with held-out JEWEL statistics while keeping a
meaningful dependence on the input? Design, pre-registration, results and
caveats: `FLOW_NOTES.md`, section "PYTHIA -> JEWEL translation pilot";
plain-language summary: `FLOW_SUMMARY.md`, Part 12. Appendix:
`slides/translation_appendix.pdf` (source `slides/translation_appendix.tex`,
`slides/body.tex`; generated tables `slides/tables/*.tex`).

Nothing here is a per-event medium modification: there is no matched JEWEL
jet for a PYTHIA jet, and any unpaired map is one choice among many with the
same marginals.

## Files

| file | content |
| :--- | :--- |
| `manifest.json` | the data set: parent counts, duplicates, selection, crop and what it drops, splits, normalisation, checks, known and unknown provenance |
| `splits/pythia.csv.gz`, `splits/jewel.csv.gz` | every used parent: generator file, event, split (train / val / test / ref), jet axis (row, col), cone energy |
| `configs/*.json` | the runs' configurations (copied from `OUTDIR/sphenix/flow/translation/runs/`) |
| `solver_check.csv`, `solver_check_ode64.csv`, `solver_check_ode128.csv` | validation: OT-CFM midpoint 32 vs 64, 64 vs 128 and 128 vs 256 NFE (and 4 Euler), alpha-DSBM 30 vs 60 SDE steps with coupled noise against two independent samples, with bootstrap errors; frozen: OT-CFM 128 NFE, alpha-DSBM 30 steps |
| `curves.csv`, `curves_refs.csv`, `tr_curves.png` | validation curves every 10 min (OT-CFM at 32 NFE and 4 Euler, alpha-DSBM at 30 SDE steps; population W1 / sigma against JEWEL val, input correlations) and the identity / JEWEL-train reference levels |
| `population.csv`, `tr_marginals.png` | test: W1 / sigma_JEWEL of the fixed-crop E, mass, girth, p_T^D, z_lead, z_g, R_g after the common E >= 10 GeV selection, bootstrap sd, acceptance, the pre-registered flags |
| `joint.csv`, `tr_joint.png` | girth and z_lead against the cone energy: W1 in energy bins, binned means, correlation-matrix difference |
| `refound.csv`, `tr_refound.png` | the leading anti-kT R = 0.4 jet refound in each canvas (FastJet), with pT >= 10 GeV on every sample |
| `dependence.csv`, `tr_changes.png` | input-output correlations (bootstrap sd), the distributions of the changes and the changes against input energy, the own-input preference rate |
| `migration.csv`, `tr_migration.png` | output energy by input energy bin, share passing the final selection |
| `fidelity.csv`, `tr_towers.png` | occupancy, soft-tower energy, leading tower, core fraction, the artefact flags; cone tower spectra |
| `tr_displays.png`, `tr_samples.png` | fixed held-out inputs (first 1000 test jets at 5 energy quantiles) with their outputs; 8 alpha-DSBM samples each |
| `variability.csv` | alpha-DSBM: spread of 8 samples per input (first 1000 test inputs) |
| `verdict.csv` | the pre-registered reading, item by item |
| `tr_profiles.png`, `tr_changes_slide.png`, `tr_displays_slide.png`, `tr_curves_slide.png` | compact versions of the figures for the appendix |
| `cost.csv`, `outputs.json` | GPU hours, updates and rates by stage, peak memory, NFE and latency; the generated outputs' settings |

Samples: `JEWEL test` (target), `JEWEL ref` (a second held-out JEWEL
sample: the finite-sample floor), `identity` (the PYTHIA test inputs),
`random JEWEL` (training-pool jets, one per input, ignoring it), and the
models. `OT-CFM (4 Euler)` is a secondary speed readout.

## Commands

From the repository root after `. ./scripts/flow/env.sh`:

    # the data set (one GPU; JEWEL ROOT files read once, ~5 min in all)
    sbatch -w saturn scripts/flow/translation_data.sbatch

    # the two models, 2 A6000 GPU hours each, then their validation curves
    KIND=otcfm LABEL=tr_otcfm_s0 MINUTES=120 SEED=0 \
        sbatch -w dahlia -J tr_otcfm_s0 scripts/flow/translation_run.sbatch
    KIND=dsbm LABEL=tr_dsbm_e025_s0 PRE_MINUTES=30 MINUTES=90 EPS=0.25 SEED=0 \
        sbatch -w dahlia -J tr_dsbm_e025_s0 scripts/flow/translation_run.sbatch

    # solver checks on validation (32 vs 64; then, as 32 failed, 64 vs 128 and
    # 128 vs 256 for OT-CFM), then the test outputs at the frozen settings
    MODE=check sbatch -w dahlia scripts/flow/translation_post.sbatch
    MODE=check ODE=64:midpoint RUNS=tr_otcfm_s0 LABELS=OT-CFM \
        CHECK_FILE=solver_check_ode64.csv sbatch -w dahlia scripts/flow/translation_post.sbatch
    MODE=check ODE=128:midpoint RUNS=tr_otcfm_s0 LABELS=OT-CFM \
        CHECK_FILE=solver_check_ode128.csv sbatch -w dahlia scripts/flow/translation_post.sbatch
    MODE=generate ODE=128:midpoint STEPS=30 sbatch -w dahlia scripts/flow/translation_post.sbatch

    # every metric, figure and table (CPU), then the appendix
    ODE=midpoint128 STEPS=30 sbatch -p a6k -w saturn -c 32 --mem=64G scripts/flow/translation_report.sh
    cd docs/flow/translation/slides && ~/pyext/tectonic_env/bin/tectonic translation_appendix.tex

Caches, runs and outputs (not in git) are under
`OUTDIR/sphenix/flow/translation/{cache,runs,outputs}`.
