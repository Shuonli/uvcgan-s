# Jamie Nagle's toy exercises with OT flow matching

Jamie Nagle's four toy exercises (`nagle_cyclegan_explained.pdf`, not in git)
repeated with this repository's OT flow matching: the UVCGAN-S ViT-ModNet
velocity network, deterministic OT-CFM (D) and noise-driven conditional FM on
an OT pseudo-joint (C). Design, pre-registered rules, results and
limitations: `FLOW_NOTES.md`, section "Jamie's toy exercises with OT flow
matching"; short version `FLOW_SUMMARY.md`, Part 14. Deck:
`slides/jamie_otfm.pdf` (source `slides/jamie_otfm.tex`).

The toy is a reimplementation from the deck's code listings
(`scripts/flow/toycalo.py`), not Jamie's toycalo.py, which was not
available. Every result is labelled by its supervision:

- **pure unpaired**: independent source and target pools; the minibatch OT
  plan pairs them;
- **synthetic paired**: mixtures M = J + B built from the jet prior and UE
  pools, trained to their own B;
- **hybrid**: both of the above, L = L_synthetic + lambda_U L_unpaired;
- **true-paired control**: C trained on a vacuum jet and its own true
  quenchings (a positive control, not an unpaired result).

## Files

| path | content |
| :--- | :--- |
| `generator_checks.json` | the toy against the deck's numbers, acceptance losses, the protected-region audit, girth recoverability |
| `matching_audit.csv`, `entropic_reg.json` | what the minibatch OT plan matches (energy, axis, shape, UE), and the soft plan's temperature |
| `solver/LABEL.csv`, `solver/LABEL_frozen.json` | step doubling on validation, and the frozen midpoint NFE of each run |
| `solver/euler4.csv` | four Euler steps against the frozen solve (speed diagnostic only) |
| `solver/E3-D_s0_fine.csv` | E3-D's step doubling continued to 512 and 1024 NFE |
| `solver/LABEL_vnorm.csv` | the velocity along a 1024-NFE solve and how trajectories started 1e-3 apart separate (all, occupied, empty towers perturbed) |
| `weights/LABEL.json` | each continuation's extra-term weights (gradient-scale audit) and rollout NFE |
| `weights/LABEL_transfer.json` | a solved-endpoint continuation's own term at its training rollout's NFE and at its test solve |
| `ex1/` | Exercise 1: `ex1_subtraction.csv` (offset, resolution, halo, core loss, negatives, Jamie's ring table, energy slices), `ex1_slices.csv`, `ex1_refound.csv` (anti-kT), `ex1_fakes.csv` (UE-only events), `ex1_onestep.csv` (the one-step mean against the solved output), displays and ring figure |
| `ex2/` | Exercise 2: `ex2_shifts.csv` (quenched - vacuum shifts, recovered fractions, lost jets, recoil in B_hat, hard-core loss), `ex2_errors.csv` (error distributions) |
| `prior_data/` | the 2 x 2 prior/data test, the same files as `ex2/`, and `prior_data_effects.csv` (the pre-registered readings: data effect in combined sd, ring invented in vacuum jets) |
| `ex3/`, `ex3g/`, `ex4/` | translation: `*_population.csv` (W1/sigma to the true quenched jets), `*_dependence.csv` (shift recovered, input correlations; Exercise 4 also the far-region tower tests and the oracle), `*_emd.csv` (own-input shape EMD), `*_conditional.csv` (the bank: conditional W1, coverage, widths), `*_swap.csv` (condition swap, C), `*_seeds.csv` (seed 0 against seed 1, same inputs), `ex3g_girth_rule.csv`; figures |
| `ledger.csv` | every run: supervision, base, updates, GPU hours, solver |
| `coverage.csv` | Jamie's changes, our counterpart, run, result or reason for omission |
| `slides/` | the deck and its tables (`slides/tables/*.tex`, from `jamie_slides_tables.py`) |

Large files stay outside git, under `OUTDIR/sphenix/flow/jamie/`: `cache/`
(the pools and test sets, 5.1 GB, `manifest.json` with pool ids and seeds,
`norm.json`), `runs/LABEL/` (checkpoints, history, curves) and
`outputs/LABEL/` (solved test outputs).

## Commands

From the repository root, after `. ./scripts/flow/env.sh`; GPU jobs on
dahlia (A6000).

    # data, checks and the matching audit (CPU)
    $PYTHON scripts/flow/jamie_data.py --build --procs 30   # pools, sets, banks, norm.json
    $PYTHON scripts/flow/jamie_data.py --checks             # generator_checks.json
    $PYTHON scripts/flow/jamie_audit.py                 # matching_audit.csv, entropic_reg.json

    # a baseline (90 min of training), then its solver check and test outputs
    LABEL=E3-D_s0 MINUTES=90 SEED=0 sbatch -w dahlia -J E3-D_s0 scripts/flow/jamie_run.sbatch \
        --method toyflow --toy-kind clean --toy-pairing unpaired \
        --toy-source e3_vac --toy-target e3_med
    LABEL=E3-D_s0 sbatch -w dahlia scripts/flow/jamie_post.sbatch

    # a continuation (4000 updates from the base; weights audited first),
    # then its outputs at the base's frozen solve
    LABEL=E3-D_energy BASE=E3-D_s0 TERMS=energy SEED=0 sbatch -w dahlia \
        scripts/flow/jamie_cont.sbatch --method toyflow --toy-kind clean \
        --toy-pairing unpaired --toy-source e3_vac --toy-target e3_med
    LABEL=E3-D_energy BASE=E3-D_s0 sbatch -w dahlia scripts/flow/jamie_post.sbatch
    # a continuation whose check at the base's NFE failed: the doubling rule
    # again from twice that NFE (SETS=t2_pair for the 2 x 2 cells)
    sbatch -p a6k -w dahlia --gres=gpu:1 scripts/flow/jamie_repost.sh E4-D_global=128

    # diagnostics (GPU, or CPU when none is free)
    $PYTHON scripts/flow/jamie_eval.py RUN --euler4 --label L     # four Euler steps
    $PYTHON scripts/flow/jamie_eval.py RUN --vnorm --label L      # velocity, trajectory separation
    $PYTHON scripts/flow/jamie_eval.py RUN --onestep --label L    # one-step mean against the solve
    $PYTHON scripts/flow/jamie_eval.py RUN --check --nfe 512 --label E3-D_s0_fine
    $PYTHON scripts/flow/jamie_weights.py --transfer RUN --label L

    # tables, figures and the ledger of every run with outputs (CPU), the
    # deck's tables, the deck
    sbatch -p a6k -w saturn -c 32 --mem=160G scripts/flow/jamie_all_report.sh
    $PYTHON scripts/flow/jamie_slides_tables.py
    cd docs/flow/jamie_otfm/slides && ~/pyext/tectonic_env/bin/tectonic jamie_otfm.tex

The run arguments (all with `--batch 256`, exact minibatch OT unless
stated; C is `--method toycond`, D `--method toyflow`):

| run | arguments |
| :--- | :--- |
| E1-U | `--toy-kind sub --toy-pairing unpaired --toy-source e1_mix --toy-target e1_ue` |
| E1-P | `--toy-kind sub --toy-pairing synthetic --toy-prior e1_jet --toy-ue e1_ue` (flat prior: `--toy-prior e1_jet_flat`) |
| E3-D, E3-Ch | `--toy-kind clean --toy-pairing unpaired --toy-source e3_vac --toy-target e3_med` |
| E3-Cs | the same with `--coupling entropic --ot-reg 89` |
| E3-Cp | `--toy-kind clean --toy-pairing paired --toy-source e3_pair` (true-paired control) |
| E3g-D, E3g-C | `--toy-kind clean --toy-pairing unpaired --toy-source e3_vac --toy-target e3g_med` |
| E4-D, E4-C | `--toy-kind ue --toy-pairing unpaired --toy-source e4_src --toy-target e4_tgt` (E4-Dnf, E4-Cnf: `--toy-cost nearfar`) |
| 2 x 2 hybrid | `--toy-kind sub --toy-pairing hybrid --toy-ue e1_ue --toy-prior e3_vac or x_broad --toy-data e4_src or e4_tgt` (lambda_U = 1) |
| 2 x 2 synthetic only | `--toy-kind sub --toy-pairing synthetic --toy-ue e1_ue --toy-prior e3_vac or x_broad` |

Continuation terms (`TERMS`): `abs`, `bal`, `ring` (Exercise 1, local
endpoint surrogate at t <= 0.25), `energy` (total GeV of a solved endpoint),
`profile`, `div` (C, solved), `inv_global`, `inv_far` (Exercise 4, solved),
`ueprof` (the 2 x 2's batch-mean UE profile, on the surrogate); empty = the
unchanged-loss control. `ROLL_AT_BASE=1` rolls out at the base's frozen
test NFE instead of the audit's cheaper K (used for E4-Cnf_far only).

Runs added after the plan was fixed (FLOW_NOTES.md says why): E4-Cnf_s0 and
E4-Cnf_far (C on the near/far cost), and E4-Cnf_far's accurate rollout.
