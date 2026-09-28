# Consolidated subtraction benchmark

Background-only against joint flow matching for M = S + B, against the
published UVCGAN-S and the Area method. Design, results and caveats:
`FLOW_NOTES.md`, section "Consolidated benchmark"; plain-language summary:
`FLOW_SUMMARY.md`, Part 9. Deck: `slides/bench_deck.pdf` (source
`slides/bench_deck.tex`, `slides/body.tex`; generated tables
`slides/tables/*.tex`).

## Files

| file | content |
| :--- | :--- |
| `bench_runs.csv`, `configs/*.json` | every flow run: method, seed, selected checkpoint (step, minutes, path under OUTDIR; checkpoints are not in git), commit, GPU; the run configurations |
| `bench_jets.csv` | per run (seed) and jet radius: response, resolution, bias / RMSE in GeV, efficiency, fake rate in pT ranges, the frozen-calibration response, and at R = 0.4 the substructure scores (per-jet bias / RMSE, W1, means), each with its statistical error (`_se`); `set = jewel-val` rows: the JEWEL - PYTHIA shift kept (`_mod_fraction`) |
| `bench_summary.csv` | the same pooled over seeds: `mean`, `half_range`, `stat_se` |
| `bench_towers.csv`, `bench_towers_summary.csv` | per-tower MAE / RMSE, event energy bias, background error by true signal energy of the tower |
| `bench_consistency.csv` | joint arm: B_hat + S_hat - M before clean-up |
| `bench_cone*.csv` | supplementary true-axis cone scores (jet_fidelity.py) |
| `bench_cost.csv`, `bench_cost_runs.csv` | training time, val resolution against hours, time to 3.70 GeV, throughput, peak memory, latency |
| `bench_loss_t.csv` | the CFM loss of the paired and joint arms at fixed t on held-out pairs (where the training signal lies) |
| `bench_fig3_*.png` ... `bench_fig7.png` | the paper's Figs. 3-7 (val and JEWEL), each readout at its val-preferred solve; `mid32/`: the same with the accurate solve for every flow |
| `bench_fig5_allR_*.png` | efficiency and fake rate at R = 0.2, 0.4, 0.5 |
| `bench_jets_R04_*.png` | scale, resolution, efficiency, fake rate at R = 0.4 (deck) |
| `bench_displays*.png`, `bench_bkgerr.png` | event displays; background error by tower |
| `bench_cost.png` | val resolution against training hours; inference latency |
| `bench_cone_s0.png`, `bench_cone_s0*.csv` | supplementary cone scores, seed 0 of every arm, both solves |

Labels: `[mid32]` = the accurate solve (midpoint, 32 network evaluations);
no suffix = 4 Euler steps; `[thr0.5]` = the labelled clean-up (towers below
0.5 GeV of the output dropped). Main figures show each readout at its
val-preferred solve (4 Euler steps; the joint arm's direct S_hat: 32 NFE).
The calibrated (`_cal`) numbers use each run's calibration fitted on val
events 10000-19999 and frozen, on JEWEL too.

## Commands

All from the repository root after `. ./scripts/flow/env.sh`.

    # training pairs of the real mixtures (once; OUTDIR/sphenix/flow/cache)
    $PYTHON scripts/flow/make_pairs.py

    # the runs: 2 h on one A6000 each, then val scores of every checkpoint
    sub() { METHOD=$1 LABEL=$2 MINUTES=120 SEED=$3 EVAL_DECODE=mixture \
        EVAL_ARGS="--nfe 4 --solver euler" sbatch -w dahlia --time=03:30:00 \
        -J $2 scripts/flow/fm_run.sbatch --ckpt-minutes 15 --inline-events 1000 \
        --inline-nfe 4 --backbone uvcgan; }
    sub otcfm1 bb_uvcgan_otcfm1_s1 1; sub otcfm1 bb_uvcgan_otcfm1_s2 2
    for s in 0 1 2; do sub otcfm1_paired bench_paired_bkg_s$s $s; done
    for s in 0 1 2; do sub joint_paired bench_joint_s$s $s; done

    # per run: cone scores of the selected checkpoint (val, JEWEL; 4 Euler,
    # 32 NFE midpoint, 64 on val for seed 0) and the benchmark images
    LABEL=bench_joint_s0 JOINT=1 CONVERGENCE=1 sbatch -w dahlia scripts/flow/bench_post.sbatch
    $PYTHON scripts/flow/bench_images.py --uvcgan          # UVCGAN-S images

    # latency, one A6000
    $PYTHON scripts/flow/fm_eval.py --latency RUN_DIRS --nfe 4,32 --solver euler,midpoint

    # jets, figures, tables, diagnostics, cone scores, cost, loss against t
    # (CPU node; STAGES="jets report diag cone cost losst" by default)
    sbatch -p a6k -w saturn -c 64 --mem=160G scripts/flow/bench_all.sh

    # deck tables and the deck
    $PYTHON scripts/flow/bench_tables.py
    cd docs/flow/bench/slides && ~/pyext/tectonic_env/bin/tectonic bench_deck.tex
