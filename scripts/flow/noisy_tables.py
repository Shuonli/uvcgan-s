#!/usr/bin/env python
"""Comparison tables of the paired noisy-interpolant pilot (FLOW_NOTES.md,
"Paired noisy-interpolant pilot"): control (eta 0) against treatment (eta
0.1) at the final update, per solve, from the benchmark outputs of
noisy_bench.sh (docs/flow/bench/noisy/bench_summary.csv,
bench_towers_summary.csv), with the pre-registered threshold of every
decision metric: 3x the larger of the benchmark paired arm's seed half
range (docs/flow/bench, same solve) and the statistical error (the W1
floor for distributions). Writes docs/flow/bench/noisy/noisy_compare.csv
and the deck tables docs/flow/bench/slides/tables/noisy_*.tex.

    noisy_tables.py [--dir docs/flow/bench/noisy]
"""

import argparse
import os

import numpy as np
import pandas as pd

ARMS = { 'control' : 'control (eta 0)', 'noisy' : 'noisy (eta 0.1)' }
BENCH = 'paired FM'
SOLVES = { 'mid32' : ' [mid32]', 'euler4' : '' }
OBS = [ 'girth', 'mass', 'zlead', 'ptd', 'zg', 'rg' ]
OBS_TEX = { 'girth' : '$g$', 'mass' : '$m$', 'zlead' : '$z_{lead}$', 'ptd' : '$p_T^D$',
            'zg' : '$z_g$', 'rg' : '$r_g$' }
# (metric, source, better, TeX); better: -1 lower is better, +1 higher,
# 0 = closer to the truth value in `target`
JETS = [ ('scale_20_30', 'jets', 0, 'jet scale 20-30 GeV'),
         ('resolution_cal_20_30', 'jets', -1, 'resolution (frozen cal.)'),
         ('eff_14_20', 'jets', +1, 'efficiency 14-20 GeV'),
         ('fake_14_20', 'jets', -1, 'fake rate 14-20 GeV'),
         ('rmse_gev_20_30', 'jets', -1, 'jet $p_T$ RMSE, GeV') ]
TOWERS = [ ('bkgerr_share[5-10]', -1, r'$\Bh$ share of $S$, 5-10 GeV'),
           ('bkgerr_share[>10]', -1, r'$\Bh$ share of $S$, $>$ 10 GeV'),
           ('bkgerr_share[2-5]', -1, r'$\Bh$ share of $S$, 2-5 GeV'),
           ('bkgerr_mean[S = 0]', 0, r'$\Bh - B$ where $S = 0$, GeV'),
           ('event_bias_gev', 0, r'$\sum(\Sh - S)$, GeV / event'),
           ('mae', -1, 'MAE / tower, GeV') ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Noisy-interpolant pilot tables')
    parser.add_argument('--dir', default = 'docs/flow/bench/noisy')
    parser.add_argument('--bench', default = 'docs/flow/bench')
    parser.add_argument('--tables', default = 'docs/flow/bench/slides/tables')
    parser.add_argument('--control-label', default = ARMS['control'])
    parser.add_argument('--noisy-label', default = ARMS['noisy'])
    return parser.parse_args()

def jets_value(summary, label, metric):
    q = summary[(summary.set == 'val') & (summary.label == label) & (summary.R == 0.4)
                & (summary.metric == metric)]
    if not len(q):
        return (np.nan, np.nan, np.nan)
    r = q.iloc[0]
    return (float(r['mean']), float(r['half_range']), float(r['stat_se']))

def towers_value(towers, label, metric):
    q = towers[(towers.set == 'val') & (towers.label == label)]
    if not len(q) or metric not in q:
        return (np.nan, np.nan)
    r = q.iloc[0]
    return (float(r[metric]), float(r.get(f'{metric}_half_range', np.nan)))

def verdict(diff, thr, better, ref_c = None, ref_n = None):
    """'better', 'worse' or 'same' for treatment - control."""
    if not np.isfinite(diff) or not np.isfinite(thr):
        return 'n/a'
    if abs(diff) <= thr:
        return 'same'
    if better == 0:
        # closer to the truth value (0 for residuals, 1 for... handled by caller)
        return 'better' if abs(ref_n) < abs(ref_c) else 'worse'
    return 'better' if diff * better > 0 else 'worse'

RUNS = [ 'bench_paired_eta0_s0', 'bench_paired_eta0p1_s0' ]

def manifest(out_dir):
    """noisy_runs.csv and configs/<run>.json: the pilot's runs at their final
    update (checkpoints stay under OUTDIR, not in git)."""
    # pylint: disable=import-outside-toplevel
    import json
    import fm_common as fc
    os.makedirs(os.path.join(out_dir, 'configs'), exist_ok = True)
    rows = []
    for run in RUNS:
        run_dir = os.path.join(fc.out_root(), run)
        with open(os.path.join(run_dir, 'config.json'), encoding = 'utf-8') as f:
            config = json.load(f)
        with open(os.path.join(run_dir, 'summary.json'), encoding = 'utf-8') as f:
            summary = json.load(f)
        with open(os.path.join(out_dir, 'configs', f'{run}.json'), 'w', encoding = 'utf-8') as f:
            json.dump(config, f, indent = 4)
        last = config['runs'][-1]
        rows.append({ 'run' : run, 'method' : config['method'], 'seed' : config['seed'],
                      'backbone' : config['backbone'], 'path' : config['path'],
                      'eta' : config['eta'], 'step' : summary['step'],
                      'train_minutes' : summary['train_time'] / 60,
                      'end_to_end_minutes' : summary['end_to_end_time'] / 60,
                      'updates_per_s' : summary['steps_per_s'],
                      'peak_mem_gb' : summary['peak_mem_gb'],
                      'checkpoint' : f"OUTDIR/sphenix/flow/{run}/checkpoints/step_{summary['step']:08d}.pt",
                      'network' : 'ema', 'n_params' : summary['n_params'],
                      'commit' : last['commit'], 'gpu' : last['gpu'], 'node' : last['node'] })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'noisy_runs.csv'), index = False)

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    ARMS.update(control = cmdargs.control_label, noisy = cmdargs.noisy_label)
    summ  = pd.read_csv(os.path.join(cmdargs.dir, 'bench_summary.csv'))
    tow   = pd.read_csv(os.path.join(cmdargs.dir, 'bench_towers_summary.csv'))
    bsum  = pd.read_csv(os.path.join(cmdargs.bench, 'bench_summary.csv'))
    btow  = pd.read_csv(os.path.join(cmdargs.bench, 'bench_towers_summary.csv'))
    rows = []
    for (solve, sfx) in SOLVES.items():
        (lc, ln, lb) = (ARMS['control'] + sfx, ARMS['noisy'] + sfx, BENCH + sfx)
        for (metric, _, better, tex) in JETS:
            (c, _, se_c) = jets_value(summ, lc, metric)
            (n, _, se_n) = jets_value(summ, ln, metric)
            (_, hr, _) = jets_value(bsum, lb, metric)
            thr = 3 * np.nanmax([ hr, se_c, se_n ])
            if metric == 'scale_20_30':
                # the true scale is 1: closer to 1 is better
                v = verdict(n - c, thr, 0, c - 1, n - 1)
            else:
                v = verdict(n - c, thr, better)
            rows.append({ 'solve' : solve, 'metric' : metric, 'tex' : tex,
                          'control' : c, 'noisy' : n, 'diff' : n - c,
                          'threshold' : thr, 'verdict' : v })
        for (metric, better, tex) in TOWERS:
            (c, _) = towers_value(tow, lc, metric)
            (n, _) = towers_value(tow, ln, metric)
            (_, hr) = towers_value(btow, lb, metric)
            thr = 3 * hr if np.isfinite(hr) else np.nan
            v = verdict(n - c, thr, better, c, n) if better == 0 else \
                verdict(n - c, thr, better)
            rows.append({ 'solve' : solve, 'metric' : metric, 'tex' : tex,
                          'control' : c, 'noisy' : n, 'diff' : n - c,
                          'threshold' : thr, 'verdict' : v })
        for obs in OBS:
            (sd, _, _) = jets_value(summ, lc, f'{obs}_truth_all_sd')
            for (kind, tex_kind) in [ ('pair_rmse', 'per-jet RMSE'), ('w1', 'W1'),
                                      ('pair_bias', 'per-jet bias') ]:
                metric = f'{obs}_{kind}'
                (c, _, se_c) = jets_value(summ, lc, metric)
                (n, _, se_n) = jets_value(summ, ln, metric)
                (_, hr, _) = jets_value(bsum, lb, metric)
                if kind == 'w1':
                    (floor, _, _) = jets_value(summ, lc, f'{obs}_w1_floor')
                    thr = 3 * np.nanmax([ hr, floor ])
                else:
                    thr = 3 * np.nanmax([ hr, se_c, se_n ])
                if kind == 'pair_bias':
                    v = verdict(n - c, thr, 0, c, n)
                else:
                    v = verdict(n - c, thr, -1)
                rows.append({ 'solve' : solve, 'metric' : metric,
                              'tex' : f'{OBS_TEX[obs]} {tex_kind}', 'control' : c,
                              'noisy' : n, 'diff' : n - c, 'threshold' : thr,
                              'verdict' : v, 'truth_sd' : sd })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cmdargs.dir, 'noisy_compare.csv'), index = False)
    manifest(cmdargs.dir)
    with pd.option_context('display.width', 250, 'display.max_rows', 200):
        print(df.drop(columns = [ 'tex' ]).round(4).to_string(index = False))

    # deck tables: jets + towers; substructure (per-jet RMSE / sigma, W1 / sigma)
    os.makedirs(cmdargs.tables, exist_ok = True)
    def cell(solve, metric, digits, scale = 1.0):
        q = df[(df.solve == solve) & (df.metric == metric)].iloc[0]
        mark = { 'better' : r'$\uparrow$', 'worse' : r'$\downarrow$' }.get(q.verdict, '')
        return (f'{q.control * scale:.{digits}f}', f'{q.noisy * scale:.{digits}f}{mark}')
    lines = [ r'\scriptsize', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lrrrr}', r'\toprule',
              r' & \multicolumn{2}{c}{32 NFE (accurate)} & \multicolumn{2}{c}{4 Euler} \\',
              r' & $\eta = 0$ & $\eta = 0.1$ & $\eta = 0$ & $\eta = 0.1$ \\', r'\midrule' ]
    for (metric, digits, scale, tex) in [
            ('bkgerr_share[>10]', 1, 100, r'\% of $S$ in $\Bh$: $S > 10$ GeV'),
            ('bkgerr_share[5-10]', 1, 100, r'\hspace{1em}$S$ 5-10 GeV'),
            ('bkgerr_share[2-5]', 1, 100, r'\hspace{1em}$S$ 2-5 GeV'),
            ('scale_20_30', 3, 1, r'jet scale 20-30'),
            ('resolution_cal_20_30', 3, 1, r'resolution (frozen cal.)'),
            ('eff_14_20', 3, 1, r'efficiency 14-20'),
            ('fake_14_20', 3, 1, r'fake rate 14-20'),
            ('bkgerr_mean[S = 0]', 3, 1, r'$\Bh - B$ at $S = 0$, GeV'),
            ('event_bias_gev', 1, 1, r'$\sum(\Sh - S)$ / event, GeV'),
            ('mae', 3, 1, r'MAE / tower, GeV') ]:
        (a, b) = cell('mid32', metric, digits, scale)
        (c, d) = cell('euler4', metric, digits, scale)
        lines.append(f'{tex} & {a} & {b} & {c} & {d} \\\\')
    lines += [ r'\bottomrule', r'\end{tabular}' ]
    with open(os.path.join(cmdargs.tables, 'noisy_main.tex'), 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    lines = [ r'\scriptsize', r'\begin{tabular}{lrrrrrr}', r'\toprule',
              ' & ' + ' & '.join(OBS_TEX[o] for o in OBS) + r' \\', r'\midrule' ]
    for (kind, name) in [ ('pair_rmse', r'per-jet RMSE$/\sigma$'), ('w1', r'W1$/\sigma$') ]:
        for (solve, sname) in [ ('mid32', '32 NFE'), ('euler4', '4 Euler') ]:
            for (arm, aname) in [ ('control', r'$\eta = 0$'), ('noisy', r'$\eta = 0.1$') ]:
                cells = []
                for obs in OBS:
                    q = df[(df.solve == solve) & (df.metric == f'{obs}_{kind}')].iloc[0]
                    v = q[arm] / q.truth_sd
                    mark = ''
                    if arm == 'noisy':
                        mark = { 'better' : r'$\uparrow$', 'worse' : r'$\downarrow$' }.get(
                            q.verdict, '')
                    cells.append(f'{v:.3f}{mark}')
                lines.append(f'{name}, {sname}, {aname} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule' if kind == 'pair_rmse' else '')
    lines += [ r'\bottomrule', r'\end{tabular}' ]
    with open(os.path.join(cmdargs.tables, 'noisy_sub.tex'), 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(l for l in lines if l) + '\n')
    print('wrote', os.path.join(cmdargs.tables, 'noisy_main.tex'), 'and noisy_sub.tex')

if __name__ == '__main__':
    main()
