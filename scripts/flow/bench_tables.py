#!/usr/bin/env python
"""LaTeX tables of the benchmark deck (docs/flow/bench/slides/tables/*.tex)
from the CSVs of bench_report.py, bench_diag.py and bench_cost.py and the
runs' cone scores (evals/{val,jewel}_truth.csv at the selected checkpoint).
Seed-pooled cells are the mean of the three seeds; each table states the
largest seed half range and a typical statistical error of its columns.

    bench_tables.py [--bench docs/flow/bench]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import fm_common as fc
from fm_eval import best_step

# the slides' rows: every readout at its val-preferred solve (4 Euler steps;
# the joint arm's direct S: 32 NFE); the backup table has both solves
ARMS = [ 'UVCGAN-S', 'Area', 'unpaired FM', 'paired FM', 'joint FM: M - B',
         'joint FM: direct S [mid32]' ]
BASE = [ 'unpaired FM', 'paired FM', 'joint FM: M - B', 'joint FM: direct S' ]
TEX  = { 'UVCGAN-S' : r'\uv', 'Area' : r'\area', 'unpaired FM' : r'\unp',
         'paired FM' : r'\pai', 'joint FM: M - B' : r'\jnt, $M - \Bh$',
         'joint FM: direct S' : r'\jnt, direct $\Sh$',
         'joint FM: direct S [mid32]' : r'\jnt, direct $\Sh$ (32)',
         'joint FM' : r'\jnt' }
RUNS = { 'unpaired FM' : [ f'bb_uvcgan_otcfm1_s{k}' for k in range(3) ],
         'paired FM' : [ f'bench_paired_bkg_s{k}' for k in range(3) ],
         'joint FM: M - B' : [ f'bench_joint_s{k}' for k in range(3) ],
         'joint FM: direct S' : [ f'bench_joint_s{k}' for k in range(3) ] }
OBS  = [ 'girth', 'mass', 'zlead', 'ptd', 'zg', 'rg' ]
OBS_TEX = { 'girth' : '$g$', 'mass' : '$m$', 'zlead' : '$z_{lead}$',
            'ptd' : '$p_T^D$', 'zg' : '$z_g$', 'rg' : '$r_g$' }

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Deck tables')
    parser.add_argument('--bench', default = 'docs/flow/bench')
    return parser.parse_args()

class Summary:
    """bench_summary.csv: value, seed half range, stat error by
    (set, label, R, metric)."""

    def __init__(self, path):
        self.df = pd.read_csv(path).set_index([ 'set', 'label', 'R', 'metric' ])

    def get(self, s, label, metric, r = 0.4):
        try:
            row = self.df.loc[(s, label, r, metric)]
        except KeyError:
            return (np.nan, np.nan, np.nan, 0)
        return (row['mean'], row['half_range'], row['stat_se'], row['n_seeds'])

def fmt(v, digits):
    return '--' if not np.isfinite(v) else f'{v:.{digits}f}'

def table(header, rows, colspec, note):
    lines = [ r'\begin{tabular}{' + colspec + '}', r'\toprule',
              ' & '.join(header) + r' \\', r'\midrule' ]
    lines += [ ' & '.join(r) + r' \\' for r in rows ]
    lines += [ r'\bottomrule', r'\end{tabular}' ]
    if note:
        lines += [ r'\par\vspace{0.15em}\snote{' + note + '}' ]
    return '\n'.join(lines) + '\n'

def jets_table(summ, s, arms, cols):
    """cols: [(metric, header, digits, R)]; the last row gives, per column,
    the median statistical error of one run."""
    rows = []
    ses  = [ [] for _ in cols ]
    for arm in arms:
        row = [ TEX[arm] ]
        for (k, (metric, _, digits, r)) in enumerate(cols):
            (v, hr, se, n) = summ.get(s, arm, metric, r)
            row.append(cell(v, hr, digits, n))
            if np.isfinite(se):
                ses[k].append(se)
        rows.append(row)
    rows.append([ r'\snote{stat.\ error}' ] + [
        r'\snote{' + (fmt(float(np.median(e)), d) if e else '--') + '}'
        for (e, (_, _, d, _)) in zip(ses, cols) ])
    header = [ '' ] + [ h for (_, h, _, _) in cols ]
    return table(header, rows, 'l' + 'r' * len(cols), '')

def cone_scores(run, truth, nfe, solver, decode):
    run_dir = os.path.join(fc.out_root(), run)
    path = os.path.join(run_dir, 'evals', f'{truth}_truth.csv')
    if not os.path.exists(path):
        return np.nan
    v = pd.read_csv(path)
    v = v[(v.step == best_step(run_dir)) & (v.net == 'ema') & (v.nfe == nfe)
          & (v.solver == solver) & (v.decode == decode) & (v.samples == 1)]
    return float(v.jer_cal.iloc[0]) if len(v) else np.nan

def cell(v, hr, digits, n):
    """The seed mean, with the seed half range in units of the last digit
    in brackets where it is at least 1: 0.68(4) = 0.68 +- 0.04."""
    out = fmt(v, digits)
    if n > 1 and np.isfinite(hr) and np.isfinite(v):
        u = int(round(hr * 10**digits))
        if u >= 1:
            out += f'({u})'
    return out

SEED_NOTE = (r'seed mean of 3; (n): seed half range in units of the last digit,'
             r' shown where $\geq$ 1')

def seeds_cell(vals, digits):
    vals = [ v for v in vals if np.isfinite(v) ]
    if not vals:
        return '--'
    if len(vals) == 1:
        return fmt(vals[0], digits)
    return f'{np.mean(vals):.{digits}f}$\\pm${(max(vals) - min(vals)) / 2:.{digits}f}'

def run_manifest(bench):
    """bench_runs.csv (the selected checkpoint of every run; checkpoints stay
    under OUTDIR, outside git) and a copy of each run's config.json."""
    rows = []
    os.makedirs(os.path.join(bench, 'configs'), exist_ok = True)
    for run in sorted({ r for rs in RUNS.values() for r in rs }):
        run_dir = os.path.join(fc.out_root(), run)
        with open(os.path.join(run_dir, 'config.json'), encoding = 'utf-8') as f:
            config = json.load(f)
        with open(os.path.join(bench, 'configs', f'{run}.json'), 'w',
                  encoding = 'utf-8') as f:
            json.dump(config, f, indent = 4)
        step = best_step(run_dir)
        v = pd.read_csv(os.path.join(run_dir, 'evals', 'val_truth.csv'))
        v = v[v.step == step].iloc[0]
        rows.append({ 'run' : run, 'method' : config['method'], 'seed' : config['seed'],
                      'backbone' : config.get('backbone'), 'selected_step' : step,
                      'minutes_at_checkpoint' : round(v.train_time / 60, 1),
                      'checkpoint' : os.path.join('OUTDIR/sphenix/flow', run, 'checkpoints',
                                                  f'step_{step:08d}.pt'),
                      'network' : 'ema', 'n_params' : config.get('n_params'),
                      'commit' : config['runs'][0].get('commit'),
                      'gpu' : config['runs'][0].get('gpu'),
                      'node' : config['runs'][0].get('node') })
    rows.append({ 'run' : 'UVCGAN-S published', 'method' : 'uvcgan-s',
                  'checkpoint' : 'OUTDIR/sphenix/pretrained/model_m(uvcgan-s)_d(resnet)'
                                 '_g(vit-modnet)_sgn_bkg_sub', 'network' : 'ema_gen_ba' })
    pd.DataFrame(rows).to_csv(os.path.join(bench, 'bench_runs.csv'), index = False)

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    out = os.path.join(cmdargs.bench, 'slides', 'tables')
    os.makedirs(out, exist_ok = True)
    run_manifest(cmdargs.bench)
    summ = Summary(os.path.join(cmdargs.bench, 'bench_summary.csv'))

    def write(name, text, size = 'scriptsize'):
        with open(os.path.join(out, name), 'w', encoding = 'utf-8') as f:
            f.write(f'\\{size}\n' + text)

    # slide 4: jets, val, R = 0.4
    cols = [ ('scale_20_30', 'scale', 3, 0.4), ('resolution_20_30', 'resolution', 3, 0.4),
             ('rmse_gev_20_30', 'RMSE, GeV', 2, 0.4),
             ('resolution_cal_20_30', r'res.\ (frozen cal.)', 3, 0.4),
             ('eff_14_20', r'eff.\ 14-20', 3, 0.4), ('fake_14_20', r'fake 14-20', 3, 0.4) ]
    write('jets.tex', jets_table(summ, 'val', ARMS, cols)
          + r'\par\vspace{0.15em}\snote{Matched jets, $20 < p_T^{real} < 30$ GeV; efficiency'
            r' ($p_T^{real}$), fake rate ($p_T^{sub}$), GeV. Raw, 4 Euler steps; (32): 32 NFE.'
            r' Seed mean of 3; (n): seed half range in the last digit.}' + '\n')

    # slide 5: substructure, val (seed spreads in the note: they differ by arm)
    fm = ARMS[:1] + ARMS[2:]
    cols = [ (f'{o}_{m}', o) for m in [ 'pair_rmse', 'w1' ] for o in OBS ]
    rows = []
    hr_bo, hr_j, ses = [], [], [ [] for _ in cols ]
    for arm in fm:
        row = [ TEX[arm] ]
        for (k, (metric, o)) in enumerate(cols):
            (v, hr, se, n) = summ.get('val', arm, metric)
            # in units of the spread of all truth jets of the selection
            sd = summ.get('val', 'UVCGAN-S', f'{o}_truth_all_sd')[0]
            row.append(fmt(v / sd, 2))
            if n > 1 and np.isfinite(hr):
                (hr_j if 'joint' in arm else hr_bo).append(hr / sd)
            if np.isfinite(se):
                ses[k].append(se / sd)
        rows.append(row)
    floor = [ r'\snote{truth vs truth}' ] + [ '' ] * len(OBS)
    for o in OBS:
        (f, _, _, _) = summ.get('val', 'UVCGAN-S', f'{o}_w1_floor')
        sd = summ.get('val', 'UVCGAN-S', f'{o}_truth_all_sd')[0]
        floor.append(r'\snote{' + fmt(f / sd, 2) + '}')
    rows.append(floor)
    header = [ '' ] + [ r'\multicolumn{6}{c}{per-jet RMSE / $\sigma_{truth}$}' ] \
        + [ r'\multicolumn{6}{c}{distribution $W_1$ / $\sigma_{truth}$}' ]
    sub = [ r'\setlength{\tabcolsep}{4pt}',
            r'\begin{tabular}{l' + 'r' * 12 + '}', r'\toprule', ' & '.join(header) + r' \\',
            r'\cmidrule(lr){2-7}\cmidrule(lr){8-13}',
            ' & '.join([ '' ] + [ OBS_TEX[o] for o in OBS ] * 2) + r' \\', r'\midrule' ]
    sub += [ ' & '.join(r) + r' \\' for r in rows ]
    sub += [ r'\bottomrule', r'\end{tabular}',
             r'\par\vspace{0.15em}\snote{R = 0.4, $20 < p_T^{real} < 30$ GeV, lower is better;'
             r' $\sigma_{truth}$: spread of all truth jets. Seed half range'
             f' $\\leq$ {max(hr_bo):.2f} (background-only), $\\leq$ {max(hr_j):.2f} (joint);'
             f' stat.\\ error $\\leq$ {max(max(e) for e in ses if e):.3f}.}}' ]
    write('sub.tex', '\n'.join(sub) + '\n', 'footnotesize')

    # slide 6: JEWEL
    rows = []
    for arm in ARMS:
        row = [ TEX[arm] ]
        for (metric, digits) in [ ('scale_20_30', 3), ('scale_cal_20_30', 3),
                                  ('resolution_cal_20_30', 3), ('fake_14_20', 3) ]:
            (v, hr, _, n) = summ.get('jewel', arm, metric)
            row.append(cell(v, hr, digits, n))
        for o in [ 'girth', 'zlead' ]:
            (v, hr, _, n) = summ.get('jewel', arm, f'{o}_pair_rmse')
            sd = summ.get('jewel', 'UVCGAN-S', f'{o}_truth_all_sd')[0]
            row.append(cell(v / sd, hr / sd, 2, n))
        for o in [ 'girth', 'zlead' ]:
            (v, hr, _, n) = summ.get('jewel-val', arm, f'{o}_mod_fraction')
            row.append(cell(v, hr, 2, n))
        rows.append(row)
    se = [ summ.get('jewel-val', 'UVCGAN-S', f'{o}_mod_fraction')[2]
           for o in [ 'girth', 'zlead' ] ]
    header = [ '', 'scale', r'scale$_{cal}$', r'res.$_{cal}$', 'fake 14-20',
               r'$g$ RMSE/$\sigma$', r'$z_{lead}$ RMSE/$\sigma$', r'$\Delta g$ kept',
               r'$\Delta z_{lead}$ kept' ]
    write('jewel.tex', r'\setlength{\tabcolsep}{4pt}' + '\n'
          + table(header, rows, 'l' + 'r' * 8, '')
          + r"\par\vspace{0.15em}\snote{JEWEL, R = 0.4, $20 < p_T^{real} < 30$ GeV, raw; cal: each run's"
            r' val (PYTHIA) calibration, frozen. Kept: the JEWEL$-$PYTHIA shift of the mean,'
            r" as a fraction of the truth's (stat.\ error "
          + ', '.join(f'{v:.2f}' for v in se) + '). ' + SEED_NOTE + '.}\n')

    # slide 7: towers
    tw = pd.read_csv(os.path.join(cmdargs.bench, 'bench_towers_summary.csv'))
    rows = []
    for arm in [ a for a in ARMS if a != 'Area' ]:
        d = tw[(tw.set == 'val') & (tw.label == arm)]
        if not len(d):
            continue
        d = d.iloc[0]
        rows.append([ TEX[arm], fmt(d['mae'], 3), fmt(100 * d['bkgerr_share[5-10]'], 0),
                      fmt(100 * d['bkgerr_share[>10]'], 0), fmt(d['event_bias_gev'], 0) ])
    write('towers.tex', r'\setlength{\tabcolsep}{3pt}' + '\n'
          + table([ '', 'MAE', r'\% $S$ lost, 5-10', r'$>$10 GeV', r'$\sum(\Sh - S)$' ],
                  rows, 'lrrrr', '')
          + r'\par\vspace{0.15em}\snote{val, raw; per tower (GeV), share of the true signal of towers with'
            r' $S$ in 5-10 / $>$10 GeV put into $\Bh$, GeV per event; seed mean.}' + '\n',
          'tiny')

    # backup: solvers and readouts
    rows = []
    for arm in BASE:
        for (setting, sfx, nfe, solver) in [ ('4 Euler', '', 4, 'euler'),
                                             ('32 NFE', ' [mid32]', 32, 'midpoint') ]:
            decode = 'direct' if 'direct' in arm else 'mixture'
            cone = [ cone_scores(r, 'val', nfe, solver, decode) for r in RUNS[arm] ]
            cone_j = [ cone_scores(r, 'jewel', nfe, solver, decode) for r in RUNS[arm] ]
            lab = arm + sfx
            row = [ TEX[arm] if not sfx else '', setting, seeds_cell(cone, 2),
                    seeds_cell(cone_j, 2) ]
            for (metric, digits) in [ ('scale_20_30', 3), ('resolution_cal_20_30', 3),
                                      ('fake_14_20', 3) ]:
                (v, hr, _, n) = summ.get('val', lab, metric)
                row.append(cell(v, hr, digits, n))
            (v, hr, _, n) = summ.get('val', lab, 'girth_pair_rmse')
            sd = summ.get('val', 'UVCGAN-S', 'girth_truth_all_sd')[0]
            row.append(cell(v / sd, hr / sd, 2, n))
            d = tw[(tw.set == 'val') & (tw.label == lab)]
            row.append(fmt(100 * d['bkgerr_share[>10]'].iloc[0], 0) if len(d) else '--')
            row.append(fmt(d['mae'].iloc[0], 3) if len(d) else '--')
            rows.append(row)
        if arm != BASE[-1]:
            rows.append([ r'\addlinespace[0.2em]' ] + [ '' ] * 9)
    cone64 = { arm : cone_scores(RUNS[arm][0], 'val', 64, 'midpoint',
                                 'direct' if 'direct' in arm else 'mixture')
               for arm in BASE }
    path = os.path.join(cmdargs.bench, 'bench_consistency.csv')
    cons = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(
        columns = [ 'set', 'label', 'cons_mean', 'cons_mae', 'cons_rms',
                    'cons_event_mean_gev' ])
    cons = cons[cons.set == 'val'].groupby('label')[[ 'cons_mean', 'cons_mae', 'cons_rms',
                                                      'cons_event_mean_gev' ]].mean()
    names = { 'joint FM' : '4 Euler', 'joint FM [mid32]' : '32 NFE' }
    note = ('Cone res.: calibrated resolution in the true R = 0.4 cone, GeV (the selection'
            ' metric), val and JEWEL, seed mean $\\pm$ half range. Jets: R = 0.4, 20-30 GeV;'
            ' fake 14-20 GeV; \\% $S{>}10$: energy of towers with $S > 10$ GeV put into $\\Bh$;'
            ' MAE per tower, GeV. Seed 0 at 64 NFE, val cone res.: '
            + ', '.join(f'{TEX[a]}{{}} {cone64[a]:.2f}' for a in cone64)
            + '. Joint $\\Bh + \\Sh - M$ before clean-up (val): '
            + '; '.join(f'{names.get(k, k)}: {r.cons_mae:.3f} GeV per tower (MAE),'
                        f' {r.cons_event_mean_gev:+.1f} GeV per event'
                        for (k, r) in cons.iterrows()) + '.')
    write('solver.tex', table([ '', 'solve', 'cone res.', 'JEWEL', 'scale',
                                r'res.$_{cal}$', 'fake', r'$g$ RMSE/$\sigma$',
                                r'\% $S{>}10$', 'MAE' ], rows, 'llrrrrrrrr', note))

    # slide 3: cost
    path = os.path.join(cmdargs.bench, 'bench_cost_runs.csv')
    if not os.path.exists(path):
        print(f'{path} missing: no cost table')
        return
    cost = pd.read_csv(path)
    rows = []
    names = { 'unpaired FM' : r'\unp', 'paired FM' : r'\pai', 'joint FM: M - B' : r'\jnt',
              'UVCGAN-S retrained, batch 4' : r'\uv{} retrained, batch 4',
              'UVCGAN-S retrained, batch 32' : r'\uv{} retrained, batch 32',
              'UVCGAN-S published' : r'\uv{} published (${\sim}$105 h)' }
    for (arm, name) in names.items():
        d = cost[cost.arm == arm]
        if not len(d):
            continue
        t_acc = '/'.join('--' if not np.isfinite(x) else f'{x:.2g}'
                         for x in d.h_T_acc.astype(float))
        num = lambda c: d[c].astype(float) if c in d else pd.Series(dtype = float)  # pylint: disable=unnecessary-lambda-assignment,cell-var-from-loop
        lat4 = num('ms_event_euler4').dropna()
        lat1 = num('ms_event_1pass').dropna()
        lat32 = num('ms_event_midpoint32').dropna()
        rows.append([ name, seeds_cell(list(num('best_val_jer_cal')), 2),
                      seeds_cell(list(num('val_jer_cal_at_2h')), 2),
                      t_acc if name != names['UVCGAN-S published'] else '--',
                      seeds_cell(list(num('steps_per_s').dropna()), 2)
                      if num('steps_per_s').notna().any() else '--',
                      fmt(float(num('peak_mem_gb').mean()), 1)
                      if num('peak_mem_gb').notna().any() else '--',
                      fmt(float(lat4.median()), 2) if len(lat4)
                      else (fmt(float(lat1.median()), 2) + ' (1 pass)' if len(lat1) else '--'),
                      fmt(float(lat32.median()), 1) if len(lat32) else '--' ])
    write('cost.tex', table([ '', 'best val res.', 'at 2 h', 'h to 3.70', 'updates/s',
                              'peak GB', 'ms/event, 4 NFE', '32 NFE' ], rows, 'lrrrrrrr', '')
          + r'\par\vspace{0.15em}\snote{One RTX A6000 each; flows 2 h, \uv{} retrained 24 h. val res.:'
            r' the selection metric (calibrated cone resolution, GeV), seed mean $\pm$ half'
            r' range. h to 3.70 (T\_acc = published + 3\%): first confirmed checkpoint, per'
            r" seed. Inference: batch 500. The joint arm's second output costs $<$1\%.}" + '\n')
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
