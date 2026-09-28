#!/usr/bin/env python
"""Cost side of the consolidated subtraction benchmark (FLOW_NOTES.md,
"Consolidated benchmark"): validation quality against GPU-hours, time to
target, training throughput and memory, and inference latency, all on one
RTX A6000.

- Quality: val `jer_cal` (the selection metric: calibrated cone energy
  resolution of the 20k val events, GeV) of the EMA network at each arm's
  selection readout (4 Euler steps, M - B_hat), against training hours;
  UVCGAN-S retrained from scratch (batch 4 = the published configuration,
  and batch 32; 3 seeds each; scored every 10k updates) against the
  cumulative epoch time; the published checkpoint (800k updates, ~105 h)
  as a point.
- Time to T_acc (3.70 GeV): the first evaluation at or below it that the
  next one confirms (fm_compare.time_to).
- Throughput and peak memory: each run's summary.json.
- Latency: OUTDIR/sphenix/flow/latency.csv (fm_eval.py --latency: 500
  events, GPU-synchronised median of 5, after a warm-up).

    bench_cost.py --arms 'LABEL=RUN+RUN+RUN,...' --baseline DIR ... \
                  --reference DIR [--out docs/flow/bench]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fm_common as fc
from fm_compare import TARGETS, baseline_curves, flow_curves, time_to

plt.rcParams.update({ 'font.size' : 8, 'axes.titlesize' : 8,
                      'axes.labelsize' : 8, 'legend.fontsize' : 6.5,
                      'xtick.labelsize' : 7, 'ytick.labelsize' : 7 })

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Benchmark cost')
    parser.add_argument('--arms', required = True)
    parser.add_argument('--baseline', nargs = '*', default = [])
    parser.add_argument('--reference', default = None)
    parser.add_argument('--out', default = 'docs/flow/bench')
    return parser.parse_args()

def latency_table():
    lat = pd.read_csv(os.path.join(fc.out_root(), 'latency.csv'))
    lat = lat[lat.gpu == 'NVIDIA RTX A6000']
    return lat

def ms_per_event(lat, run, nfe, solver):
    x = lat[lat.model.str.startswith(f'{run} (') & (lat.nfe == nfe)
            & (lat.solver == solver)]
    return float(x.ms_per_event.median()) if len(x) else np.nan

def band_curve(ax, curves, label, **kw):
    grid = np.unique(np.round(curves.hours, 3))
    vals = []
    for (_, x) in curves.groupby('run'):
        x = x.sort_values('hours')
        vals.append(np.interp(grid, np.round(x.hours, 3), x.jer_cal, left = np.nan,
                              right = np.nan))
    vals = np.array(vals)
    with np.errstate(all = 'ignore'):
        mean = np.nanmean(vals, axis = 0)
    n = curves.run.nunique()
    ax.plot(grid, mean, lw = 1.3, marker = 'o', ms = 2,
            label = f'{label} ({n} seed{"s" if n > 1 else ""})', **kw)
    if n > 1:
        ax.fill_between(grid, np.nanmin(vals, axis = 0), np.nanmax(vals, axis = 0),
                        color = kw.get('color'), alpha = 0.18, lw = 0)

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    os.makedirs(cmdargs.out, exist_ok = True)
    style = json.load(open(os.path.join(os.path.dirname(__file__), 'bench_style.json'),
                           encoding = 'utf-8'))
    lat  = latency_table()
    rows = []
    flow = {}
    for item in cmdargs.arms.split(','):
        (label, runs) = item.split('=', 1)
        curves = []
        for run in runs.split('+'):
            run_dir = os.path.join(fc.out_root(), run)
            (v, config, summary) = flow_curves(run_dir)
            v = v[v.net == 'ema']
            curves.append(v)
            best = v.loc[v.jer_cal.idxmin()]
            (h_acc, h_acc_i) = time_to(v, TARGETS['T_acc'])
            rows.append({
                'arm' : label, 'run' : run, 'seed' : config['seed'],
                'hours_trained' : summary['train_time'] / 3600,
                'best_val_jer_cal' : best.jer_cal, 'best_hours' : best.hours,
                'best_step' : int(best.step),
                'h_T_acc' : h_acc, 'h_T_acc_interp' : h_acc_i,
                'steps_per_s' : summary['steps_per_s'],
                'samples_per_s' : summary['samples_per_s'],
                'coupling_frac' : summary['coupling_frac'],
                'peak_mem_gb' : summary['peak_mem_gb'],
                'n_params' : summary['n_params'],
                'ms_event_euler4' : ms_per_event(lat, run, 4, 'euler'),
                'ms_event_midpoint32' : ms_per_event(lat, run, 32, 'midpoint'),
                'gpu' : config['runs'][0]['gpu'], 'node' : config['runs'][0]['node'],
            })
        flow[label] = pd.concat(curves)
    base = {}
    for run_dir in cmdargs.baseline:
        (v, lab) = baseline_curves(run_dir.rstrip('/'))
        v = v[v.net == 'ema']
        arm = v.arm.iloc[0].replace('uvcgan-s', 'UVCGAN-S retrained,')
        base.setdefault(arm, []).append(v)
        best = v.loc[v.jer_cal.idxmin()]
        (h_acc, h_acc_i) = time_to(v, TARGETS['T_acc'])
        at2 = float(np.interp(2.0, v.sort_values('hours').hours,
                              v.sort_values('hours').jer_cal)) \
            if v.hours.min() <= 2.0 else np.nan
        rows.append({ 'arm' : arm, 'run' : lab, 'seed' : int(v.seed.iloc[0]),
                      'hours_trained' : v.hours.max(),
                      'best_val_jer_cal' : best.jer_cal, 'best_hours' : best.hours,
                      'best_step' : int(best.step), 'h_T_acc' : h_acc,
                      'h_T_acc_interp' : h_acc_i, 'val_jer_cal_at_2h' : at2,
                      'ms_event_euler4' : np.nan, 'ms_event_midpoint32' : np.nan })
    ref = None
    if cmdargs.reference:
        r = pd.read_csv(os.path.join(cmdargs.reference, 'evals', 'val_truth.csv'))
        ref = float(r[r.net == 'ema'].jer_cal.iloc[0])
        uv = lat[lat.model.str.startswith('uvcgan-s published')]
        rows.append({ 'arm' : 'UVCGAN-S published', 'run' : 'published',
                      'hours_trained' : 105.0, 'best_val_jer_cal' : ref,
                      'ms_event_1pass' : float(uv.ms_per_event.median()),
                      'n_params' : int(uv.n_params.iloc[0]) })
    df = pd.DataFrame(rows)
    for (label, x) in flow.items():
        at2 = []
        for (_, xr) in x.groupby('run'):
            xr = xr.sort_values('hours')
            at2.append(float(np.interp(2.0, xr.hours, xr.jer_cal)))
        df.loc[df.arm == label, 'val_jer_cal_at_2h'] = at2
    df.to_csv(os.path.join(cmdargs.out, 'bench_cost_runs.csv'), index = False)

    num = df.select_dtypes('number').columns.drop('seed', errors = 'ignore')
    arms = df.groupby('arm', sort = False)[list(num)].agg([ 'mean', 'min', 'max' ])
    arms.columns = [ f'{a}_{b}' for (a, b) in arms.columns ]
    arms.insert(0, 'seeds', df.groupby('arm', sort = False).size())
    arms.to_csv(os.path.join(cmdargs.out, 'bench_cost.csv'))

    (fig, axes) = plt.subplots(1, 2, figsize = (6.6, 2.7),
                               gridspec_kw = { 'width_ratios' : [ 1.6, 1 ] })
    ax = axes[0]
    for (label, x) in flow.items():
        band_curve(ax, x, label, color = style[label]['color'])
    for (arm, vs) in base.items():
        band_curve(ax, pd.concat(vs), arm, color = style['UVCGAN-S']['color'],
                   ls = '-' if 'batch 4' in arm else ':')
    if ref is not None:
        ax.plot([ 105 ], [ ref ], marker = '*', ms = 8, color = style['UVCGAN-S']['color'],
                ls = 'none', label = 'UVCGAN-S published (~105 h)')
    ax.axhline(TARGETS['T_acc'], color = 'grey', ls = '--', lw = 0.8)
    ax.annotate('T_acc 3.70', (0.21, TARGETS['T_acc']), fontsize = 6.5, color = 'grey',
                va = 'bottom')
    ax.axvline(2.0, color = 'k', lw = 0.6, ls = ':')
    ax.set_xscale('log')
    ax.set_xlim(0.2, 150)
    ax.set_ylim(3.4, 5.6)
    ax.set_xlabel('training time on one RTX A6000, hours')
    ax.set_ylabel('val jer_cal (selection metric), GeV')
    ax.grid(alpha = 0.3, which = 'both')
    ax.legend(fontsize = 5.5, loc = 'upper right')

    ax = axes[1]
    names, vals, cols = [], [], []
    uv = lat[lat.model.str.startswith('uvcgan-s published')]
    names.append('UVCGAN-S\n1 pass')
    vals.append(float(uv.ms_per_event.median()))
    cols.append(style['UVCGAN-S']['color'])
    for label in flow:
        d = df[df.arm == label]
        for (key, nfe) in [ ('ms_event_euler4', '4 Euler'),
                            ('ms_event_midpoint32', '32 NFE') ]:
            if d[key].notna().any():
                names.append(f'{label}\n{nfe}')
                vals.append(float(d[key].median()))
                cols.append(style[label]['color'])
    ax.barh(np.arange(len(vals)), vals, color = cols)
    for (i, v) in enumerate(vals):
        ax.text(v * 1.1, i, f'{v:.2g}', va = 'center', fontsize = 6.5)
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(names, fontsize = 5.5)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlim(0.1, 60)
    ax.set_xlabel('inference, ms per event (batch 500)')
    ax.grid(alpha = 0.3, axis = 'x', which = 'both')
    fig.tight_layout()
    fig.savefig(os.path.join(cmdargs.out, 'bench_cost.png'), dpi = 150)
    plt.close(fig)

    show = [ 'best_val_jer_cal', 'val_jer_cal_at_2h', 'h_T_acc', 'steps_per_s',
             'peak_mem_gb', 'n_params', 'ms_event_euler4', 'ms_event_midpoint32' ]
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(df[[ 'arm', 'run' ] + [ c for c in show if c in df ]].round(3)
              .to_string(index = False))

if __name__ == '__main__':
    main()
