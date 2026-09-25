#!/usr/bin/env python
"""Time to quality of the flow runs against the UVCGAN-S baseline.

    fm_compare.py --flow RUN_DIR [RUN_DIR ...] --baseline RUN_DIR [...]
                  [--reference PUBLISHED_DIR] [--out PREFIX]

Flow runs: `evals/val_truth.csv` of fm_eval.py (checkpoints scored at the
method's selection setting, `fm_eval.SELECTION`) against their training
time.
Baseline runs: `evals/val_truth.csv` of eval_val_truth.py against the
cumulative `epoch_time` of `history.csv` (the training loop, without the
held-out scoring of the callback). Both are one GPU (RTX A6000).

Time to a target (FLOW_NOTES.md): the first evaluation at or below it that
the next evaluation confirms; "-" when not reached. Also given: the time
interpolated linearly to the first crossing (the baseline is scored only
every 10k updates, 1.3-1.7 h apart).

Also merged where present: the JEWEL score of each run's val-selected
checkpoint (EMA; `evals/jewel_truth.csv`), the inference latency of
`OUTDIR/sphenix/flow/latency.csv` at the selection setting, and the
solver curve (val jer_cal against NFE at the selected checkpoint).

Writes PREFIX.csv (one row per run and network), PREFIX_arms.csv (seeds
pooled: mean and half range), PREFIX_curves.csv (every scored point, both
families, on a common hours axis), PREFIX.png and PREFIX_nfe.png.
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fm_eval import SELECTION

TARGETS = { 'T_useful' : 4.00, 'T_acc' : 3.70, 'T_match' : 3.65 }

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Flow vs baseline')
    parser.add_argument('--flow', nargs = '*', default = [])
    parser.add_argument('--baseline', nargs = '*', default = [])
    parser.add_argument('--reference', default = None)
    parser.add_argument('--out', default = 'outdir/sphenix/flow/compare')
    return parser.parse_args()

def flow_curves(run_dir):
    with open(os.path.join(run_dir, 'config.json'), 'r', encoding = 'utf-8') as f:
        config = json.load(f)

    v = pd.read_csv(os.path.join(run_dir, 'evals', 'val_truth.csv'))
    (nfe, solver, decode) = SELECTION[config['method']]
    v = v[(v.nfe == nfe) & (v.solver == solver) & (v.decode == decode)
          & (v.samples == 1)].copy()

    v['hours']  = v.train_time / 3600
    v['family'] = 'flow'
    v['arm']    = config['method'] + (' (m - b)' if decode == 'mixture' else '')
    v['seed']   = config['seed']
    v['run']    = config['label']

    summary = {}
    path = os.path.join(run_dir, 'summary.json')
    if os.path.exists(path):
        with open(path, 'r', encoding = 'utf-8') as f:
            summary = json.load(f)

    return (v, config, summary)

def baseline_curves(run_dir):
    with open(os.path.join(run_dir, 'label'), 'r', encoding = 'utf-8') as f:
        label = f.read().strip()

    m = re.match(r'base_b(\d+)_lr([^_]+)(?:_s(\d+))?$', label)
    h = pd.read_csv(os.path.join(run_dir, 'history.csv'))
    h['hours'] = h.epoch_time.cumsum() / 3600

    v = pd.read_csv(os.path.join(run_dir, 'evals', 'val_truth.csv'))
    v = v[v.epoch > 0].merge(h[[ 'epoch', 'hours' ]], on = 'epoch')

    v['family'] = 'uvcgan-s'
    v['arm']    = f'uvcgan-s batch {m.group(1)}'
    v['seed']   = int(m.group(3) or 0)
    v['run']    = label
    v['step']   = v.updates

    return (v, label)

def jewel_at(run_dir, family, step, method = None):
    """(jer_cal, l1_sig) on JEWEL of the EMA network at a checkpoint."""
    csv = os.path.join(run_dir, 'evals', 'jewel_truth.csv')
    if not os.path.exists(csv):
        return (None, None)

    j = pd.read_csv(csv)
    j = j[j.net == 'ema']

    if family == 'flow':
        (nfe, solver, decode) = SELECTION[method]
        j = j[(j.step == step) & (j.nfe == nfe) & (j.solver == solver)
              & (j.decode == decode) & (j.samples == 1)]
    else:
        j = j[j.updates == step]

    if len(j) == 0:
        return (None, None)

    return (float(j.jer_cal.iloc[0]), float(j.l1_sig.iloc[0]))

def latency(method):
    """ms per event of a method at its selection setting, and of the
    published UVCGAN-S generator."""
    path = os.path.join(
        os.environ.get('UVCGAN_S_OUTDIR', 'outdir'), 'sphenix', 'flow',
        'latency.csv'
    )
    if not os.path.exists(path):
        return None

    lat = pd.read_csv(path)
    if method is None:
        x = lat[lat.model.str.startswith('uvcgan-s')]
    else:
        (nfe, solver, _) = SELECTION[method]
        x = lat[lat.model.str.endswith(f'({method})') & (lat.nfe == nfe)
                & (lat.solver == solver) & (lat.samples == 1)]

    return float(x.ms_per_event.median()) if len(x) else None

def time_to(x, target):
    """(confirmed, interpolated) hours to jer_cal <= target."""
    x = x.sort_values('hours')
    (t, y) = (x.hours.values, x.jer_cal.values)

    confirmed = None
    for i in range(len(y) - 1):
        if (y[i] <= target) and (y[i + 1] <= target):
            confirmed = t[i]
            break

    interp = None
    below  = np.nonzero(y <= target)[0]
    if len(below):
        i = below[0]
        interp = t[0] if i == 0 else float(np.interp(
            target, [ y[i], y[i - 1] ], [ t[i], t[i - 1] ]
        ))

    return (confirmed, interp)

def summarize(curves, summaries, runs):
    # pylint: disable=too-many-locals
    rows = []

    for ((run, net), x) in curves.groupby([ 'run', 'net' ]):
        best = x.loc[x.jer_cal.idxmin()]
        last = x.loc[x.hours.idxmax()]
        (run_dir, method) = runs[run]
        (jewel, jewel_l1) = jewel_at(
            run_dir, x.family.iloc[0], best.step, method
        )
        row  = {
            'family' : x.family.iloc[0], 'arm' : x.arm.iloc[0],
            'seed' : x.seed.iloc[0], 'run' : run, 'net' : net,
            'hours_trained' : last.hours,
            'final_jer_cal' : last.jer_cal, 'final_l1_sig' : last.l1_sig,
            'best_jer_cal' : best.jer_cal, 'best_hours' : best.hours,
            'best_step' : best.step, 'best_l1_sig' : best.l1_sig,
            'best_jes' : best.jes,
            'jewel_jer_cal' : jewel if net == 'ema' else None,
            'jewel_l1_sig' : jewel_l1 if net == 'ema' else None,
            'ms_per_event' : latency(method),
        }

        for (name, target) in TARGETS.items():
            (confirmed, interp) = time_to(x, target)
            row[f'h_{name}']        = confirmed
            row[f'h_{name}_interp'] = interp

        row.update({
            f'train_{k}' : v for (k, v) in summaries.get(run, {}).items()
                if k in ('steps_per_s', 'samples_per_s', 'coupling_frac',
                         'peak_mem_gb', 'n_params', 'startup_time',
                         'load_time', 'eval_time', 'end_to_end_time')
        })
        rows.append(row)

    return pd.DataFrame(rows)

def arms_table(table):
    """Seeds pooled per arm, EMA network: mean and half range."""
    rows = []
    cols = [ 'best_jer_cal', 'final_jer_cal', 'hours_trained', 'best_hours',
             'h_T_useful', 'h_T_acc', 'h_T_match', 'h_T_acc_interp',
             'jewel_jer_cal', 'best_l1_sig', 'best_jes', 'ms_per_event',
             'train_steps_per_s', 'train_samples_per_s', 'train_peak_mem_gb',
             'train_n_params', 'train_coupling_frac' ]

    for (arm, x) in table[table.net == 'ema'].groupby('arm'):
        row = { 'arm' : arm, 'seeds' : len(x) }

        for c in cols:
            if c not in x:
                continue
            v = pd.to_numeric(x[c], errors = 'coerce')
            n = int(v.notna().sum())
            row[c] = v.mean() if n else None
            row[f'{c}_hr'] = (v.max() - v.min()) / 2 if n > 1 else None
            if c.startswith('h_T'):
                row[f'{c}_reached'] = f'{n}/{len(x)}'

        rows.append(row)

    return pd.DataFrame(rows)

def plot_nfe(runs, out):
    """val jer_cal against NFE at each flow run's selected checkpoint."""
    (fig, ax) = plt.subplots(figsize = (7, 5))
    drawn = False

    for (run, (run_dir, method)) in runs.items():
        if method is None or method.startswith('regress'):
            continue

        v = pd.read_csv(os.path.join(run_dir, 'evals', 'val_truth.csv'))
        (nfe, solver, decode) = SELECTION[method]
        sel = v[(v.net == 'ema') & (v.nfe == nfe) & (v.solver == solver)
                & (v.decode == decode) & (v.samples == 1)]
        if len(sel) == 0:
            continue

        step = sel.loc[sel.jer_cal.idxmin()].step
        v = v[(v.net == 'ema') & (v.step == step) & (v.decode == decode)]

        for ((solv, samples), x) in v.groupby([ 'solver', 'samples' ]):
            if len(x) < 2 and samples == 1:
                continue
            x = x.sort_values('nfe')
            ax.plot(x.nfe * samples, x.jer_cal, marker = 'o',
                    label = f'{run} {solv}' + (f' x{samples}' if samples > 1
                                              else ''))
            drawn = True

    if not drawn:
        plt.close(fig)
        return

    ax.set_xscale('log', base = 2)
    ax.set_xlabel('network evaluations per event (NFE x samples)')
    ax.set_ylabel('val jer_cal, GeV (EMA, selected checkpoint)')
    ax.grid(alpha = 0.3, which = 'both')
    ax.legend(fontsize = 7)
    fig.tight_layout()
    fig.savefig(out, dpi = 110)

def plot(curves, reference, out):
    arms   = sorted(curves.arm.unique())
    colors = dict(zip(arms, plt.cm.tab10(np.arange(len(arms)))))

    (fig, axes) = plt.subplots(2, 2, figsize = (15, 10))

    for (col, net) in enumerate([ 'ema', 'raw' ]):
        for (row, (metric, label)) in enumerate([
            ('jer_cal', 'val jer_cal, GeV'), ('l1_sig', 'val l1_sig, GeV')
        ]):
            ax = axes[row][col]

            for ((run, arm), x) in curves[curves.net == net].groupby(
                [ 'run', 'arm' ]
            ):
                x = x.sort_values('hours')
                ax.plot(x.hours, x[metric], color = colors[arm],
                        marker = 'o', ms = 2.5, lw = 1.1, label = run)

            if metric == 'jer_cal':
                for (name, target) in TARGETS.items():
                    ax.axhline(target, color = 'grey', ls = ':', lw = 0.8)
                    ax.text(ax.get_xlim()[0], target, f' {name}',
                            fontsize = 7, va = 'bottom', color = 'grey')
                ax.axhline(5.28, color = 'k', ls = '-.', lw = 0.8,
                           label = 'median-rho')
                ax.set_ylim(3.4, 6.0)

            ref = reference.get(net)
            if ref is not None:
                ax.axhline(ref[metric], color = 'k', ls = '--', lw = 1,
                           label = 'published, 800k updates')

            ax.set_xscale('log')
            ax.set_xlabel('training time on one A6000, hours')
            ax.set_ylabel(f'{label} ({net})')
            ax.grid(alpha = 0.3, which = 'both')

            if (row == 0) and (col == 0):
                ax.legend(fontsize = 6, ncol = 2)

    fig.tight_layout()
    fig.savefig(out, dpi = 110)

def main():
    cmdargs = parse_cmdargs()

    curves    = []
    summaries = {}
    runs      = {}

    for run_dir in cmdargs.flow:
        (v, config, summary) = flow_curves(run_dir.rstrip('/'))
        curves.append(v)
        summaries[config['label']] = summary
        runs[config['label']] = (run_dir.rstrip('/'), config['method'])

    for run_dir in cmdargs.baseline:
        (v, label) = baseline_curves(run_dir.rstrip('/'))
        curves.append(v)
        runs[label] = (run_dir.rstrip('/'), None)

    cols   = [ 'family', 'arm', 'seed', 'run', 'net', 'step', 'hours',
               'jer_cal', 'l1_sig', 'l1_bkg', 'jes', 'bias' ]
    curves = pd.concat([ c[cols] for c in curves ], ignore_index = True)

    reference = {}
    if cmdargs.reference:
        r = pd.read_csv(os.path.join(cmdargs.reference, 'evals',
                                     'val_truth.csv'))
        reference = { n : r[r.net == n].iloc[0] for n in r.net.unique() }

    table = summarize(curves, summaries, runs)
    arms  = arms_table(table)
    os.makedirs(os.path.dirname(cmdargs.out) or '.', exist_ok = True)
    table.to_csv(f'{cmdargs.out}.csv', index = False)
    arms.to_csv(f'{cmdargs.out}_arms.csv', index = False)
    curves.to_csv(f'{cmdargs.out}_curves.csv', index = False)

    show = [ 'arm', 'seed', 'net', 'hours_trained', 'best_jer_cal',
             'best_hours', 'final_jer_cal', 'h_T_useful', 'h_T_acc',
             'h_T_match', 'h_T_acc_interp' ]
    with pd.option_context('display.width', 200):
        print(table[show].sort_values([ 'net', 'arm', 'seed' ])
              .round(3).to_string(index = False))

    with pd.option_context('display.width', 250):
        print()
        print(arms.round(3).to_string(index = False))

    plot(curves, reference, f'{cmdargs.out}.png')
    plot_nfe(runs, f'{cmdargs.out}_nfe.png')
    print(f'wrote {cmdargs.out}.csv, {cmdargs.out}_arms.csv,'
          f' {cmdargs.out}_curves.csv, {cmdargs.out}.png, {cmdargs.out}_nfe.png')

if __name__ == '__main__':
    main()
