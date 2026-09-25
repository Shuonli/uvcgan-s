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
    parser.add_argument('--extra-samples', type = int, default = None,
        help = 'also show conditional CFM averaged over this many samples'
               ' (only the checkpoints scored that way)')
    return parser.parse_args()

def flow_curves(run_dir, samples = None):
    """Scores of a flow run at its selection setting; `samples` overrides
    the number of samples averaged (a second arm of the same run)."""
    with open(os.path.join(run_dir, 'config.json'), 'r', encoding = 'utf-8') as f:
        config = json.load(f)

    v = pd.read_csv(os.path.join(run_dir, 'evals', 'val_truth.csv'))
    (nfe, solver, decode, default) = SELECTION[config['method']]
    samples = samples or default
    v = v[(v.nfe == nfe) & (v.solver == solver) & (v.decode == decode)
          & (v.samples == samples)].copy()

    v['hours']  = v.train_time / 3600
    v['family'] = 'flow'
    v['arm']    = config['method'] + (
        ' (m - b)' if decode == 'mixture' else ''
    ) + (f' (mean of {samples})' if samples > 1 else '')
    v['seed']   = config['seed']
    v['run']    = config['label'] + (
        f' x{samples}' if samples != default else ''
    )

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

def jewel_at(run_dir, family, step, method = None, samples = None):
    """(jer_cal, l1_sig) on JEWEL of the EMA network at a checkpoint."""
    csv = os.path.join(run_dir, 'evals', 'jewel_truth.csv')
    if not os.path.exists(csv):
        return (None, None)

    j = pd.read_csv(csv)
    j = j[j.net == 'ema']

    if family == 'flow':
        (nfe, solver, decode, default) = SELECTION[method]
        j = j[(j.step == step) & (j.nfe == nfe) & (j.solver == solver)
              & (j.decode == decode) & (j.samples == (samples or default))]
    else:
        j = j[j.updates == step]

    if len(j) == 0:
        return (None, None)

    return (float(j.jer_cal.iloc[0]), float(j.l1_sig.iloc[0]))

def latency(method, samples = None):
    """ms per event of a method at its selection setting (or averaging
    `samples`), and of the published UVCGAN-S generator."""
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
        (nfe, solver, _, default) = SELECTION[method]
        x = lat[lat.model.str.endswith(f'({method})') & (lat.nfe == nfe)
                & (lat.solver == solver)]
        if len(x):
            # the samples of a mean run one after the other
            return float(x.ms_per_event.median()) * (samples or default)

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
        (run_dir, method, samples) = runs[run]
        (jewel, jewel_l1) = jewel_at(
            run_dir, x.family.iloc[0], best.step, method, samples
        )
        row  = {
            'family' : x.family.iloc[0], 'arm' : x.arm.iloc[0],
            'seed' : x.seed.iloc[0], 'run' : run, 'net' : net,
            'hours_trained' : last.hours,
            'final_jer_cal' : last.jer_cal, 'final_l1_sig' : last.l1_sig,
            'best_jer_cal' : best.jer_cal, 'best_hours' : best.hours,
            'best_step' : best.step, 'best_l1_sig' : best.l1_sig,
            'best_mse_sig' : best.get('mse_sig'), 'best_jes' : best.jes,
            'jewel_jer_cal' : jewel if net == 'ema' else None,
            'jewel_l1_sig' : jewel_l1 if net == 'ema' else None,
            'ms_per_event' : latency(method, samples),
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
             'jewel_jer_cal', 'best_l1_sig', 'best_mse_sig', 'best_jes',
             'ms_per_event',
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

MATCHED_HOURS = [ 0.25, 0.5, 1.0, 2.0, 3.0, 8.0, 16.0, 24.0 ]

def matched_table(curves):
    """val jer_cal of the EMA network at fixed training times, per arm:
    mean of the seeds, half their range and how many reached that time."""
    rows = []

    for (arm, x) in curves[curves.net == 'ema'].groupby('arm'):
        row = { 'arm' : arm, 'seeds' : x.run.nunique() }

        for h in MATCHED_HOURS:
            vals = []
            for (_, xr) in x.groupby('run'):
                xr = xr.sort_values('hours')
                # 1e-2 h: checkpoints land a few seconds after their mark
                if xr.hours.iloc[0] - 1e-2 <= h <= xr.hours.iloc[-1] + 1e-2:
                    vals.append(float(np.interp(h, xr.hours, xr.jer_cal)))

            if vals:
                row[f'{h:g}h'] = (
                    f'{np.mean(vals):.2f}+-{(max(vals) - min(vals)) / 2:.2f}'
                    + (f' ({len(vals)})' if len(vals) < row['seeds'] else '')
                )
        rows.append(row)

    cols = [ 'arm', 'seeds' ] + [ f'{h:g}h' for h in MATCHED_HOURS ]
    return pd.DataFrame(rows).reindex(columns = cols)

def plot_nfe(runs, out):
    """val jer_cal against network evaluations per event (NFE x samples),
    for each flow run at the checkpoint with the most solver settings
    scored (the solver sweep)."""
    (fig, ax) = plt.subplots(figsize = (7.5, 5))
    drawn = False

    for (run, (run_dir, method, samples)) in runs.items():
        if (method is None) or method.startswith('regress') or samples:
            continue

        v = pd.read_csv(os.path.join(run_dir, 'evals', 'val_truth.csv'))
        decode = SELECTION[method][2]
        v = v[(v.net == 'ema') & (v.decode == decode)]
        if len(v) == 0:
            continue

        step = v.groupby('step').size().idxmax()
        v = v[v.step == step]
        minutes = v.train_time.iloc[0] / 60

        for ((solv, k), x) in v.groupby([ 'solver', 'samples' ]):
            if len(x) < 2 and k == 1:
                continue
            x = x.sort_values('nfe')
            ax.plot(x.nfe * k, x.jer_cal, marker = 'o',
                    label = f'{method} {solv}'
                            + (f', mean of {k}' if k > 1 else '')
                            + f' ({run}, {minutes:.0f} min)')
            drawn = True

    if not drawn:
        plt.close(fig)
        return

    ax.axhline(3.70, color = 'grey', ls = ':', lw = 0.8)
    ax.set_xscale('log', base = 2)
    ax.set_ylim(3.4, 5.3)
    ax.set_xlabel('network evaluations per event (NFE x samples averaged)')
    ax.set_ylabel('val jer_cal, GeV (EMA)')
    ax.grid(alpha = 0.3, which = 'both')
    ax.legend(fontsize = 7)
    fig.tight_layout()
    fig.savefig(out, dpi = 110)

ARM_STYLE = {
    'uvcgan-s batch 32'    : ('#7f7f7f', '-'),
    'uvcgan-s batch 4'     : ('#000000', '-'),
    'otcfm (m - b)'        : ('#2ca02c', '-'),
    'condcfm (mean of 4)'  : ('#ff7f0e', '-'),
    'condcfm (mean of 16)' : ('#d62728', '-'),
    'regress_l1'           : ('#9467bd', '--'),
    'regress'              : ('#8c564b', '--'),
    'sbcfm (m - b)'        : ('#17becf', ':'),
}

def plot_report(curves, table, reference, out):
    """Three panels: val jer_cal and l1_sig against training hours (mean
    of the seeds, band = their range, EMA network at each arm's selection
    setting), and JEWEL against val at each run's selected checkpoint."""
    # pylint: disable=too-many-locals
    (fig, axes) = plt.subplots(1, 4, figsize = (25, 6))
    ema = curves[curves.net == 'ema']
    lat = table.set_index('run').ms_per_event.to_dict()

    for (arm, x) in ema.groupby('arm'):
        (color, ls) = ARM_STYLE.get(arm, (None, '-'))
        runs = x.run.unique()
        ms   = np.nanmedian([ lat.get(r, np.nan) for r in runs ])
        label = f'{arm} ({len(runs)} seed{"s" if len(runs) > 1 else ""}'
        label += f', {ms:.2g} ms/event)' if np.isfinite(ms) else ')'

        for (col, metric) in enumerate([ 'jer_cal', 'l1_sig', 'mse_sig' ]):
            if x[metric].notna().sum() == 0:
                continue
            ax = axes[col]
            # a common grid of times over which every seed has a value
            grid = np.unique(np.round(x.hours, 3))
            vals = []
            for r in runs:
                xr = x[(x.run == r) & x[metric].notna()].sort_values('hours')
                if len(xr) == 0:
                    continue
                v  = np.interp(grid, xr.hours, xr[metric], left = np.nan,
                               right = np.nan)
                vals.append(v)
            if not vals:
                continue
            vals = np.array(vals)
            mean = np.nanmean(vals, axis = 0)
            ax.plot(grid, mean, color = color, ls = ls, lw = 1.6,
                    marker = 'o', ms = 2.5, label = label)
            if len(runs) > 1:
                ax.fill_between(grid, np.nanmin(vals, axis = 0),
                                np.nanmax(vals, axis = 0), color = color,
                                alpha = 0.2, lw = 0)

    ax = axes[0]
    for (name, target) in TARGETS.items():
        ax.axhline(target, color = 'grey', ls = ':', lw = 0.8)
        ax.annotate(f'{name} {target:.2f}', (1.0, target),
                    xycoords = ('axes fraction', 'data'), fontsize = 7,
                    color = 'grey', ha = 'right', va = 'bottom')
    ax.axhline(5.28, color = 'k', ls = '-.', lw = 0.8)
    ax.annotate('median-rho 5.28', (1.0, 5.28), xycoords = ('axes fraction',
                'data'), fontsize = 7, ha = 'right', va = 'bottom')
    ref = reference.get('ema')
    if ref is not None:
        for (col, metric) in enumerate([ 'jer_cal', 'l1_sig', 'mse_sig' ]):
            if pd.isna(ref.get(metric)):
                continue
            axes[col].axhline(ref[metric], color = 'k', ls = '--', lw = 1)
            axes[col].annotate('published, 800k updates',
                (0.0, ref[metric]), xycoords = ('axes fraction', 'data'),
                fontsize = 7, va = 'top')
    ax.set_ylim(3.45, 5.5)
    axes[1].set_ylim(0.02, 0.2)
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')

    for (col, label) in enumerate([ 'val jer_cal, GeV',
                                    'val l1_sig (mean |error| per tower), GeV',
                                    'val mse_sig (mean error^2 per tower), GeV^2' ]):
        axes[col].set_xscale('log')
        axes[col].set_xlabel('training time on one RTX A6000, hours')
        axes[col].set_ylabel(label)
        axes[col].grid(alpha = 0.3, which = 'both')
    axes[0].legend(fontsize = 7, loc = 'upper right',
                   bbox_to_anchor = (1.0, 0.93))

    ax = axes[3]
    t = table[(table.net == 'ema') & table.jewel_jer_cal.notna()]
    for (arm, x) in t.groupby('arm'):
        (color, _) = ARM_STYLE.get(arm, (None, '-'))
        ax.scatter(x.best_jer_cal, x.jewel_jer_cal, color = color, s = 30,
                   label = arm, zorder = 3)
    lo, hi = 3.4, 4.7
    ax.plot([ lo, hi ], [ lo, hi ], color = 'grey', lw = 0.8, ls = ':')
    ax.scatter([ 3.588 ], [ 3.992 ], marker = '*', s = 120, color = 'k',
               label = 'published, 800k updates', zorder = 3)
    ax.set_xlim(3.5, 3.95)
    ax.set_ylim(3.45, 4.6)
    ax.set_xlabel('val jer_cal at the val-selected checkpoint, GeV')
    ax.set_ylabel('JEWEL jer_cal (out of distribution), GeV')
    ax.grid(alpha = 0.3)
    ax.legend(fontsize = 7, loc = 'upper left')

    fig.tight_layout()
    fig.savefig(out, dpi = 120)

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
        runs[config['label']] = (run_dir.rstrip('/'), config['method'], None)

        if cmdargs.extra_samples and (config['method'] == 'condcfm'):
            (v, _, _) = flow_curves(run_dir.rstrip('/'), cmdargs.extra_samples)
            if len(v):
                curves.append(v)
                label = v.run.iloc[0]
                summaries[label] = summary
                runs[label] = (
                    run_dir.rstrip('/'), 'condcfm', cmdargs.extra_samples
                )

    for run_dir in cmdargs.baseline:
        (v, label) = baseline_curves(run_dir.rstrip('/'))
        curves.append(v)
        runs[label] = (run_dir.rstrip('/'), None, None)

    cols   = [ 'family', 'arm', 'seed', 'run', 'net', 'step', 'hours',
               'jer_cal', 'l1_sig', 'l1_bkg', 'mse_sig', 'mse_bkg', 'jes',
               'bias' ]
    curves = pd.concat([ c.reindex(columns = cols) for c in curves ],
                       ignore_index = True)

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

    matched = matched_table(curves)
    matched.to_csv(f'{cmdargs.out}_matched.csv', index = False)

    with pd.option_context('display.width', 250):
        print()
        print(arms.round(3).to_string(index = False))
        print('\nval jer_cal (EMA) at matched training times, mean +- half'
              ' range of the seeds (n if fewer seeds reached it):')
        print(matched.fillna('').to_string(index = False))

    plot(curves, reference, f'{cmdargs.out}.png')
    plot_report(curves, table, reference, f'{cmdargs.out}_report.png')
    plot_nfe(runs, f'{cmdargs.out}_nfe.png')
    print(f'wrote {cmdargs.out}.csv, {cmdargs.out}_arms.csv,'
          f' {cmdargs.out}_curves.csv, {cmdargs.out}.png, {cmdargs.out}_nfe.png')

if __name__ == '__main__':
    main()
