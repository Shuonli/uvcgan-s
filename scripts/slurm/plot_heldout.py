#!/usr/bin/env python
"""Held-out quality of the runs of diag_heldout.sbatch against time.

Each run scores its generators on val events at the end of every epoch
(`val_truth_history.csv`, c.f. eval_val_truth.py). The time axis is the
training time proper, the sum of `epoch_time`, which leaves out the
scoring itself.

    plot_heldout.py --job 20016 [--out heldout.png]
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANELS = [
    ('raw', 'l1_sig'), ('raw', 'jer'),
    ('ema', 'l1_sig'), ('ema', 'jer'),
]
HOURS = [ 1, 2, 3, 4, 5, 6, 7 ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Plot held-out diagnostic')
    parser.add_argument('--root', default = os.path.join(
        os.environ.get('UVCGAN_S_OUTDIR', 'outdir'), 'sphenix', 'heldout'))
    parser.add_argument('--job', required = True, help = 'job id suffix')
    parser.add_argument('--out', default = 'heldout.png')
    parser.add_argument('--smooth', type = int, default = 10,
        help = 'rolling mean window, epochs')
    return parser.parse_args()

def load(root, job):
    runs = {}

    for d in sorted(glob.glob(os.path.join(root, f'model_*_{job}'))):
        scores = os.path.join(d, 'val_truth_history.csv')
        if not os.path.exists(scores):
            continue

        with open(os.path.join(d, 'label'), encoding = 'utf-8') as f:
            label = f.read().strip()

        with open(os.path.join(d, 'config.json'), encoding = 'utf-8') as f:
            config = json.load(f)

        h = pd.read_csv(os.path.join(d, 'history.csv'))
        h['hours']   = h.epoch_time.cumsum() / 3600
        h['updates'] = h.epoch * config['steps_per_epoch']

        v = pd.read_csv(scores)
        v = v.merge(h[[ 'epoch', 'hours', 'updates' ]], on = 'epoch')

        m = re.match(r'ho_(.*)_s(\d+)_\d+$', label)
        runs[label] = {
            'arm'     : m.group(1),
            'seed'    : int(m.group(2)),
            'history' : h,
            'scores'  : v,
        }

    return runs

def smoothed(v, net, metric, window):
    x = v[v.net == net].sort_values('epoch')
    return (x, x[metric].rolling(window, min_periods = 1).mean())

def value_at(x, y, axis, point):
    """Smoothed value at `point` of `axis`, None if not reached."""
    if len(x) == 0 or x[axis].iloc[-1] < point:
        return None

    return float(np.interp(point, x[axis].values, y.values))

def main():
    cmdargs = parse_cmdargs()
    runs    = load(cmdargs.root, cmdargs.job)

    if not runs:
        print(f"no runs of job {cmdargs.job} under '{cmdargs.root}'")
        return

    arms   = sorted({ r['arm'] for r in runs.values() })
    colors = dict(zip(arms, plt.cm.tab10(np.arange(len(arms)))))
    styles = { 0 : '-', 1 : '--', 2 : ':' }

    fig, axes = plt.subplots(
        2, len(PANELS), figsize = (4.3 * len(PANELS), 7.5)
    )

    for (col, (net, metric)) in enumerate(PANELS):
        for (row, axis) in enumerate([ 'hours', 'updates' ]):
            ax = axes[row][col]

            for (label, r) in sorted(runs.items()):
                (x, y) = smoothed(r['scores'], net, metric, cmdargs.smooth)
                ax.plot(
                    x[axis], y, color = colors[r['arm']],
                    ls = styles.get(r['seed'], '-.'), lw = 1.4,
                    label = f"{r['arm']} s{r['seed']}"
                )

            ax.set_xlabel('training hours' if axis == 'hours' else axis)
            ax.set_ylabel(f'{net} {metric}')
            ax.set_yscale('log')
            ax.grid(alpha = 0.3, which = 'both')

            if (row == 0) and (col == 0):
                ax.legend(fontsize = 7)

    fig.suptitle(
        f'held-out scores on val truth, rolling mean of {cmdargs.smooth}'
        f' epochs (job {cmdargs.job})'
    )
    fig.tight_layout()
    fig.savefig(cmdargs.out, dpi = 110)
    print(f"wrote {cmdargs.out}")

    for (net, metric) in PANELS:
        print(f"\n{net} {metric} after N training hours"
              f" (rolling mean of {cmdargs.smooth} epochs):")
        print(f"  {'run':28s}" + ''.join(f'{h:>9d}' for h in HOURS))

        per_arm = {}

        for (label, r) in sorted(runs.items()):
            (x, y) = smoothed(r['scores'], net, metric, cmdargs.smooth)
            cells  = [ value_at(x, y, 'hours', h) for h in HOURS ]
            per_arm.setdefault(r['arm'], []).append(cells)

            print(f"  {r['arm'] + ' s' + str(r['seed']):28s}" + ''.join(
                '        -' if c is None else f'{c:9.4f}' for c in cells
            ))

        # mean over seeds, and half their range as a measure of the spread
        for (arm, seeds) in sorted(per_arm.items()):
            means  = []
            spread = []

            for cells in zip(*seeds):
                if (len(cells) < 2) or any(c is None for c in cells):
                    means.append(None)
                    spread.append(None)
                else:
                    means.append(np.mean(cells))
                    spread.append((max(cells) - min(cells)) / 2)

            print(f"  {arm + ' mean':28s}" + ''.join(
                '        -' if m is None else f'{m:9.4f}' for m in means
            ))
            print(f"  {arm + ' +-':28s}" + ''.join(
                '        -' if s is None else f'{s:9.4f}' for s in spread
            ))

    print(f"\n{'run':28s} {'epochs':>6s} {'updates':>8s} {'hours':>6s}"
          f" {'s/epoch':>8s} {'eval s':>7s}")
    for (label, r) in sorted(runs.items()):
        h = r['history']
        v = r['scores']
        print(f"{r['arm'] + ' s' + str(r['seed']):28s} {len(h):6d}"
              f" {h.updates.iloc[-1]:8d} {h.hours.iloc[-1]:6.2f}"
              f" {h.epoch_time.median():8.1f}"
              f" {v.groupby('epoch').eval_time.sum().median():7.1f}")

if __name__ == '__main__':
    main()
