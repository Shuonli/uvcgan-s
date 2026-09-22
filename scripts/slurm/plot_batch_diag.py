#!/usr/bin/env python
"""Plot the batch size diagnostic: quality against updates and samples.

If the curves of different batch sizes fall on top of each other when
drawn against the number of samples, progress is limited by data and a
larger batch (or more GPUs) shortens training. If they fall on top of each
other against the number of updates, progress is limited by the updates
themselves and a larger batch only wastes computation.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [ 'idt_aa_a0', 'idt_aa_a1', 'cycle_a1', 'cycle_b' ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Plot batch diagnostic')
    parser.add_argument('root', nargs = '?',
        default = os.path.join(os.environ.get('UVCGAN_S_OUTDIR', 'outdir'),
                               'sphenix', 'diag'))
    parser.add_argument('--job', default = None, help = 'job id suffix')
    parser.add_argument('--out', default = 'batch_diag.png')
    parser.add_argument('--smooth', type = int, default = 5,
        help = 'rolling median window over history rows')
    parser.add_argument('--target', type = float, default = None,
        help = 'metric value to report updates / samples needed for')
    return parser.parse_args()

def load(root, job):
    runs = {}

    for d in sorted(glob.glob(os.path.join(root, 'model_*'))):
        hist = os.path.join(d, 'history.csv')
        if not os.path.exists(hist):
            continue

        with open(os.path.join(d, 'label'), encoding = 'utf-8') as f:
            label = f.read().strip()

        if job is not None and not label.endswith(job):
            continue

        with open(os.path.join(d, 'config.json'), encoding = 'utf-8') as f:
            config = json.load(f)

        h = pd.read_csv(hist)
        h['updates'] = h.epoch * config['steps_per_epoch']
        h['samples'] = h.updates * config['batch_size']

        name = label.replace('diag_', '')
        if job is not None:
            name = name[: -len(job) - 1]

        runs[name] = (h, config['batch_size'])

    return runs

def crossing(h, metric, target, axis):
    """First value of `axis` at which `metric` drops to `target`."""
    below = h[h[metric] <= target]
    return None if len(below) == 0 else int(below[axis].iloc[0])

def main():
    cmdargs = parse_cmdargs()
    runs    = load(cmdargs.root, cmdargs.job)

    if not runs:
        print(f"no runs found under '{cmdargs.root}'")
        return

    order  = sorted(runs, key = lambda k: (runs[k][1], k))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(order)))

    fig, axes = plt.subplots(
        2, len(METRICS), figsize = (4.2 * len(METRICS), 7.5), sharex = 'row'
    )

    for (col, metric) in enumerate(METRICS):
        for (axis, row) in (('updates', 0), ('samples', 1)):
            ax = axes[row][col]

            for (name, color) in zip(order, colors):
                h, _ = runs[name]
                if metric not in h:
                    continue

                y = h[metric]
                if cmdargs.smooth > 1:
                    y = y.rolling(cmdargs.smooth, min_periods = 1).median()

                ax.plot(h[axis], y, label = name, color = color, lw = 1.6)

            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel(axis); ax.grid(alpha = 0.3, which = 'both')
            if col == 0:
                ax.set_ylabel(f'{metric}\n(vs {axis})')
            else:
                ax.set_ylabel(metric)
            if row == 0 and col == 0:
                ax.legend(fontsize = 7)

    fig.suptitle(
        'top: same x = same number of updates    '
        'bottom: same x = same number of samples seen'
    )
    fig.tight_layout()
    fig.savefig(cmdargs.out, dpi = 110)
    print(f"wrote {cmdargs.out}")

    print(f"\n{'run':18s} {'batch':>6s} {'rows':>5s} {'updates':>9s} {'samples':>10s}"
          f" {'final idt_aa_a1':>15s}")
    for name in order:
        h, b = runs[name]
        print(f"{name:18s} {b:6d} {len(h):5d} {h.updates.iloc[-1]:9d}"
              f" {h.samples.iloc[-1]:10d} {h.idt_aa_a1.iloc[-1]:15.4f}")

    if cmdargs.target is not None:
        print(f"\nto reach idt_aa_a1 <= {cmdargs.target}:")
        print(f"  {'run':18s} {'updates':>9s} {'samples':>10s}")
        for name in order:
            h, _ = runs[name]
            u = crossing(h, 'idt_aa_a1', cmdargs.target, 'updates')
            s = crossing(h, 'idt_aa_a1', cmdargs.target, 'samples')
            us = 'not reached' if u is None else f'{u}'
            ss = 'not reached' if s is None else f'{s}'
            print(f"  {name:18s} {us:>9s} {ss:>10s}")

if __name__ == '__main__':
    main()
