#!/usr/bin/env python
"""Compare runs of the short recipe on held-out truth, by updates and hours.

Reads the checkpoint scores of each run (`evals/{val,jewel}_truth.csv`, c.f.
eval_val_truth.py) and its `history.csv` for the training time, groups the
runs by batch size and averages over seeds. The published model, if given,
is the reference line.

    compare_runs.py RUN_DIR [RUN_DIR ...] [--reference PUBLISHED_DIR]
                    [--out compare.png]
"""

import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRUTHS  = [ 'val', 'jewel' ]
NETS    = [ 'ema', 'raw' ]
METRICS = [ 'jer_cal', 'l1_sig' ]
HOURS   = [ 4, 8, 12, 16, 20, 24 ]
UPDATES = [ 20000, 40000, 60000, 80000, 100000, 120000, 140000 ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Compare runs on truth')
    parser.add_argument('runs', nargs = '+', help = 'run directories')
    parser.add_argument('--reference', default = None,
        help = 'model directory of the published model')
    parser.add_argument('--out', default = 'compare.png')
    return parser.parse_args()

def arm_and_seed(label):
    m = re.match(r'base_b(\d+)_lr([^_]+)(?:_s(\d+))?$', label)
    if m is None:
        raise ValueError(f"unexpected label '{label}'")

    return (f'batch {m.group(1)}, {m.group(2)}', int(m.group(3) or 0))

def load_run(path):
    with open(os.path.join(path, 'label'), encoding = 'utf-8') as f:
        label = f.read().strip()

    (arm, seed) = arm_and_seed(label)

    h = pd.read_csv(os.path.join(path, 'history.csv'))
    h['hours'] = h.epoch_time.cumsum() / 3600

    scores = {}
    for truth in TRUTHS:
        csv = os.path.join(path, 'evals', f'{truth}_truth.csv')
        if not os.path.exists(csv):
            continue

        v = pd.read_csv(csv)
        v = v[v.epoch > 0].merge(h[[ 'epoch', 'hours' ]], on = 'epoch')
        scores[truth] = v

    return { 'label' : label, 'arm' : arm, 'seed' : seed, 'scores' : scores }

def load_reference(path):
    result = {}

    for truth in TRUTHS:
        csv = os.path.join(path, 'evals', f'{truth}_truth.csv')
        if os.path.exists(csv):
            result[truth] = pd.read_csv(csv)

    return result

def series(run, truth, net, metric):
    v = run['scores'].get(truth)
    if v is None:
        return None

    return v[v.net == net].sort_values('epoch')

def value_at(x, axis, metric, point):
    if (x is None) or (len(x) == 0):
        return None

    if (x[axis].iloc[0] > point) or (x[axis].iloc[-1] < point):
        return None

    return float(np.interp(point, x[axis].values, x[metric].values))

def print_table(runs, truth, net, metric, axis, points, reference):
    unit  = 'h' if axis == 'hours' else 'k'
    scale = 1 if axis == 'hours' else 1000
    width = 13 if metric == 'jer_cal' else 17

    print(f"\n{truth} {net} {metric} by {axis}"
          f" (mean of the seeds +- half their range):")
    print(f"  {'':24s}" + ''.join(
        f'{str(p // scale) + unit:>{width}s}' for p in points
    ))

    for arm in sorted({ r['arm'] for r in runs }):
        members = [ r for r in runs if r['arm'] == arm ]
        cells   = []

        for p in points:
            vals = [
                value_at(series(r, truth, net, metric), axis, metric, p)
                    for r in members
            ]
            vals = [ v for v in vals if v is not None ]

            if not vals:
                cells.append('-')
                continue

            spread = (max(vals) - min(vals)) / 2
            digits = 2 if metric == 'jer_cal' else 4
            cells.append(
                f'{np.mean(vals):.{digits}f}+-{spread:.{digits}f}'
                + ('' if len(vals) == len(members) else f'({len(vals)})')
            )

        name = f'{arm} ({len(members)})'
        print(f"  {name:24s}" + ''.join(f'{c:>{width}s}' for c in cells))

    ref = reference.get(truth)
    if (ref is not None) and len(ref[ref.net == net]):
        value = ref[ref.net == net][metric].iloc[0]
        print(f"  {'published, 800k updates':24s} {value:.4f}")

def plot(runs, reference, out):
    arms   = sorted({ r['arm'] for r in runs })
    colors = dict(zip(arms, plt.cm.tab10(np.arange(len(arms)))))

    fig, axes = plt.subplots(2, 2 * len(TRUTHS), figsize = (18, 8))

    for (col, (truth, net)) in enumerate(
        [ (t, n) for t in TRUTHS for n in NETS ]
    ):
        for (row, axis) in enumerate([ 'hours', 'updates' ]):
            ax = axes[row][col]

            for r in runs:
                x = series(r, truth, net, 'jer_cal')
                if x is None:
                    continue

                ax.plot(
                    x[axis], x['jer_cal'], color = colors[r['arm']],
                    marker = 'o', ms = 3, lw = 1.2,
                    label = f"{r['arm']} s{r['seed']}"
                )

            ref = reference.get(truth)
            if ref is not None and len(ref[ref.net == net]):
                ax.axhline(
                    ref[ref.net == net].jer_cal.iloc[0], color = 'k',
                    ls = '--', lw = 1, label = 'published, 800k updates'
                )

            ax.set_xlabel('training hours' if axis == 'hours' else axis)
            ax.set_ylabel(f'{truth} {net} jer_cal, GeV')
            ax.set_ylim(top = min(ax.get_ylim()[1], 5.5))
            ax.grid(alpha = 0.3)

            if (row == 0) and (col == 0):
                ax.legend(fontsize = 7)

    fig.tight_layout()
    fig.savefig(out, dpi = 110)
    print(f'wrote {out}')

def main():
    cmdargs   = parse_cmdargs()
    runs      = [ load_run(p.rstrip('/')) for p in cmdargs.runs ]
    reference = {}

    if cmdargs.reference is not None:
        reference = load_reference(cmdargs.reference)

    for r in runs:
        print(f"{r['label']:28s} {r['arm']:18s} seed {r['seed']}"
              f"  checkpoints: " + ', '.join(
                  f'{t} {len(v) // 2}' for (t, v) in r['scores'].items()))

    for metric in METRICS:
        for truth in TRUTHS:
            for net in NETS:
                for (axis, points) in [
                    ('updates', UPDATES), ('hours', HOURS)
                ]:
                    print_table(
                        runs, truth, net, metric, axis, points, reference
                    )

    plot(runs, reference, cmdargs.out)

if __name__ == '__main__':
    main()
