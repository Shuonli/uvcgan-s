#!/usr/bin/env python
"""Validation curves of the paired noisy-interpolant pilot (FLOW_NOTES.md,
"Paired noisy-interpolant pilot"): the per-checkpoint val cone scores of
fm_eval.py (evals/val_truth.csv: 20k val events, EMA; 4 Euler steps at every
checkpoint, 32 and 64 midpoint NFE at the last) of each run against its
updates, with the benchmark's paired seed 0 (same seed, straight path) as
the reproducibility reference. Training behaviour only: the comparison is at
the final update.

    noisy_curves.py LABEL=RUN ... [--out docs/flow/bench/noisy/noisy_curves]
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd

import fm_common as fc

plt.rcParams.update({ 'font.size' : 8, 'legend.fontsize' : 6.5 })
COLOURS = [ '#ff7f0e', '#9467bd', '#7f7f7f' ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Pilot validation curves')
    parser.add_argument('runs', nargs = '+', help = 'LABEL=RUN')
    parser.add_argument('--out', default = 'docs/flow/bench/noisy/noisy_curves')
    return parser.parse_args()

def main():
    cmdargs = parse_cmdargs()
    (fig, axes) = plt.subplots(1, 3, figsize = (10, 2.9))
    rows = []
    for (k, item) in enumerate(cmdargs.runs):
        (label, run) = item.split('=', 1)
        df = pd.read_csv(os.path.join(fc.out_root(), run, 'evals', 'val_truth.csv'))
        df = df[df.net == 'ema'].sort_values('step')
        c = COLOURS[k % len(COLOURS)]
        ls = '--' if 'benchmark' in label else '-'
        for (ax, q) in zip(axes, [ 'jer_cal', 'jes', 'l1_sig' ]):
            e4 = df[(df.nfe == 4) & (df.solver == 'euler')]
            ax.plot(e4.step, e4[q], color = c, ls = ls, marker = 'o', ms = 2.5,
                    label = f'{label}, 4 Euler')
            for (nfe, mk) in [ (32, 's'), (64, '^') ]:
                m = df[(df.nfe == nfe) & (df.solver == 'midpoint')]
                ax.plot(m.step, m[q], color = c, ls = 'none', marker = mk, ms = 5,
                        label = f'{label}, midpoint {nfe}' if q == 'jer_cal' else None)
        for (_, r) in df.iterrows():
            rows.append({ 'label' : label, 'run' : run, 'step' : r.step, 'nfe' : r.nfe,
                          'solver' : r.solver, 'jer_cal' : r.jer_cal, 'jes' : r.jes,
                          'l1_sig' : r.l1_sig, 'train_time_min' : r.train_time / 60 })
    for (ax, title) in zip(axes, [ 'cone resolution after calibration (jer_cal), GeV',
                                   'cone energy scale (jes)', 'per-tower MAE of S_hat, GeV' ]):
        ax.set_title(title, fontsize = 7.5)
        ax.set_xlabel('update')
        ax.grid(alpha = 0.3)
    axes[0].legend(fontsize = 5.5)
    fig.suptitle('Paired pilot, 20k val events, EMA network: 4 Euler steps at every checkpoint,'
                 ' midpoint 32 / 64 NFE at the final update', fontsize = 7.5)
    fig.tight_layout()
    fig.savefig(f'{cmdargs.out}.png', dpi = 150)
    pd.DataFrame(rows).to_csv(f'{cmdargs.out}.csv', index = False)
    with pd.option_context('display.width', 200):
        print(pd.DataFrame(rows).round(4).to_string(index = False))

if __name__ == '__main__':
    main()
