#!/usr/bin/env python
"""Validation curves of the alpha-DSBM closure test (FLOW_NOTES.md,
"alpha-DSBM closure test"): each checkpoint's scores on the first 5000
validation pairs (the selection files of dsbm_eval.py --select and
closure_eval.py --select) against the GPU time spent on the run, the
pretraining included for the arms that start from it.

    dsbm_curves.py LABEL=RUN[:TAG] ... --out docs/flow/dsbm/closure_curves

TAG is the selection file's tag (default: _sde30 for a bridge run,
_32midpoint for an ODE run).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd

import fm_common as fc
from dsbm_eval import colour

plt.rcParams.update({ 'font.size' : 8, 'legend.fontsize' : 6.5 })

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'alpha-DSBM validation curves')
    parser.add_argument('runs', nargs = '+', help = 'LABEL=RUN[:TAG]')
    parser.add_argument('--identity', type = float, default = None,
        help = 'shape EMD of the identity on the same pairs (a line)')
    parser.add_argument('--out', default = 'docs/flow/dsbm/closure_curves')
    return parser.parse_args()

def load(run, tag):
    path = os.path.join(fc.out_root(), run)
    with open(os.path.join(path, 'config.json'), encoding = 'utf-8') as f:
        config = json.load(f)
    bridge = 'stage' in config
    tag = tag if tag is not None else ('_sde30' if bridge else '_32midpoint')
    df = pd.read_csv(os.path.join(path, 'evals', f'closure_val{tag}.csv'))
    if 'total_train_time' not in df:
        df['total_train_time'] = df['train_time']
    return (df.sort_values('step'), bridge)

def main():
    cmdargs = parse_cmdargs()
    (fig, axes) = plt.subplots(1, 3, figsize = (10, 2.9))
    rows = []
    for item in cmdargs.runs:
        (label, spec) = item.split('=', 1)
        (run, tag) = (spec.split(':', 1) + [ None ])[:2]
        (df, bridge) = load(run, tag)
        x = df.total_train_time / 3600
        style = dict(color = colour(label), marker = 'o', ms = 2.5, lw = 1.2,
                     ls = '-' if bridge else '--', label = label)
        axes[0].plot(x, df.shape_emd, **style)
        axes[1].plot(x, df.emd_gev, **style)
        axes[2].plot(x, df.response_mean, **style)
        sel = df[df.selected]
        for (ax, q) in zip(axes, [ 'shape_emd', 'emd_gev', 'response_mean' ]):
            ax.plot(sel.total_train_time / 3600, sel[q], marker = '*', ms = 8,
                    color = colour(label), ls = 'none')
        rows.append({ 'arm' : label, 'run' : run, 'checkpoints' : len(df),
                      'selected_step' : int(sel.step.iloc[0]),
                      'selected_gpu_h' : float(sel.total_train_time.iloc[0] / 3600),
                      'selected_val_shape_emd' : float(sel.shape_emd.iloc[0]),
                      'selected_val_emd_gev' : float(sel.emd_gev.iloc[0]),
                      'last_val_shape_emd' : float(df.shape_emd.iloc[-1]),
                      'min_val_shape_emd' : float(df.shape_emd.min()) })
    if cmdargs.identity is not None:
        axes[0].axhline(cmdargs.identity, color = 'k', lw = 0.8, ls = ':',
                        label = 'identity F(J) = J')
    axes[2].axhline(0.8, color = 'k', lw = 0.8, ls = ':', label = 'truth 0.8')
    for (ax, title) in zip(axes, [ 'shape EMD to T(J) (unit-energy jets)',
                                   'EMD to T(J), GeV (selection metric)',
                                   'energy response E_out / E_in' ]):
        ax.set_title(title, fontsize = 7.5)
        ax.set_xlabel('GPU hours (A6000), pretraining and rollouts included')
        ax.grid(alpha = 0.3)
    axes[0].set_yscale('log')
    axes[0].legend(fontsize = 5.5)
    fig.suptitle('Closure test, 5000 validation pairs: one output per jet (bridges: one SDE'
                 ' sample, fixed seed; OT-CFM: ODE, 32 midpoint evaluations); * = selected',
                 fontsize = 7.5)
    fig.tight_layout()
    fig.savefig(f'{cmdargs.out}.png', dpi = 150)
    table = pd.DataFrame(rows)
    table.to_csv(f'{cmdargs.out}.csv', index = False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(table.round(4).to_string(index = False))

if __name__ == '__main__':
    main()
