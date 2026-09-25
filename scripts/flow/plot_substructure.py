#!/usr/bin/env python
"""Distributions of the jet substructure observables of substructure.py:
truth, truth with a 0.5 GeV tower threshold, and the extractions, on the
PYTHIA val and the JEWEL test jets.

    plot_substructure.py [--out docs/flow/substructure.png]
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

SHOW = [
    ('truth',                      'k',       '-',  2.0),
    ('truth, towers > 0.5 GeV',    'k',       ':',  1.5),
    ('uvcgan-s published',         '#7f7f7f', '-',  1.3),
    ('OT-CFM coarse',              '#2ca02c', '--', 1.3),
    ('OT-CFM seeded8_zero_thr0.5', '#2ca02c', '-',  1.6),
    ('cond. CFM, 1 sample',        '#ff7f0e', '--', 1.3),
    ('cond. CFM, mean of 16',      '#d62728', '-',  1.3),
]
OBS = [ ('et', 'cone E_T, GeV', (0, 70)), ('mass', 'mass, GeV', (0, 20)),
        ('girth', 'girth', (0, 0.25)), ('ptd', 'p_T^D', (0.2, 1.0)),
        ('zlead', 'leading tower fraction', (0, 1)),
        ('zg', 'soft drop z_g', (0.1, 0.5)), ('rg', 'soft drop R_g', (0, 0.4)) ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Plot substructure')
    parser.add_argument('--data', default = os.path.join(
        os.environ.get('UVCGAN_S_OUTDIR', 'outdir'), 'sphenix', 'flow',
        'substructure_distributions.npz'))
    parser.add_argument('--out', default = 'docs/flow/substructure.png')
    return parser.parse_args()

def main():
    cmdargs = parse_cmdargs()
    d = np.load(cmdargs.data)

    (fig, axes) = plt.subplots(2, len(OBS), figsize = (3.1 * len(OBS), 6.2))
    for (row, truth_set) in enumerate([ 'val', 'jewel' ]):
        for (col, (q, label, rng)) in enumerate(OBS):
            ax = axes[row][col]
            bins = np.linspace(*rng, 31)
            for (model, color, ls, lw) in SHOW:
                x = d[f'{truth_set}|{model}|{q}']
                x = x[np.isfinite(x)]
                ax.hist(x, bins = bins, density = True, histtype = 'step',
                        color = color, ls = ls, lw = lw, label = model)
            ax.set_xlabel(label, fontsize = 8)
            ax.tick_params(labelsize = 7)
            if col == 0:
                ax.set_ylabel(
                    'PYTHIA val' if truth_set == 'val' else 'JEWEL test',
                    fontsize = 9)
    axes[0][0].legend(fontsize = 6, loc = 'upper right')
    fig.suptitle('R = 0.4 cone around the true leading-jet axis, towers as'
                 ' constituents (10k events each)', fontsize = 9)
    fig.tight_layout()
    fig.savefig(cmdargs.out, dpi = 110)
    print(f'wrote {cmdargs.out}')

if __name__ == '__main__':
    main()
