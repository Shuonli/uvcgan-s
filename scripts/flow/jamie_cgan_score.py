#!/usr/bin/env python
"""Scores of the toy CycleGAN's outputs on the girth-rule test jets
(FLOW_NOTES.md, "CycleGAN on the girth-rule toy"), for runs the flow report
cannot score fairly: a CycleGAN trained on an N x N window around the jet
(jamie_cgan.py --crop N) can fill only that window, so the truth and the
input are restricted to it before anything is compared.

    jamie_cgan_score.py --crop 16 --tag cgan16 \
        --models "EMA 8 h=E3g-CG16_s0_8h,no EMA 8 h=E3g-CG16_s0_raw_8h"

Writes, under docs/flow/jamie_otfm/ex3g/: ex3g_TAG.csv (means, W1 of six
marginals to the truth in units of its spread, the girth-rule correlations,
the fractional cone loss in fifths of the input girth), and the figures
ex3g_marginals_TAG.png and ex3g_girth_scatter_TAG.png. `ring` here is 0.4 <=
R < 1.0 inside the window.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import jamie_cgan as jc
import jamie_figs as jf
import jamie_report as jr

PANELS = [ ('total', 'total energy (GeV)', (0, 90)), ('cone', 'cone energy, R < 0.4 (GeV)', (0, 80)),
           ('ring', 'ring energy, 0.4 <= R < 1.0 (GeV)', (0, 20)), ('girth', 'cone girth', (0, 0.3)),
           ('mass', 'cone mass (GeV)', (0, 8)), ('ptd', 'cone p_T^D', (0.2, 1.0)) ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Score the toy CycleGAN outputs')
    parser.add_argument('--models', required = True, help = 'label=output directory, comma separated')
    parser.add_argument('--crop', type = int, default = 0)
    parser.add_argument('--tag', required = True)
    parser.add_argument('--set', default = 't3g_pair')
    parser.add_argument('--out', default = 'docs/flow/jamie_otfm/ex3g')
    return parser.parse_args()

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    plt = jf.setup()
    d = jr.cache(cmdargs.set)
    ax = jr.axes(d)
    dr = jr.dr_of(ax)
    mask = jc.window_mask(d['eta0'], d['phi0'], cmdargs.crop) if cmdargs.crop else None
    inside = (lambda x: x * mask) if cmdargs.crop else (lambda x: x)
    def read(x):
        o = jr.obs(x, ax, sub = False)
        return { 'total' : np.asarray(o['total']), 'cone' : jr.region_sum(x, dr, 0, 0.4),
                 'ring' : jr.region_sum(x, dr, 0.4, 1.0), 'girth' : np.asarray(o['girth']),
                 'mass' : np.asarray(o['mass']), 'ptd' : np.asarray(o['ptd']) }
    (truth, inp) = (read(inside(d['med'])), read(inside(d['jet'])))
    if cmdargs.crop:
        full = d['med'].sum((1, 2))
        print(f'window {cmdargs.crop}: holds {(truth["total"] / full).mean():.4f} of the true quenched '
              f'energy, {(inp["total"] / d["jet"].sum((1, 2))).mean():.4f} of the input', flush = True)
    g_in = inp['girth']
    edges = np.quantile(g_in, np.linspace(0, 1, 6))
    fifths = [ (g_in >= lo) & (g_in <= hi) for (lo, hi) in zip(edges[:-1], edges[1:]) ]
    true_loss = 1 - truth['cone'] / inp['cone']
    def row(name, o):
        loss = 1 - o['cone'] / inp['cone']
        ok = np.isfinite(loss)
        r = { 'model' : name }
        for q in o:
            r[f'mean_{q}'] = float(np.nanmean(o[q]))
            (a, b) = (o[q][np.isfinite(o[q])], truth[q][np.isfinite(truth[q])])
            r[f'w1_{q}'] = wasserstein_distance(a, b) / b.std()
        r['corr_loss_input_girth'] = float(np.corrcoef(loss[ok], g_in[ok])[0, 1])
        r['corr_loss_input_energy'] = float(np.corrcoef(loss[ok], inp['cone'][ok])[0, 1])
        r['corr_loss_true_loss'] = float(np.corrcoef(loss[ok], true_loss[ok])[0, 1])
        for (k, b) in enumerate(fifths):
            r[f'loss_girth_fifth{k + 1}'] = float(loss[b & ok].mean())
        r['loss_mean'] = float(loss[ok].mean())
        r['loss_sd'] = float(loss[ok].std())
        r['w1_loss'] = wasserstein_distance(loss[ok], true_loss)
        r['w1_loss_in_fifths'] = float(np.mean([ wasserstein_distance(loss[b & ok], true_loss[b])
                                                 for b in fifths ]))
        return r
    models = [ (lab, dirn, jr.output(dirn, f'{cmdargs.set}_vac')) for (lab, dirn) in jr.specs(cmdargs.models) ]
    models = [ (lab, dirn, read(x)) for (lab, dirn, x) in models if x is not None ]
    rows = [ row('truth (one quenching)', truth), row('input (vacuum)', inp) ]
    rows += [ row(lab, o) for (lab, _, o) in models ]
    df = pd.DataFrame(rows)
    os.makedirs(cmdargs.out, exist_ok = True)
    df.to_csv(os.path.join(cmdargs.out, f'ex3g_{cmdargs.tag}.csv'), index = False)
    with pd.option_context('display.width', 250, 'display.max_columns', 80):
        print(df.set_index('model').T.round(3).to_string(), flush = True)

    where = f' inside the {cmdargs.crop} x {cmdargs.crop} window' if cmdargs.crop else ''
    (fig, axs) = plt.subplots(2, 3, figsize = (7.6, 4.6))
    for (a, (q, lab, rng)) in zip(axs.flat, PANELS):
        bins = np.linspace(*rng, 51)
        a.hist(truth[q], bins, histtype = 'step', density = True, color = jf.INK, lw = 1.7,
               label = 'true quenched')
        a.hist(inp[q], bins, histtype = 'step', density = True, color = jf.MUTED, lw = 1.0,
               label = 'input (vacuum)')
        ws = []
        for (k, (name, _, o)) in enumerate(models):
            a.hist(o[q], bins, histtype = 'step', density = True, lw = 1.2,
                   color = jf.PALETTE[k % len(jf.PALETTE)], label = name)
            ws.append(f"{df.loc[df['model'] == name, f'w1_{q}'].iloc[0]:.2f}")
        a.set_title(f'{lab}\nW1 to truth: ' + ', '.join(ws), fontsize = 6.5)
        a.set_yticks([])
        a.set_xlim(*rng)
    axs[0][0].legend(fontsize = 5.5, frameon = False)
    fig.suptitle(f'Girth-rule toy{where}: 20k held-out jets, one output each. W1 in units of the '
                 "truth's spread, models in the legend's order.", fontsize = 6.5)
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    fig.savefig(os.path.join(cmdargs.out, f'ex3g_marginals_{cmdargs.tag}.png'))
    plt.close(fig)
    jf.fig_girth([ (lab, dirn) for (lab, dirn, _) in models ], cmdargs.out, plt, mask = mask,
                 name = f'ex3g_girth_scatter_{cmdargs.tag}.png', note = where)

if __name__ == '__main__':
    main()
