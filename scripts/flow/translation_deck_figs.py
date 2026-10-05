#!/usr/bin/env python
"""Figures of the short PYTHIA -> JEWEL translation deck
(docs/flow/translation/slides/translation_deck.tex), drawn at the size they
are printed, from the outputs translation_eval.py --generate saved and the
held-out test sets. The numbers on the slides come from
translation_report.py's tables; these figures show the same samples.

Series: JEWEL test (the target, black with a light wash), the PYTHIA input
(grey, dotted), OT-CFM (blue) and alpha-DSBM (orange, dashed). The
blue/orange pair passes the colour-vision-deficiency check (dataviz
validator: deutan/protan Delta E 24.7); the green/red of the earlier
figures does not (3.9).

    translation_deck_figs.py [--out docs/flow/translation]
"""

import argparse
import os

import numpy as np
import torch

import translation_eval as te
from closure_data import OFFSET
from substructure import Geometry
from translation_report import display_indices

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
AXIS, GRID = '#c3c2b7', '#e1e0d9'
BLUE, ORANGE = '#2a78d6', '#eb6834'
STYLE = {
    'JEWEL'        : dict(color = INK, lw = 1.5, ls = '-'),
    'PYTHIA input' : dict(color = MUTED, lw = 1.4, ls = ':'),
    'OT-CFM'       : dict(color = BLUE, lw = 1.5, ls = '-'),
    r'$\alpha$-DSBM' : dict(color = ORANGE, lw = 1.5, ls = '--'),
}
# the sequential ramp of the displays: one hue, light -> dark (dataviz
# reference blue, steps 100 -> 700); empty towers are the surface
RAMP = [ '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b' ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Translation deck figures')
    parser.add_argument('--out', default = 'docs/flow/translation')
    return parser.parse_args()

def setup():
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size' : 8, 'axes.titlesize' : 8, 'axes.labelsize' : 8,
        'xtick.labelsize' : 7, 'ytick.labelsize' : 7, 'legend.fontsize' : 7.5,
        'axes.edgecolor' : AXIS, 'axes.linewidth' : 0.6, 'axes.labelcolor' : INK2,
        'xtick.color' : INK2, 'ytick.color' : INK2, 'xtick.major.width' : 0.6,
        'ytick.major.width' : 0.6, 'axes.titlecolor' : INK, 'text.color' : INK,
        'axes.spines.top' : False, 'axes.spines.right' : False,
        'legend.frameon' : False, 'savefig.dpi' : 250, 'figure.dpi' : 100,
        'lines.solid_capstyle' : 'round', 'lines.dash_capstyle' : 'round',
    })
    return plt

def samples():
    test = te.load_set('test', 'pythia')
    jewel = te.load_set('test', 'jewel')
    out = lambda stem: te.clip(np.load(os.path.join(te.output_dir(), f'{stem}.npy')))  # pylint: disable=unnecessary-lambda-assignment
    canv = { 'JEWEL' : (jewel['canvas'], jewel['row']),
             'PYTHIA input' : (test['canvas'], test['row']),
             'OT-CFM' : (out('OT-CFM__midpoint128'), test['row']),
             r'$\alpha$-DSBM' : (out('alpha-DSBM__sde30'), test['row']) }
    return (canv, test, jewel)

def step_hist(ax, values, bins, name, wash = False):
    h = np.histogram(te.finite(values), bins, density = True)[0]
    x = np.repeat(bins, 2)[1:-1]
    y = np.repeat(h, 2)
    ax.plot(x, y, **STYLE[name], label = name)
    if wash:
        ax.fill_between(x, y, color = INK, alpha = 0.07, lw = 0)

def fig_marginals(plt, obs, path):
    """2 x 2: cone energy (all outputs) and mass, girth, z_lead (E >= 10 GeV)."""
    panels = [ ('E', 'cone energy (GeV)', (0, 62)), ('mass', 'mass (GeV)', (0.5, 9.5)),
               ('girth', 'girth', (0.04, 0.34)), ('zlead', r'$z_\mathrm{lead}$', (0.08, 0.86)) ]
    (fig, axes) = plt.subplots(2, 2, figsize = (3.75, 2.95))
    for (ax, (q, label, rng)) in zip(axes.flat, panels):
        bins = np.linspace(*rng, 40)
        for name in STYLE:
            o = obs[name]
            sel = np.ones(len(o['E']), bool) if q == 'E' else o['E'] >= te.MIN_JET
            step_hist(ax, o[q][sel], bins, name, wash = name == 'JEWEL')
        if q == 'E':
            ax.axvline(te.MIN_JET, color = AXIS, lw = 0.6)
        ax.set_xlabel(label, labelpad = 1)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlim(*rng)
    (h, l) = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc = 'lower center', ncol = 4, handlelength = 2.2, columnspacing = 1.2,
               bbox_to_anchor = (0.5, -0.01))
    fig.tight_layout(rect = (0, 0.07, 1, 1), h_pad = 0.6, w_pad = 1.0)
    fig.savefig(path)
    plt.close(fig)

def fig_profiles(plt, obs, path):
    """Mean girth and z_lead against the cone energy, E >= 10 GeV."""
    edges = np.array([ 10, 13, 16, 20, 25, 30, 36, 45, 60 ])
    mids  = 0.5 * (edges[1:] + edges[:-1])
    (fig, axes) = plt.subplots(1, 2, figsize = (5.4, 2.05))
    for (ax, q, label) in zip(axes, [ 'girth', 'zlead' ],
                              [ 'mean girth', r'mean $z_\mathrm{lead}$' ]):
        for name in STYLE:
            o = obs[name]
            m, e = [], []
            for (lo, hi) in zip(edges[:-1], edges[1:]):
                v = te.finite(o[q][(o['E'] >= lo) & (o['E'] < hi)])
                m.append(v.mean() if len(v) > 30 else np.nan)
                e.append(v.std() / np.sqrt(len(v)) if len(v) > 30 else np.nan)
            ax.errorbar(mids, m, e, **STYLE[name], label = name, marker = 'o', ms = 2.6,
                        mew = 0, capsize = 0, elinewidth = 0.8)
        ax.set_xlabel('cone energy (GeV)', labelpad = 1)
        ax.set_title(label, loc = 'left', pad = 3)
        ax.grid(axis = 'y', color = GRID, lw = 0.5)
        ax.set_axisbelow(True)
    axes[1].legend(loc = 'upper left', handlelength = 2.2)
    fig.tight_layout(w_pad = 1.5)
    fig.savefig(path)
    plt.close(fig)

def fig_energy(plt, obs, path):
    """Left: the cone-energy spectra; right: E_out / E_in against E_in."""
    (fig, axes) = plt.subplots(1, 2, figsize = (5.6, 2.25))
    bins = np.linspace(0, 62, 42)
    for name in STYLE:
        step_hist(axes[0], obs[name]['E'], bins, name, wash = name == 'JEWEL')
    axes[0].set_yticks([])
    axes[0].spines['left'].set_visible(False)
    axes[0].set_xlabel('cone energy (GeV)', labelpad = 1)
    axes[0].set_title('energy spectra', loc = 'left', pad = 3)
    e_in = obs['PYTHIA input']['E']
    edges = np.array([ 15, 20, 25, 30, 35, 40, 50, 60 ])
    mids  = 0.5 * (edges[1:] + edges[:-1])
    ax = axes[1]
    ax.axhline(1, color = AXIS, lw = 0.8)
    for name in [ 'OT-CFM', r'$\alpha$-DSBM' ]:
        r = obs[name]['E'] / e_in
        q = np.array([ np.percentile(r[(e_in >= a) & (e_in < b)], [ 16, 50, 84 ])
                       for (a, b) in zip(edges[:-1], edges[1:]) ])
        ax.fill_between(mids, q[:, 0], q[:, 2], color = STYLE[name]['color'], alpha = 0.12,
                        lw = 0)
        ax.plot(mids, q[:, 1], **STYLE[name], marker = 'o', ms = 2.6, mew = 0, label = name)
    ax.set_xlabel('input cone energy (GeV)', labelpad = 1)
    ax.set_title(r'$E_\mathrm{out} / E_\mathrm{in}$ (median, 16-84%)', loc = 'left', pad = 3)
    ax.set_ylim(0.3, 1.4)
    ax.grid(axis = 'y', color = GRID, lw = 0.5)
    ax.set_axisbelow(True)
    (h, l) = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc = 'lower center', ncol = 4, handlelength = 2.2, columnspacing = 1.4,
               bbox_to_anchor = (0.5, -0.01))
    fig.tight_layout(rect = (0, 0.1, 1, 1), w_pad = 1.8)
    fig.savefig(path)
    plt.close(fig)

def cmap(plt):
    # pylint: disable=import-outside-toplevel
    from matplotlib.colors import LinearSegmentedColormap
    m = LinearSegmentedColormap.from_list('blue', RAMP)
    m.set_bad('#ffffff')
    del plt
    return m

def show(ax, img, mask, colours, norm):
    w = img[OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * mask
    ax.imshow(np.ma.masked_less_equal(w, 0.01), cmap = colours, norm = norm)
    # the cone outline, as a reading aid
    ax.imshow(np.ma.masked_where(mask, np.ones_like(w)), cmap = 'Greys', vmin = 0, vmax = 8,
              alpha = 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRID)
        s.set_linewidth(0.5)
    return w.sum()

def fig_displays(plt, canv, test, jewel, mask, path, qs = (0.1, 0.5, 0.9)):
    """Three fixed inputs (the pre-registered display rule), their outputs,
    and a JEWEL jet at the same energy quantile (a reference, not a truth)."""
    # pylint: disable=import-outside-toplevel,too-many-locals
    from matplotlib.colors import LogNorm
    pick  = display_indices(test, qs = qs)
    ej    = jewel['e_cone']
    jpick = [ int(np.argmin(np.abs(ej - np.quantile(ej, q)))) for q in qs ]
    cols  = [ ('PYTHIA input', canv['PYTHIA input'][0][pick]),
              ('OT-CFM', canv['OT-CFM'][0][pick]),
              (r'$\alpha$-DSBM', canv[r'$\alpha$-DSBM'][0][pick]),
              ('a JEWEL jet\n(same E quantile)', jewel['canvas'][jpick]) ]
    colours = cmap(plt)
    norm = LogNorm(0.05, 30)
    (fig, axes) = plt.subplots(len(pick), len(cols), figsize = (3.55, 2.9))
    for (c, (name, imgs)) in enumerate(cols):
        for r in range(len(pick)):
            e = show(axes[r][c], imgs[r], mask, colours, norm)
            axes[r][c].set_xlabel(f'{e:.0f} GeV', labelpad = 1.5, fontsize = 7, color = INK2)
            if r == 0:
                axes[r][c].set_title(name, fontsize = 7.5, pad = 3)
    fig.tight_layout(h_pad = 0.5, w_pad = 0.4)
    fig.subplots_adjust(right = 0.86)
    cax = fig.add_axes((0.89, 0.2, 0.025, 0.6))
    sm = plt.cm.ScalarMappable(norm = norm, cmap = colours)
    cb = fig.colorbar(sm, cax = cax)
    cb.set_label('tower energy (GeV)', fontsize = 7, color = INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize = 6.5)
    fig.savefig(path)
    plt.close(fig)

def fig_thumbs(plt, canv, test, jewel, mask, out, q = 0.7):
    """One input, its OT-CFM output and a JEWEL jet, for the method sketch."""
    # pylint: disable=import-outside-toplevel
    from matplotlib.colors import LogNorm
    i = display_indices(test, qs = (q,))[0]
    ej = jewel['e_cone']
    j = int(np.argmin(np.abs(ej - np.quantile(ej, q))))
    colours = cmap(plt)
    for (name, img) in [ ('in', canv['PYTHIA input'][0][i]), ('out', canv['OT-CFM'][0][i]),
                         ('jewel', jewel['canvas'][j]) ]:
        (fig, ax) = plt.subplots(figsize = (0.9, 0.9))
        show(ax, img, mask, colours, LogNorm(0.05, 30))
        fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
        fig.savefig(os.path.join(out, f'deck_thumb_{name}.png'))
        plt.close(fig)

def main():
    cmdargs = parse_cmdargs()
    plt = setup()
    torch.set_num_threads(8)
    (canv, test, jewel) = samples()
    obs = { name : te.jet_observables(img, rows, torch.device('cpu'))
            for (name, (img, rows)) in canv.items() }
    mask = Geometry(torch.device('cpu')).mask.numpy()
    out = cmdargs.out
    fig_marginals(plt, obs, os.path.join(out, 'deck_marginals.png'))
    fig_profiles(plt, obs, os.path.join(out, 'deck_profiles.png'))
    fig_energy(plt, obs, os.path.join(out, 'deck_energy.png'))
    fig_displays(plt, canv, test, jewel, mask, os.path.join(out, 'deck_displays.png'))
    fig_thumbs(plt, canv, test, jewel, mask, out)
    print(f'wrote the deck figures to {out}')

if __name__ == '__main__':
    main()
