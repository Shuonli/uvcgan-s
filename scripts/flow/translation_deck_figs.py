#!/usr/bin/env python
"""Figures of the PYTHIA -> JEWEL translation deck
(docs/flow/translation/slides/translation_deck.tex): OT flow matching against
CycleGAN, drawn at the size they are printed.

Every distribution comes from the per-jet observables translation_report.py
saved for its report (OUTDIR/sphenix/flow/translation/report/<--report>.npz:
the fixed R = 0.4 crop and the leading anti-kT jet refound in each canvas),
so the figures show the report's numbers; the displays come from the saved
outputs (--models LABEL=STEM), and the training curves from the report's
curves.csv.

Series: JEWEL test (the target: ink with a light wash), the PYTHIA input
(grey, dotted), OT-FM (blue), CycleGAN (orange, dashed; a darker step for
longer training). Blue/orange passes the colour-vision check (dataviz
validator: worst Delta E 24.7); the green/red of the pilot's figures does
not (3.9).

    translation_deck_figs.py --report cgan \\
        --models 'OT-FM=OT-CFM__midpoint128,CycleGAN=CycleGAN-2h__gen'
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import fm_common as fc
import translation_eval as te
from closure_data import OFFSET
from substructure import Geometry
from translation_report import display_indices

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
AXIS, GRID = '#c3c2b7', '#e1e0d9'
COLOURS = { 'OT-FM' : '#2a78d6', 'CycleGAN' : '#eb6834', 'CycleGAN 8 h' : '#c5521f',
            'CycleGAN 24 h' : '#9c3d14' }
DASHES  = { 'CycleGAN' : '--', 'CycleGAN 8 h' : (0, (5, 1.5, 1, 1.5)),
            'CycleGAN 24 h' : '-.' }
RANDOM = '#4a3aa7'      # random JEWEL: violet (passes the CVD check with blue and orange)
# the stochastic pilot's conditional FM arms: aqua (hard OT), yellow (soft OT);
# with OT-FM blue and random JEWEL violet they pass the CVD check
CONDFM = { 'hard' : ('#1baf7a', '--'), 'soft' : ('#eda100', '-.') }
# report sample names -> deck names
NAMES = { 'JEWEL test' : 'JEWEL', 'identity' : 'PYTHIA input', 'random JEWEL' : 'random JEWEL' }
# the sequential ramp of the displays (dataviz reference blue, 100 -> 700)
RAMP = [ '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b' ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Translation deck figures')
    parser.add_argument('--report', default = 'cgan',
        help = 'tag of the report (its observables in translation/report/TAG.npz,'
               ' its tables in docs/flow/translation/TAG)')
    parser.add_argument('--models', required = True,
        help = 'comma separated LABEL=STEM of the outputs, in plotting order')
    parser.add_argument('--out', default = 'docs/flow/translation/deck')
    return parser.parse_args()

def style(name):
    if name == 'JEWEL':
        return dict(color = INK, lw = 1.5, ls = '-')
    if name == 'PYTHIA input':
        return dict(color = MUTED, lw = 1.4, ls = ':')
    if name == 'random JEWEL':
        return dict(color = RANDOM, lw = 1.5, ls = '-.')
    if name.startswith('CondFM'):
        (c, ls) = CONDFM['hard' if 'hard' in name else 'soft']
        return dict(color = c, lw = 1.5, ls = ls)
    if name.startswith('CycleGAN'):
        key = 'CycleGAN 24 h' if '24' in name else ('CycleGAN 8 h' if ' 8 ' in name + ' '
                                                    else 'CycleGAN')
        return dict(color = COLOURS[key], lw = 1.5, ls = DASHES[key])
    return dict(color = COLOURS.get(name, INK2), lw = 1.5, ls = '-')

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

def load_observables(tag, labels):
    """{deck name: {'crop': {obs: array}, 'refound': {obs: array}}}."""
    path = os.path.join(fc.translation_root(), 'report', f'{tag}.npz')
    out = {}
    with np.load(path) as f:
        for key in f.files:
            (kind, name, q) = key.split('|')
            name = NAMES.get(name, name)
            if name in [ 'JEWEL', 'PYTHIA input', 'random JEWEL' ] + labels:
                out.setdefault(name, {}).setdefault(kind, {})[q] = f[key]
    return out

def series(labels):
    return [ 'JEWEL', 'PYTHIA input' ] + labels

def step_hist(ax, values, bins, name, wash = False, log = False):
    (h, _) = np.histogram(te.finite(values), bins)
    dens = h / (h.sum() * np.diff(bins))
    x = np.repeat(bins, 2)[1:-1]
    y = np.repeat(dens, 2)
    if log:
        y = np.where(y > 0, y, np.nan)
    ax.plot(x, y, **style(name), label = name)
    if wash:
        ax.fill_between(x, np.nan_to_num(y), color = INK, alpha = 0.07, lw = 0)
    return h

def legend_below(fig, ax, ncol = 4, y = -0.01):
    (h, l) = ax.get_legend_handles_labels()
    fig.legend(h, l, loc = 'lower center', ncol = ncol, handlelength = 2.4,
               columnspacing = 1.3, bbox_to_anchor = (0.5, y))

def fig_pt(plt, obs, labels, path):
    """Jet pT spectra (log) and their ratio to JEWEL: the R = 0.4 cone pT
    (all outputs) and the leading anti-kT jet refound in each canvas."""
    names = series(labels)
    # up to 70 GeV, or further when a model's spectrum reaches beyond it
    top99 = max(np.nanpercentile(obs[name][kind][q], 99.5) for name in names
                for (kind, q) in [ ('crop', 'E'), ('refound', 'pt') ])
    bins = np.arange(0, min(162.5, max(72.5, 2.5 * np.ceil(top99 / 2.5) + 2.5)), 2.5)
    (fig, axes) = plt.subplots(2, 2, figsize = (5.7, 3.25), sharex = 'col',
                               gridspec_kw = { 'height_ratios' : (2.2, 1), 'hspace' : 0.08 })
    for (c, (kind, q, label)) in enumerate([
            ('crop', 'E', r'jet $p_T$ in the R = 0.4 cone (GeV)'),
            ('refound', 'pt', r'leading anti-$k_T$ jet $p_T$ (GeV)') ]):
        (top, bot) = (axes[0][c], axes[1][c])
        counts = {}
        for name in names:
            counts[name] = step_hist(top, obs[name][kind][q], bins, name,
                                     wash = name == 'JEWEL', log = True)
        top.set_yscale('log')
        top.set_ylim(2e-5, 0.15)
        top.set_ylabel('jets per GeV (norm.)' if c == 0 else '')
        top.axvline(10, color = AXIS, lw = 0.6)
        ref = counts['JEWEL']
        x = np.repeat(bins, 2)[1:-1]
        ok = ref >= 20
        err = np.where(ok, 1 / np.sqrt(np.maximum(ref, 1)), np.nan)
        bot.fill_between(x, np.repeat(1 - err, 2), np.repeat(1 + err, 2),
                         color = INK, alpha = 0.10, lw = 0)
        bot.axhline(1, color = INK, lw = 0.8)
        for name in names[1:]:
            n = counts[name]
            r = np.where(ok & (n > 0), (n / n.sum()) / (ref / ref.sum()), np.nan)
            bot.plot(x, np.repeat(r, 2), **style(name))
        bot.set_ylim(0, 2.4)
        bot.set_xlim(0, bins[-1])
        bot.set_xlabel(label, labelpad = 1)
        bot.set_ylabel('ratio to JEWEL' if c == 0 else '')
        bot.axvline(10, color = AXIS, lw = 0.6)
        bot.grid(axis = 'y', color = GRID, lw = 0.5)
        bot.set_axisbelow(True)
    legend_below(fig, axes[0][0], ncol = len(names))
    fig.subplots_adjust(left = 0.1, right = 0.98, top = 0.97, bottom = 0.21, wspace = 0.22)
    fig.savefig(path)
    plt.close(fig)

def fig_marginals(plt, obs, labels, path):
    """Substructure in the cone, every sample cut at pT >= 10 GeV."""
    names = series(labels)
    panels = [ ('mass', 'mass (GeV)', (0.5, 9.5)), ('girth', 'girth', (0.04, 0.34)),
               ('zlead', r'$z_\mathrm{lead}$', (0.08, 0.86)), ('ptd', r'$p_T^D$', (0.22, 0.85)) ]
    (fig, axes) = plt.subplots(2, 2, figsize = (3.75, 2.95))
    for (ax, (q, label, rng)) in zip(axes.flat, panels):
        bins = np.linspace(*rng, 40)
        for name in names:
            o = obs[name]['crop']
            step_hist(ax, o[q][o['E'] >= te.MIN_JET], bins, name, wash = name == 'JEWEL')
        ax.set_xlabel(label, labelpad = 1)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlim(*rng)
    ncol = len(names) if len(names) <= 3 else int(np.ceil(len(names) / 2))
    legend_below(fig, axes[0][0], ncol = ncol)
    fig.tight_layout(rect = (0, 0.07 if ncol == len(names) else 0.13, 1, 1), h_pad = 0.6,
                     w_pad = 1.0)
    fig.savefig(path)
    plt.close(fig)

def fig_profiles(plt, obs, labels, path):
    """Mean girth and z_lead against the cone pT."""
    names = series(labels)
    edges = np.array([ 10, 13, 16, 20, 25, 30, 36, 45, 60 ])
    mids  = 0.5 * (edges[1:] + edges[:-1])
    (fig, axes) = plt.subplots(1, 2, figsize = (5.4, 2.05))
    for (ax, q, label) in zip(axes, [ 'girth', 'zlead' ],
                              [ 'mean girth', r'mean $z_\mathrm{lead}$' ]):
        for name in names:
            o = obs[name]['crop']
            m, e = [], []
            for (lo, hi) in zip(edges[:-1], edges[1:]):
                v = te.finite(o[q][(o['E'] >= lo) & (o['E'] < hi)])
                m.append(v.mean() if len(v) > 30 else np.nan)
                e.append(v.std() / np.sqrt(len(v)) if len(v) > 30 else np.nan)
            ax.errorbar(mids, m, e, **style(name), label = name, marker = 'o', ms = 2.6,
                        mew = 0, capsize = 0, elinewidth = 0.8)
        ax.set_xlabel(r'jet $p_T$ (GeV)', labelpad = 1)
        ax.set_title(label, loc = 'left', pad = 3)
        ax.grid(axis = 'y', color = GRID, lw = 0.5)
        ax.set_axisbelow(True)
    axes[1].legend(loc = 'upper left', handlelength = 2.4)
    fig.tight_layout(w_pad = 1.5)
    fig.savefig(path)
    plt.close(fig)

def fig_ptchange(plt, obs, labels, path):
    """pT_out / pT_in (median, 16-84%) against the input pT."""
    pt_in = obs['PYTHIA input']['crop']['E']
    edges = np.array([ 15, 20, 25, 30, 35, 40, 50, 60 ])
    mids  = 0.5 * (edges[1:] + edges[:-1])
    (fig, ax) = plt.subplots(figsize = (3.3, 2.15))
    ax.axhline(1, color = AXIS, lw = 0.8)
    top = 1.5
    for name in labels:
        r = obs[name]['crop']['E'] / pt_in
        q = np.array([ np.percentile(r[(pt_in >= a) & (pt_in < b)], [ 16, 50, 84 ])
                       for (a, b) in zip(edges[:-1], edges[1:]) ])
        top = max(top, 1.08 * q[:, 2].max())
        ax.fill_between(mids, q[:, 0], q[:, 2], color = style(name)['color'], alpha = 0.12,
                        lw = 0)
        ax.plot(mids, q[:, 1], **style(name), marker = 'o', ms = 2.6, mew = 0, label = name)
    ax.set_xlabel(r'input jet $p_T$ (GeV)', labelpad = 1)
    ax.set_title(r'$p_{T,\mathrm{out}} / p_{T,\mathrm{in}}$ (median, 16-84%)', loc = 'left',
                 pad = 3)
    ax.set_ylim(0.3, min(top, 3.5))
    ax.grid(axis = 'y', color = GRID, lw = 0.5)
    ax.set_axisbelow(True)
    ax.legend(loc = 'lower right', handlelength = 2.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def fig_curves(plt, curves, refs, path):
    """Validation: mean W1/sigma of the 7 observables to JEWEL val against the
    training time; OT-FM at 32 NFE, CycleGAN at each checkpoint."""
    (fig, ax) = plt.subplots(figsize = (3.3, 2.15))
    for (name, d) in curves.items():
        d = d.sort_values('train_time')
        y = d[[ f'w1_{q}' for q in te.OBS ]].mean(axis = 1)
        ax.plot(d['train_time'] / 3600, y, **style(name), marker = 'o', ms = 2.6, mew = 0,
                label = name)
    for (key, label) in [ ('identity', 'PYTHIA input'), ('random JEWEL', 'JEWEL vs JEWEL') ]:
        v = refs[refs['sample'] == key][[ f'w1_{q}' for q in te.OBS ]].mean(axis = 1).iloc[0]
        ax.axhline(v, color = MUTED, lw = 0.8, ls = ':')
        ax.text(0.02, v * 1.08, label, color = INK2, fontsize = 6.5,
                transform = ax.get_yaxis_transform())
    ax.axvline(2, color = AXIS, lw = 0.8)
    ax.set_yscale('log')
    ax.set_xlabel('training time (A6000 GPU hours)', labelpad = 1)
    ax.set_title(r'distance to JEWEL (mean W1/$\sigma$, validation)', loc = 'left', pad = 3)
    ax.legend(loc = 'upper right', handlelength = 2.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

SHAPES = [ ('girth', 'girth', 0.12), ('zlead', r'$z_\mathrm{lead}$', 0.5),
           ('ptd', r'$p_T^D$', 0.45), ('mass', 'mass (GeV)', 5.0),
           ('zg', r'$z_g$', 0.35), ('rg', r'$R_g$', 0.45) ]

def fig_shape_changes(plt, obs, labels, path):
    """Per-jet change of each shape observable, output minus its own input
    (all 20k test jets): the models against a random JEWEL jet, which has the
    right distributions but ignores the input."""
    o_in = obs['PYTHIA input']['crop']
    names = labels + [ 'random JEWEL' ]
    (fig, axes) = plt.subplots(2, 3, figsize = (5.7, 3.1))
    for (ax, (q, label, half)) in zip(axes.flat, SHAPES):
        bins = np.linspace(-half, half, 61)
        for name in names:
            d = obs[name]['crop'][q] - o_in[q]
            step_hist(ax, d[np.isfinite(d)], bins, name)
        ax.axvline(0, color = AXIS, lw = 0.6)
        ax.set_xlabel(r'$\Delta$ ' + label, labelpad = 1)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlim(-half, half)
    legend_below(fig, axes[0][0], ncol = len(names))
    fig.tight_layout(rect = (0, 0.07, 1, 1), h_pad = 0.8, w_pad = 0.8)
    fig.savefig(path)
    plt.close(fig)

def fig_shape_scatter(plt, obs, labels, path):
    """Output against its own input, per jet: the first model and a random
    JEWEL jet, for girth, z_lead and p_T^D (density, log colour; r: the
    correlation)."""
    # pylint: disable=import-outside-toplevel
    from matplotlib.colors import LogNorm
    o_in = obs['PYTHIA input']['crop']
    rows = [ labels[0], 'random JEWEL' ]
    cols = [ ('girth', 'girth', (0.03, 0.33)), ('zlead', r'$z_\mathrm{lead}$', (0.08, 0.85)),
             ('ptd', r'$p_T^D$', (0.2, 0.85)) ]
    (fig, axes) = plt.subplots(2, 3, figsize = (5.4, 3.55))
    for (r, name) in enumerate(rows):
        for (c, (q, label, rng)) in enumerate(cols):
            ax = axes[r][c]
            (a, b) = (o_in[q], obs[name]['crop'][q])
            ok = np.isfinite(a) & np.isfinite(b)
            ax.hist2d(a[ok], b[ok], bins = 50, range = (rng, rng), cmap = cmap(),
                      norm = LogNorm(1, 400), rasterized = True)
            ax.plot(rng, rng, color = INK, lw = 0.7, ls = '--')
            corr = np.corrcoef(a[ok], b[ok])[0, 1]
            ax.text(0.04, 0.95, f'r = {corr:.2f}'.replace('-0.00', '0.00'), transform = ax.transAxes, va = 'top',
                    fontsize = 7, color = INK)
            ax.set_xlabel(f'input {label}' if r == 1 else '', labelpad = 1)
            ax.set_ylabel(f'{name}\noutput {label}' if c == 0 else f'output {label}',
                          labelpad = 1)
            ax.set_aspect('equal')
            ax.tick_params(labelsize = 6.5)
    fig.tight_layout(h_pad = 0.6, w_pad = 0.6)
    fig.savefig(path)
    plt.close(fig)

def cmap():
    # pylint: disable=import-outside-toplevel
    from matplotlib.colors import LinearSegmentedColormap
    m = LinearSegmentedColormap.from_list('blue', RAMP)
    m.set_bad('#ffffff')
    return m

def show(ax, img, mask, colours, norm):
    w = img[OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * mask
    ax.imshow(np.ma.masked_less_equal(w, 0.01), cmap = colours, norm = norm)
    ax.imshow(np.ma.masked_where(mask, np.ones_like(w)), cmap = 'Greys', vmin = 0, vmax = 8,
              alpha = 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRID)
        s.set_linewidth(0.5)
    return w.sum()

def fig_displays(plt, outputs, mask, path, qs = (0.1, 0.5, 0.9)):
    """Three fixed inputs (the pilot's display rule), their outputs, and a
    JEWEL jet at the same pT quantile (a reference, not a truth)."""
    # pylint: disable=import-outside-toplevel,too-many-locals
    from matplotlib.colors import LogNorm
    test  = te.load_set('test', 'pythia')
    jewel = te.load_set('test', 'jewel')
    pick  = display_indices(test, qs = qs)
    ej    = jewel['e_cone']
    jpick = [ int(np.argmin(np.abs(ej - np.quantile(ej, q)))) for q in qs ]
    cols  = [ ('PYTHIA input', test['canvas'][pick]) ]
    cols += [ (name, img[pick]) for (name, img) in outputs.items() ]
    cols += [ ('a JEWEL jet\n(same $p_T$ quantile)', jewel['canvas'][jpick]) ]
    colours = cmap()
    norm = LogNorm(0.05, 30)
    (fig, axes) = plt.subplots(len(pick), len(cols), figsize = (0.86 * len(cols) + 0.5, 2.9))
    for (c, (name, imgs)) in enumerate(cols):
        for r in range(len(pick)):
            e = show(axes[r][c], imgs[r], mask, colours, norm)
            axes[r][c].set_xlabel(f'{e:.0f} GeV', labelpad = 1.5, fontsize = 7, color = INK2)
            if r == 0:
                axes[r][c].set_title(name, fontsize = 7.5, pad = 3)
    fig.tight_layout(h_pad = 0.5, w_pad = 0.4)
    fig.subplots_adjust(right = 0.87)
    cax = fig.add_axes((0.895, 0.2, 0.022, 0.6))
    sm = plt.cm.ScalarMappable(norm = norm, cmap = colours)
    cb = fig.colorbar(sm, cax = cax)
    cb.set_label(r'tower $E_T$ (GeV)', fontsize = 7, color = INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize = 6.5)
    fig.savefig(path)
    plt.close(fig)

def fig_thumbs(plt, model_output, mask, out, q = 0.7):
    """One fixed input, the first model's output and a JEWEL jet at the
    same pT quantile, for the method sketch."""
    # pylint: disable=import-outside-toplevel
    from matplotlib.colors import LogNorm
    test  = te.load_set('test', 'pythia')
    jewel = te.load_set('test', 'jewel')
    i = display_indices(test, qs = (q,))[0]
    j = int(np.argmin(np.abs(jewel['e_cone'] - np.quantile(jewel['e_cone'], q))))
    for (name, img) in [ ('in', test['canvas'][i]), ('out', model_output[i]),
                         ('jewel', jewel['canvas'][j]) ]:
        (fig, ax) = plt.subplots(figsize = (0.9, 0.9))
        show(ax, img, mask, cmap(), LogNorm(0.05, 30))
        fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
        fig.savefig(os.path.join(out, f'thumb_{name}.png'))
        plt.close(fig)

def main():
    cmdargs = parse_cmdargs()
    plt = setup()
    specs = [ s.split('=', 1) for s in cmdargs.models.split(',') ]
    labels = [ l for (l, _) in specs ]
    obs = load_observables(cmdargs.report, labels)
    os.makedirs(cmdargs.out, exist_ok = True)
    out = lambda name: os.path.join(cmdargs.out, name)    # pylint: disable=unnecessary-lambda-assignment
    fig_pt(plt, obs, labels, out('deck_pt.png'))
    fig_marginals(plt, obs, labels, out('deck_marginals.png'))
    fig_profiles(plt, obs, labels, out('deck_profiles.png'))
    fig_ptchange(plt, obs, labels, out('deck_ptchange.png'))
    fig_shape_changes(plt, obs, labels, out('deck_shape_changes.png'))
    fig_shape_scatter(plt, obs, labels, out('deck_shape_scatter.png'))
    tables = os.path.join('docs/flow/translation', cmdargs.report)
    if os.path.exists(os.path.join(tables, 'curves.csv')):
        c = pd.read_csv(os.path.join(tables, 'curves.csv'))
        c = c[c['setting'] != 'euler4']
        # OT-FM's curve, and CycleGAN's over all its checkpoints
        names = [ m for m in [ 'OT-FM' ] + sorted(set(c['model'])) if (c['model'] == m).any()
                  and (m == 'OT-FM' or m.startswith(('CycleGAN', 'CondFM'))) ]
        curves = { m : c[c['model'] == m] for m in dict.fromkeys(names) }
        fig_curves(plt, curves, pd.read_csv(os.path.join(tables, 'curves_refs.csv')),
                   out('deck_curves.png'))
    mask = Geometry(torch.device('cpu')).mask.numpy()
    outputs = { l : te.clip(np.load(os.path.join(te.output_dir(), f'{s}.npy')))
                for (l, s) in specs }
    fig_displays(plt, outputs, mask, out('deck_displays.png'))
    fig_thumbs(plt, outputs[labels[0]], mask, cmdargs.out)
    print(f'wrote the deck figures to {cmdargs.out}')

if __name__ == '__main__':
    main()
