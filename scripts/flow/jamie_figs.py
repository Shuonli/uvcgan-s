#!/usr/bin/env python
"""Figures of the toy study (FLOW_NOTES.md, "Jamie's toy exercises with OT
flow matching"), from the held-out sets, the frozen-solver outputs and the
report tables (jamie_report.py). Every figure states what it shows; mean
images are labelled as ensemble means.

    jamie_figs.py --exercise 1 --models 'E1-U=E1-U_s0,...' [--out DIR]
"""

import argparse
import os

import numpy as np
import pandas as pd

import jamie_report as jr
import toycalo as tc

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
AXIS, GRID = '#c3c2b7', '#e1e0d9'
# categorical order of the reference palette (validated for colour vision in
# the earlier decks); a figure assigns them to its models in this order
PALETTE = [ '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7' ]
DASHES = [ '-', '--', '-.', (0, (5, 1.5, 1, 1.5)), ':', (0, (3, 1)), (0, (1, 1)) ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Toy study figures')
    parser.add_argument('--exercise', required = True,
                        choices = [ '1', '2', '3', '3g', '4', '22' ])
    parser.add_argument('--models', required = True)
    parser.add_argument('--out', default = 'docs/flow/jamie_otfm')
    return parser.parse_args()

def setup():
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size' : 7, 'axes.titlesize' : 7, 'axes.labelsize' : 7,
        'xtick.labelsize' : 6, 'ytick.labelsize' : 6, 'legend.fontsize' : 6,
        'axes.edgecolor' : AXIS, 'axes.linewidth' : 0.6, 'axes.labelcolor' : INK2,
        'xtick.color' : INK2, 'ytick.color' : INK2, 'axes.titlecolor' : INK,
        'axes.spines.top' : False, 'axes.spines.right' : False,
        'legend.frameon' : False, 'savefig.dpi' : 220,
    })
    return plt

def style(k):
    return dict(color = PALETTE[k % len(PALETTE)], ls = DASHES[k % len(DASHES)], lw = 1.3)

def draw_image(ax, img, eta0, phi0, title, vmax = 1.5, half_width = None):
    """A canvas centred in phi on the jet axis, log10(E + 0.1), circles at
    R = 0.4 and 1.0; half_width: show only |phi - phi_jet| < half_width."""
    shift = int(round(tc.NPHI / 2 - np.mod(phi0, 2 * np.pi) / tc.DPHI))
    x = np.roll(img, shift, axis = 1)
    ax.imshow(np.log10(np.clip(x, 0, None) + 0.1), origin = 'lower', aspect = 'equal',
              cmap = 'viridis', vmin = -1, vmax = vmax,
              extent = (-np.pi, np.pi, tc.ETA_MIN, tc.ETA_MAX))
    phi_c = np.mod(phi0, 2 * np.pi) + shift * tc.DPHI - np.pi
    phi_c = np.angle(np.exp(1j * phi_c))
    t = np.linspace(0, 2 * np.pi, 100)
    for r in (0.4, 1.0):
        ax.plot(phi_c + r * np.cos(t), eta0 + r * np.sin(t), color = 'white', lw = 0.5)
    if half_width:
        ax.set_xlim(phi_c - half_width, phi_c + half_width)
    else:
        ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(tc.ETA_MIN, tc.ETA_MAX)
    ax.set_title(title, fontsize = 5.5, pad = 2)
    ax.set_xticks([])
    ax.set_yticks([])

def pick_events(e, targets, n = 300):
    """Fixed display events: among the first n, the nearest to each energy."""
    out = []
    for t in targets:
        d = np.abs(e[:n] - t)
        d[out] = np.inf
        out.append(int(np.argmin(d)))
    return out

# --- Exercise 1

def fig_ex1(models, out, plt):
    # pylint: disable=too-many-locals
    d = jr.cache('t1_mix')
    ax = jr.axes(d)
    dr = jr.dr_of(ax[:300])
    jet = d['jet'][:300]
    mix = (d['jet'] + d['ue'])[:300]
    cone = dr < 0.4
    e = (jet * cone).sum((1, 2))
    ev = pick_events(e, (25, 35, 50))
    rho = jr.rho_far(mix, dr)
    cols = [ ('mixture M', mix), ('true jet J', jet) ]
    for (lab, dirn) in models:
        b = jr.output(dirn, 't1_mix_mix')
        if b is not None:
            cols.append((f'{lab}: M - B_hat', mix - b[:300]))
    cols.append(('rho per tower (signed)', mix - rho[:, None, None]))
    (fig, axes) = plt.subplots(len(ev), len(cols), figsize = (1.5 * len(cols), 0.8 * len(ev) + 0.4))
    for (r, i) in enumerate(ev):
        for (c, (name, x)) in enumerate(cols):
            cs = (x[i] * cone[i]).sum()
            draw_image(axes[r][c], x[i], ax[i, 0], ax[i, 1], f'{name}\ncone {cs:.1f} GeV'
                       if r == 0 else f'cone {cs:.1f} GeV')
    fig.suptitle('Exercise 1: fixed held-out events (true cone energy 25, 35, 50 GeV); '
                 'log10(E + 0.1), circles R = 0.4 and 1.0', fontsize = 6.5)
    fig.tight_layout(rect = (0, 0, 1, 0.94), h_pad = 0.3, w_pad = 0.2)
    fig.savefig(os.path.join(out, 'ex1_displays.png'))
    plt.close(fig)
    # Jamie's ring table: true support against empty towers
    t = pd.read_csv(os.path.join(out, 'ex1_subtraction.csv'))
    # the per-tower rho with negatives dropped is in the table only (255 GeV far)
    t = t[t['ring_0_0.1_true'].notna() & ~t['model'].str.contains('negatives dropped')]
    rings = [ '0_0.1', '0.1_0.2', '0.2_0.3', '0.3_0.4', '0.4_0.6', '0.6_inf' ]
    labels = [ 'r<0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.6', '>0.6' ]
    (fig, axs) = plt.subplots(1, 2, figsize = (6.4, 2.1))
    x = np.arange(len(rings))
    w = 0.8 / (len(t) + 1)
    axs[0].bar(x - 0.4 + w / 2, [ t.iloc[0][f'ring_{r}_true'] for r in rings ], w,
               color = INK, label = 'true jet')
    for (k, (_, row)) in enumerate(t.iterrows()):
        axs[0].bar(x - 0.4 + w * (k + 1.5), [ row[f'ring_{r}_on_true'] for r in rings ], w,
                   color = PALETTE[k % len(PALETTE)], label = f"{row['model']}")
        axs[1].bar(x - 0.4 + w * (k + 1.5), [ row[f'ring_{r}_on_empty'] for r in rings ], w,
                   color = PALETTE[k % len(PALETTE)])
    for (a, title) in zip(axs, [ 'J_hat on the true jet towers (GeV per jet)',
                                 'J_hat on truth-empty towers: the halo (GeV per jet)' ]):
        a.set_xticks(x)
        a.set_xticklabels(labels)
        a.set_title(title, loc = 'left')
        a.axhline(0, color = AXIS, lw = 0.6)
        a.set_xlabel('distance to the jet axis')
    for a in axs:
        a.set_yscale('symlog', linthresh = 0.1)
    axs[0].legend(fontsize = 5, ncol = 2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'ex1_rings.png'))
    plt.close(fig)

# --- Exercise 2 and the prior/data matrix

def fig_shifts(out, name, plt, title):
    t = pd.read_csv(os.path.join(out, f'{name}_shifts.csv'))
    qs = [ ('cone', 'cone energy (GeV)'), ('girth', 'girth'), ('ring', 'ring 0.4-1.0 (GeV)'),
           ('mass', 'mass (GeV)') ]
    (fig, axs) = plt.subplots(1, len(qs), figsize = (7.0, 2.2))
    for (a, (q, lab)) in zip(axs, qs):
        vals = list(t[f'{q}_shift'])
        errs = list(t[f'{q}_shift_sd'])
        names = list(t['model'])
        a.barh(np.arange(len(vals)), vals, xerr = errs, color = [ PALETTE[k % len(PALETTE)]
               for k in range(len(vals)) ], height = 0.7, error_kw = dict(lw = 0.6, ecolor = INK2))
        a.axvline(t[f'{q}_truth_shift'].iloc[0], color = INK, lw = 1.0, ls = '--', label = 'truth')
        a.axvline(0, color = AXIS, lw = 0.6)
        a.set_yticks(np.arange(len(vals)))
        a.set_yticklabels(names if a is axs[0] else [ '' ] * len(vals), fontsize = 5.5)
        a.set_title(lab, loc = 'left', fontsize = 6)
        a.invert_yaxis()
    fig.suptitle(title + '; quenched - vacuum, mean per jet; dashed: truth', fontsize = 6.5)
    fig.tight_layout(rect = (0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, f'{name}_shifts.png'))
    plt.close(fig)

# --- Exercise 3 / 4: marginals and the conditional distributions

def fig_translation(models, out, ex, plt):
    # pylint: disable=too-many-locals,too-many-statements
    ue_case = ex == '4'
    setname = { '3' : 't3_pair', '3g' : 't3g_pair', '4' : 't4_pair' }[ex]
    bankname = { '3' : 't3_bank', '3g' : 't3g_bank', '4' : 't4_bank' }[ex]
    d = jr.cache(setname)
    ax = jr.axes(d)
    dr = jr.dr_of(ax)
    if ue_case:
        (inp, truth) = (d['jet'] + d['ue'], d['med'] + d['ue'])
    else:
        (inp, truth) = (d['jet'], d['med'])
    def readout(x, drr):
        if ue_case:
            u = jr.ue_readout(x, drr)
            return { 'cone' : u['cone_sub'], 'ring' : u['ring_sub'],
                     'far' : jr.region_sum(x, drr, 1.0, 99) }
        return { 'cone' : jr.region_sum(x, drr, 0, 0.4), 'ring' : jr.region_sum(x, drr, 0.4, 1.0),
                 'far' : jr.region_sum(x, drr, 1.0, 99) }
    series = [ ('true quenched', readout(truth, dr), dict(color = INK, lw = 1.6, ls = '-')),
               ('input (vacuum)', readout(inp, dr), dict(color = MUTED, lw = 1.3, ls = ':')) ]
    outs = {}
    for (k, (lab, dirn)) in enumerate(models):
        x = jr.output(dirn, f'{setname}_vac')
        if x is not None:
            outs[lab] = x
            series.append((lab, readout(x, dr), style(k)))
    qs = [ ('cone', 'cone R < 0.4 (GeV)' + (', minus rho A' if ue_case else '')),
           ('ring', 'ring 0.4-1.0 (GeV)' + (', minus rho A' if ue_case else '')),
           ('far', 'R > 1.0 (GeV)') ]
    (fig, axs) = plt.subplots(1, 3, figsize = (7.0, 2.1))
    for (a, (q, lab)) in zip(axs, qs):
        allv = np.concatenate([ s[1][q] for s in series ])
        lo, hi = np.percentile(allv, [ 0.5, 99.5 ])
        bins = np.linspace(lo, hi, 50)
        for (name, o, st) in series:
            a.hist(o[q], bins, histtype = 'step', density = True, label = name, **st)
        a.set_xlabel(lab)
        a.set_yticks([])
    axs[0].legend(fontsize = 5)
    fig.suptitle(f'Exercise {ex}: 20k held-out jets, one output each', fontsize = 6.5)
    fig.tight_layout(rect = (0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, f'ex{ex}_marginals.png'))
    plt.close(fig)
    # the illustration parents: true quenchings against model draws
    b = jr.cache(bankname)
    par = b['illu_parents']
    a4 = ax[par]
    reps = b['illu_reps'].astype(np.float32)
    if ue_case:
        reps = reps + d['ue'][par][:, None]
    (fig, axs) = plt.subplots(2, len(par), figsize = (1.75 * len(par), 3.2))
    for (c, p) in enumerate(par):
        drp = jr.dr_of(np.repeat(a4[c:c + 1], reps.shape[1], axis = 0))
        tv = readout(reps[c], drp)
        for (r, q) in enumerate([ 'cone', 'ring' ]):
            a = axs[r][c]
            bins = np.linspace(*np.percentile(tv[q], [ 0.5, 99.5 ]), 30)
            a.hist(tv[q], bins, density = True, histtype = 'stepfilled', color = INK, alpha = 0.15,
                   label = f'true quenchings ({reps.shape[1]})')
            iv = readout(inp[p:p + 1], drp[:1])[q][0]
            a.axvline(iv, color = MUTED, ls = ':', lw = 1.0, label = 'input')
            for (k, (lab, dirn)) in enumerate(models):
                x = jr.output(dirn, f'{bankname}_illu')
                if x is None:
                    continue
                x = x[c].astype(np.float32)
                v = readout(x, drp[:len(x)])[q]
                if len(x) > 1:
                    a.hist(v, bins, density = True, histtype = 'step', label = f'{lab} ({len(x)})',
                           **style(k))
                else:
                    a.axvline(v[0], label = f'{lab} (one output)', **style(k))
            if r == 0:
                a.set_title(f'parent {p}', fontsize = 6)
            a.set_xlabel(f'{q} (GeV)', fontsize = 6)
            a.set_yticks([])
    (handles, labels) = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'lower center', ncol = len(labels), fontsize = 5.5)
    fig.suptitle(f'Exercise {ex}: one input jet, many true quenchings against many model '
                 'draws (the conditional law)', fontsize = 6.5)
    fig.tight_layout(rect = (0, 0.07, 1, 0.93))
    fig.savefig(os.path.join(out, f'ex{ex}_conditional.png'))
    plt.close(fig)
    # displays: one illustration parent, true quenchings and model draws, every image
    p = par[1]
    rows = [ ('truth', reps[1][:5]) ]
    for (lab, dirn) in models:
        x = jr.output(dirn, f'{bankname}_illu')
        if x is not None:
            rows.append((lab, x[1][:5].astype(np.float32)))
    (fig, axs) = plt.subplots(len(rows), 6, figsize = (6.6, 1.05 * len(rows) + 0.4),
                              squeeze = False)
    for (r, (lab, imgs)) in enumerate(rows):
        draw_image(axs[r][0], inp[p], a4[1, 0], a4[1, 1], 'input', half_width = 1.6)
        axs[r][0].set_ylabel(lab, fontsize = 5.5)
        for c in range(5):
            a = axs[r][c + 1]
            if c < len(imgs):
                name = (f'true quenching {c + 1}' if r == 0 else
                        f'draw {c + 1}' if len(imgs) > 1 else 'its one output')
                draw_image(a, imgs[c], a4[1, 0], a4[1, 1], name, half_width = 1.6)
            else:
                a.axis('off')
    fig.suptitle(f'Exercise {ex}: one fixed parent (cone {readout(inp[p:p + 1], jr.dr_of(a4[1:2]))["cone"][0]:.1f}'
                 ' GeV); each panel one image, no averaging; |phi - phi_jet| < 1.6, circles R = 0.4, 1.0',
                 fontsize = 6.5)
    fig.tight_layout(rect = (0, 0, 1, 0.94), h_pad = 0.2, w_pad = 0.2)
    fig.savefig(os.path.join(out, f'ex{ex}_displays.png'))
    plt.close(fig)
    if ue_case:
        fig_far(outs, inp, dr, out, plt)

def fig_far(outs, inp, dr, out, plt):
    """Exercise 4: far-region towers, output against input."""
    far = dr[:2000] >= 1.0
    (fig, axs) = plt.subplots(1, max(1, len(outs)), figsize = (1.9 * max(1, len(outs)), 2.0),
                              squeeze = False)
    for (a, (lab, x)) in zip(axs[0], outs.items()):
        xi = inp[:2000][far]
        xo = x[:2000][far]
        a.hexbin(xi, xo, gridsize = 40, extent = (0, 3, 0, 3), bins = 'log', cmap = 'Greys')
        a.plot([ 0, 3 ], [ 0, 3 ], color = PALETTE[0], lw = 0.6, ls = '--')
        a.set_title(f'{lab}: r = {np.corrcoef(xi, xo)[0, 1]:.3f}', fontsize = 6)
        a.set_xlabel('input tower, R > 1 (GeV)')
    axs[0][0].set_ylabel('output tower (GeV)')
    fig.suptitle('Exercise 4: the UE far from the jet, tower by tower (2000 events)', fontsize = 6.5)
    fig.tight_layout(rect = (0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, 'ex4_far.png'))
    plt.close(fig)

def main():
    cmdargs = parse_cmdargs()
    plt = setup()
    models = jr.specs(cmdargs.models)
    ex = cmdargs.exercise
    if ex == '1':
        fig_ex1(models, os.path.join(cmdargs.out, 'ex1'), plt)
    elif ex == '2':
        fig_shifts(os.path.join(cmdargs.out, 'ex2'), 'ex2', plt,
                   'Exercise 2: a frozen subtractor on the same jets, vacuum and quenched, '
                   'over the same UE (20k)')
    elif ex == '22':
        fig_shifts(os.path.join(cmdargs.out, 'prior_data'), 'prior_data', plt,
                   'Prior against data: the same 20k pairs, each cell a continuation of E1-P')
    else:
        os.makedirs(os.path.join(cmdargs.out, f'ex{ex}'), exist_ok = True)
        fig_translation(models, os.path.join(cmdargs.out, f'ex{ex}'), ex, plt)

if __name__ == '__main__':
    main()
