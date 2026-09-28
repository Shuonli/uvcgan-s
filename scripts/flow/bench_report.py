#!/usr/bin/env python
"""Tables and paper-style figures of the consolidated subtraction benchmark
(FLOW_NOTES.md, "Consolidated benchmark"), from the jets of bench_jets.py.

Definitions (bench_jets.py lists the jet settings):
- **Matching:** per event, one-to-one, greedily closest in dR first, within
  dR < 0.75 R, among jets above 5 GeV.
- **Efficiency:** the share of truth jets (|eta| < 0.6, pT_real bin) with a
  match.
- **Fake rate:** the share of reconstructed jets (|eta| < 0.6, pT_sub bin)
  without one.
- **Response:** pT_sub / pT_real of matched pairs (truth |eta| < 0.6), mean
  (scale) and standard deviation (resolution) per pT_real bin; also the
  bias and RMSE of pT_sub - pT_real in GeV.
- **Calibrated response** (labelled `_cal`, secondary): pT_sub = a + b
  pT_real fitted to the matched jets of val (PYTHIA) events 10000-19999
  (14-50 GeV) of the same run, frozen, and inverted on events 0-9999 of val
  and of JEWEL, where it is scored (on JEWEL it carries the domain shift).
- **Position:** d_eta = eta_sub - eta_real, RMS per bin.
- **Substructure:** R = 0.4, truth jets with 20 <= pT_real < 30 GeV,
  |eta| < 0.6. The truth distribution uses every such truth jet, a method's
  distribution its matched jets. Distribution agreement: the 1-D
  Wasserstein distance W1 to the truth distribution, in the observable's
  units (`_w1`; `_w1_floor` is W1 between the truth jets of even and odd
  events, the statistical floor). Per-jet agreement, from the same pairs:
  the bias and RMSE of reco - truth (`_pair_bias`, `_pair_rmse`).
- **The JEWEL - PYTHIA difference** of the mean of an observable in the same
  selection, as a share of the truth's (`_mod_fraction`).

Uncertainties: `_se` columns are statistical (binomial for efficiency and
fake rate, standard errors elsewhere) of one run on these events.
`bench_summary.csv` pools the seeds of an arm: `mean` and `half_range`
(max - min) / 2 over seeds, and `stat_se` the mean statistical error of one
run. Figures show the first seed with its statistical errors and, as a band,
the range of the seeds.

    bench_report.py --models 'LABEL=MODEL[+MODEL...],...' [--figure-models LABEL,...]
                    [--sets val,jewel] [--out docs/flow/bench]

A LABEL ending in ' [mid32]' is drawn in its base label's colour, dotted.
"""

import argparse
import json
import math
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from bench_jets import jet_dir

BINS     = np.arange(14, 52, 2)
RANGES   = [ (14, 20), (20, 30), (30, 50) ]
SUB_OBS  = [ 'zg', 'rg', 'girth', 'mass', 'zlead', 'xi' ]
PAIR_OBS = [ 'zg', 'rg', 'girth', 'mass', 'zlead', 'ptd' ]
CALIB_SPLIT = 10000
SUB_RANGE = { 'zg' : (0.1, 0.5), 'rg' : (0, 0.4), 'girth' : (0, 0.25),
              'mass' : (0, 12), 'zlead' : (0, 1), 'xi' : (0, 6),
              'ptd' : (0, 1) }
SUB_LABEL = { 'zg' : '$z_g$', 'rg' : '$r_g$', 'girth' : 'girth $g$',
              'mass' : 'mass, GeV', 'zlead' : '$z_{leading}$',
              'xi' : r'$\xi = \ln(p_T^{jet} / p_T^{const})$', 'ptd' : '$p_T^D$' }
SET_NAME = { 'val' : 'PYTHIA+HIJING (val)', 'jewel' : 'JEWEL+HIJING' }

plt.rcParams.update({ 'font.size' : 8, 'axes.titlesize' : 8,
                      'axes.labelsize' : 8, 'legend.fontsize' : 6.5,
                      'xtick.labelsize' : 7, 'ytick.labelsize' : 7 })

def load(set_name, model):
    z = np.load(os.path.join(jet_dir(set_name), f'{model}.npz'))
    return { k : z[k] for k in z.files }

def match(truth, reco, key, radius):
    """(truth index, reco index) of matched pairs over all events."""
    te = truth[f'{key}_event'].astype(int)
    re = reco[f'{key}_event'].astype(int)
    pairs = []
    t_by = pd.Series(np.arange(len(te))).groupby(te).apply(list).to_dict()
    r_by = pd.Series(np.arange(len(re))).groupby(re).apply(list).to_dict()
    for (ev, ti) in t_by.items():
        ri = r_by.get(ev)
        if not ri:
            continue
        ti = np.array(ti)
        ri = np.array(ri)
        deta = truth[f'{key}_eta'][ti][:, None] - reco[f'{key}_eta'][ri][None, :]
        dphi = np.abs(truth[f'{key}_phi'][ti][:, None]
                      - reco[f'{key}_phi'][ri][None, :])
        dphi = np.minimum(dphi, 2 * math.pi - dphi)
        dr = np.sqrt(deta**2 + dphi**2)
        used_t = set()
        used_r = set()
        for flat in np.argsort(dr, axis = None):
            (a, b) = np.unravel_index(flat, dr.shape)
            if dr[a, b] >= 0.75 * radius:
                break
            if a in used_t or b in used_r:
                continue
            used_t.add(a)
            used_r.add(b)
            pairs.append((ti[a], ri[b]))
    return np.array(pairs, dtype = int).reshape(-1, 2)

def acc(d, key):
    return np.abs(d[f'{key}_eta']) < 0.6

def binned(x, values, bins, fn):
    out = []
    for (lo, hi) in zip(bins[:-1], bins[1:]):
        sel = (x >= lo) & (x < hi)
        out.append(fn(values[sel]) if sel.sum() > 1 else (math.nan, math.nan))
    return np.array(out)

def mean_se(v):
    if len(v) < 2:
        return (math.nan, math.nan)
    return (float(np.mean(v)), float(np.std(v) / math.sqrt(len(v))))

def std_se(v):
    if len(v) < 2:
        return (math.nan, math.nan)
    s = float(np.std(v))
    return (s, s / math.sqrt(2 * (len(v) - 1)))

def rms_se(v):
    if len(v) < 2:
        return (math.nan, math.nan)
    r = float(np.sqrt(np.mean(v**2)))
    return (r, r / math.sqrt(2 * len(v)))

def ratio_se(k, n):
    p = k / n if n else math.nan
    return (p, math.sqrt(p * (1 - p) / n) if n else math.nan)

def put(out, name, value_se):
    (out[name], out[f'{name}_se']) = value_se

def jet_metrics(truth, reco, radius):
    """Per-bin response, position, efficiency and fake rate (figures), and
    their values in RANGES (tables)."""
    # pylint: disable=too-many-locals
    key   = f'R{int(radius * 10)}'
    pairs = match(truth, reco, key, radius)
    t_ok  = acc(truth, key)
    r_ok  = acc(reco, key)
    (ti, ri) = (pairs[:, 0], pairs[:, 1])
    in_acc  = t_ok[ti]
    (ti, ri) = (ti[in_acc], ri[in_acc])
    pt_real = truth[f'{key}_pt'][ti]
    pt_sub  = reco[f'{key}_pt'][ri]
    resp    = pt_sub / pt_real
    deta    = reco[f'{key}_eta'][ri] - truth[f'{key}_eta'][ti]
    event   = truth[f'{key}_event'][ti]

    matched_t = np.zeros(len(t_ok), bool)
    matched_t[pairs[:, 0]] = True
    matched_r = np.zeros(len(r_ok), bool)
    matched_r[pairs[:, 1]] = True
    t_pt = truth[f'{key}_pt']
    r_pt = reco[f'{key}_pt']

    def eff_fake(lo, hi):
        sel_t = t_ok & (t_pt >= lo) & (t_pt < hi)
        sel_r = r_ok & (r_pt >= lo) & (r_pt < hi)
        return (ratio_se(int((sel_t & matched_t).sum()), int(sel_t.sum())),
                ratio_se(int((sel_r & ~matched_r).sum()), int(sel_r.sum())))

    per_bin = [ eff_fake(lo, hi) for (lo, hi) in zip(BINS[:-1], BINS[1:]) ]
    out = {
        'scale' : binned(pt_real, resp, BINS, mean_se),
        'resolution' : binned(pt_real, resp, BINS, std_se),
        'deta_rms' : binned(pt_real, deta, BINS, rms_se),
        'efficiency' : np.array([ e for (e, _) in per_bin ]),
        'fake' : np.array([ f for (_, f) in per_bin ]),
        'resp_30_32' : resp[(pt_real >= 30) & (pt_real < 32)],
        'deta_30_40' : deta[(pt_real >= 30) & (pt_real < 40)],
    }
    scalars = {}
    for (lo, hi) in RANGES:
        sel = (pt_real >= lo) & (pt_real < hi)
        d   = pt_sub[sel] - pt_real[sel]
        put(scalars, f'scale_{lo}_{hi}', mean_se(resp[sel]))
        put(scalars, f'resolution_{lo}_{hi}', std_se(resp[sel]))
        put(scalars, f'bias_gev_{lo}_{hi}', mean_se(d))
        put(scalars, f'rmse_gev_{lo}_{hi}', rms_se(d))
        (e, f) = eff_fake(lo, hi)
        put(scalars, f'eff_{lo}_{hi}', e)
        put(scalars, f'fake_{lo}_{hi}', f)
    put(scalars, 'deta_rms_20_50', rms_se(deta[(pt_real >= 20) & (pt_real < 50)]))
    out['scalars'] = scalars
    # for the frozen calibration (calibrated())
    out['matched'] = (pt_real, pt_sub, event)
    return out

def calibration(jets):
    """(a, b) of pT_sub = a + b pT_real, fitted to the matched jets of
    events >= CALIB_SPLIT (14-50 GeV)."""
    (pt_real, pt_sub, event) = jets['matched']
    fit = (event >= CALIB_SPLIT) & (pt_real >= 14) & (pt_real < 50)
    (b, a) = np.polyfit(pt_real[fit], pt_sub[fit], 1)
    return (float(a), float(b))

def calibrated(jets, cal):
    """The response after inverting the frozen calibration `cal`, on events
    < CALIB_SPLIT."""
    (pt_real, pt_sub, event) = jets['matched']
    (a, b) = cal
    pt_cal = (pt_sub - a) / b
    out = { 'calib_a' : a, 'calib_b' : b }
    for (lo, hi) in RANGES[1:]:
        sel = (event < CALIB_SPLIT) & (pt_real >= lo) & (pt_real < hi)
        put(out, f'scale_cal_{lo}_{hi}', mean_se(pt_cal[sel] / pt_real[sel]))
        put(out, f'resolution_cal_{lo}_{hi}', std_se(pt_cal[sel] / pt_real[sel]))
        put(out, f'bias_cal_gev_{lo}_{hi}', mean_se(pt_cal[sel] - pt_real[sel]))
    return out

def sub_values(d, idx, obs):
    if obs == 'xi':
        xi = d['R4_xi']
        return xi[np.isin(xi[:, 0].astype(int), idx), 1]
    v = d[f'R4_{obs}'][idx]
    return v[np.isfinite(v)]

def sub_metrics(truth, reco):
    """Substructure of R = 0.4 jets with 20 <= pT_real < 30 GeV."""
    pairs = match(truth, reco, 'R4', 0.4)
    t_ok  = acc(truth, 'R4') & (truth['R4_pt'] >= 20) & (truth['R4_pt'] < 30)
    t_sel = np.nonzero(t_ok)[0]
    keep  = t_ok[pairs[:, 0]]
    (ti, ri) = (pairs[keep, 0], pairs[keep, 1])
    out = { 'hist' : {}, 'scalars' : { 'sub_truth_jets' : len(t_sel),
                                       'sub_matched' : len(ti) } }
    sc = out['scalars']
    for obs in SUB_OBS + [ 'ptd' ]:
        t_all = sub_values(truth, t_sel, obs)
        r_val = sub_values(reco, ri, obs)
        out['hist'][obs] = (t_all, r_val)
        put(sc, f'{obs}_mean_truth', mean_se(t_all))
        sc[f'{obs}_truth_all_sd'] = float(np.std(t_all))
        put(sc, f'{obs}_mean_reco', mean_se(r_val))
        sc[f'{obs}_w1'] = float(wasserstein_distance(r_val, t_all))
        if obs == 'xi':
            continue
        a = reco[f'R4_{obs}'][ri]
        b = truth[f'R4_{obs}'][ti]
        ok = np.isfinite(a) & np.isfinite(b)
        put(sc, f'{obs}_pair_bias', mean_se(a[ok] - b[ok]))
        put(sc, f'{obs}_pair_rmse', rms_se(a[ok] - b[ok]))
        sc[f'{obs}_truth_sd'] = float(np.std(b[ok]))
        # soft drop found a split in one jet of the pair but not the other
        sc[f'{obs}_pair_undefined'] = float(1 - ok.mean())
    return out

def truth_floor(truth):
    """W1 between the truth jets of even and odd events."""
    t_ok = acc(truth, 'R4') & (truth['R4_pt'] >= 20) & (truth['R4_pt'] < 30)
    ev   = truth['R4_event'].astype(int)
    out  = {}
    for obs in SUB_OBS + [ 'ptd' ]:
        a = sub_values(truth, np.nonzero(t_ok & (ev % 2 == 0))[0], obs)
        b = sub_values(truth, np.nonzero(t_ok & (ev % 2 == 1))[0], obs)
        out[f'{obs}_w1_floor'] = float(wasserstein_distance(a, b))
    return out

def evaluate(task):
    (set_name, _label, _seed, model) = task
    truth = load(set_name, 'truth')
    reco  = load(set_name, model)
    res   = { 'jets' : { r : jet_metrics(truth, reco, r) for r in (0.2, 0.4, 0.5) } }
    if model != 'area':
        res['sub'] = sub_metrics(truth, reco)
    return (task, res)

STYLE = {}

def base_label(label):
    return label.split(' [')[0]

def style(label, **extra):
    st = dict(STYLE.get(base_label(label), {}))
    if label.endswith('[mid32]'):
        st['ls'] = ':'
    st.update(extra)
    return st

def band(ax, x, values, label):
    """Seed range as a band, if more than one seed."""
    if len(values) < 2:
        return
    v = np.array(values, dtype = float)
    with np.errstate(all = 'ignore'):
        ax.fill_between(x, np.nanmin(v, axis = 0), np.nanmax(v, axis = 0),
                        color = style(label).get('color', 'k'), alpha = 0.18, lw = 0)

def curve_panel(ax, res, s, labels, r, key):
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    for label in labels:
        runs = res[s].get(label)
        if not runs:
            continue
        v = runs[0]['jets'][r][key]
        ax.errorbar(centers, v[:, 0], yerr = v[:, 1], label = label, capsize = 0,
                    lw = 1.1, marker = '.', ms = 2.5, **style(label))
        band(ax, centers, [ x['jets'][r][key][:, 0] for x in runs ], label)
    ax.grid(alpha = 0.3)

def dist_panel(ax, res, s, labels, r, key, bins):
    c = 0.5 * (bins[:-1] + bins[1:])
    for label in labels:
        runs = res[s].get(label)
        if not runs:
            continue
        (h, _) = np.histogram(runs[0]['jets'][r][key], bins, density = True)
        ax.step(c, h, where = 'mid', lw = 1.1, label = label, **style(label))
    ax.grid(alpha = 0.3)

def fig3(res, s, labels, out):
    (fig, axes) = plt.subplots(1, 2, figsize = (6.4, 2.5))
    dist_panel(axes[0], res, s, labels, 0.5, 'deta_30_40', np.linspace(-0.3, 0.3, 41))
    axes[0].set_xlabel('$\\Delta\\eta = \\eta^{sub} - \\eta^{real}$')
    axes[0].set_title('30 < $p_T^{real}$ < 40 GeV')
    curve_panel(axes[1], res, s, labels, 0.5, 'deta_rms')
    axes[1].set_xlabel('$p_T^{real}$, GeV')
    axes[1].set_ylabel('RMS of $\\Delta\\eta$')
    axes[0].legend()
    fig.suptitle(f'Fig. 3: jet position, R = 0.5, {SET_NAME[s]}')
    fig.tight_layout()
    fig.savefig(out, dpi = 150)
    plt.close(fig)

def fig4(res, s, labels, out, radii = (0.2, 0.5)):
    (fig, axes) = plt.subplots(len(radii), 3, figsize = (6.6, 2.2 * len(radii)),
                               squeeze = False)
    for (k, r) in enumerate(radii):
        dist_panel(axes[k][0], res, s, labels, r, 'resp_30_32', np.linspace(0, 2, 41))
        axes[k][0].set_xlabel('$p_T^{sub} / p_T^{real}$')
        axes[k][0].set_title(f'R = {r}, 30 < $p_T^{{real}}$ < 32 GeV')
        curve_panel(axes[k][1], res, s, labels, r, 'scale')
        axes[k][1].axhline(1, color = 'k', lw = 0.7)
        axes[k][1].set_ylabel('mean $p_T^{sub} / p_T^{real}$')
        axes[k][1].set_title(f'R = {r}: scale')
        curve_panel(axes[k][2], res, s, labels, r, 'resolution')
        axes[k][2].set_ylabel('std $p_T^{sub} / p_T^{real}$')
        axes[k][2].set_title(f'R = {r}: resolution')
        for a in axes[k][1:]:
            a.set_xlabel('$p_T^{real}$, GeV')
    axes[0][0].legend(fontsize = 5.5)
    fig.suptitle(f'Fig. 4: jet $p_T$ response (raw), {SET_NAME[s]}')
    fig.tight_layout()
    fig.savefig(out, dpi = 150)
    plt.close(fig)

def fig5(res, s, labels, out, radii = (0.5,)):
    (fig, axes) = plt.subplots(len(radii), 2, figsize = (6.4, 2.5 * len(radii)),
                               squeeze = False)
    for (k, r) in enumerate(radii):
        curve_panel(axes[k][0], res, s, labels, r, 'efficiency')
        axes[k][0].set_xlabel('$p_T^{real}$, GeV')
        axes[k][0].set_ylabel('efficiency')
        axes[k][0].set_title(f'R = {r}: efficiency')
        curve_panel(axes[k][1], res, s, labels, r, 'fake')
        axes[k][1].set_xlabel('$p_T^{sub}$, GeV')
        axes[k][1].set_ylabel('fake rate')
        axes[k][1].set_title(f'R = {r}: fake rate')
        axes[k][1].set_yscale('log')
    axes[0][0].legend()
    fig.suptitle(f'Fig. 5: efficiency and fake rate, {SET_NAME[s]}')
    fig.tight_layout()
    fig.savefig(out, dpi = 150)
    plt.close(fig)

def slide_jets(res, s, labels, out, r = 0.4):
    """Scale, resolution, efficiency and fake rate at one radius, one row."""
    (fig, axes) = plt.subplots(1, 4, figsize = (6.8, 2.1))
    for (ax, key, ylabel) in zip(axes, [ 'scale', 'resolution', 'efficiency', 'fake' ],
                                 [ 'mean $p_T^{sub}/p_T^{real}$',
                                   'std $p_T^{sub}/p_T^{real}$', 'efficiency',
                                   'fake rate' ]):
        curve_panel(ax, res, s, labels, r, key)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('$p_T^{sub}$, GeV' if key == 'fake' else '$p_T^{real}$, GeV')
    axes[0].axhline(1, color = 'k', lw = 0.7)
    axes[3].set_yscale('log')
    axes[2].set_ylim(top = 1.005)
    (handles, names) = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc = 'upper center', ncol = len(names),
               fontsize = 5.5, frameon = False)
    fig.tight_layout(rect = (0, 0, 1, 0.9))
    fig.savefig(out, dpi = 170)
    plt.close(fig)

def fig6(res, s, labels, out, obs_list = SUB_OBS):
    # pylint: disable=too-many-locals
    (fig, axes) = plt.subplots(2, len(obs_list),
                               figsize = (1.75 * len(obs_list) + 0.4, 3.3),
                               squeeze = False, sharex = 'col',
                               gridspec_kw = { 'height_ratios' : [ 2, 1 ] })
    for (col, obs) in enumerate(obs_list):
        (ax, axr) = (axes[0][col], axes[1][col])
        bins = np.linspace(*SUB_RANGE[obs], 21)
        c = 0.5 * (bins[:-1] + bins[1:])
        first = next(r for r in res[s].values() if 'sub' in r[0])[0]['sub']
        (h_t, _) = np.histogram(first['hist'][obs][0], bins, density = True)
        ax.step(c, h_t, where = 'mid', color = 'k', lw = 1.8, label = 'real (truth)')
        for label in labels:
            runs = res[s].get(label)
            if not runs or 'sub' not in runs[0]:
                continue
            ratios = []
            for (i, run) in enumerate(runs):
                (h, _) = np.histogram(run['sub']['hist'][obs][1], bins, density = True)
                with np.errstate(all = 'ignore'):
                    ratios.append(h / h_t)
                if i == 0:
                    ax.step(c, h, where = 'mid', lw = 1.0, label = label, **style(label))
                    axr.plot(c, ratios[0], lw = 1.0, **style(label))
            band(axr, c, ratios, label)
        axr.axhline(1, color = 'k', lw = 0.7)
        axr.set_ylim(0, 2)
        ax.set_title(SUB_LABEL[obs])
        axr.set_xlabel(SUB_LABEL[obs])
        for a in (ax, axr):
            a.grid(alpha = 0.3)
    axes[1][0].set_ylabel('ratio to real')
    axes[0][0].set_ylabel('normalised')
    axes[0][0].legend(fontsize = 5)
    fig.suptitle(f'Fig. 6: substructure, R = 0.4, 20 < $p_T^{{real}}$ < 30 GeV, '
                 f'{SET_NAME[s]} (ratio band: seed range)')
    fig.tight_layout()
    fig.savefig(out, dpi = 150)
    plt.close(fig)

def fig7(res, labels, out):
    """Girth and z_leading, PYTHIA (val) against JEWEL: the distributions of
    the truth and UVCGAN-S (as the paper's Fig. 7), and the JEWEL/PYTHIA
    ratio of every method against the truth's."""
    # pylint: disable=too-many-locals
    (fig, axes) = plt.subplots(2, 2, figsize = (6.4, 4.0), sharex = 'col',
                               gridspec_kw = { 'height_ratios' : [ 1.3, 1 ] })
    for (col, obs) in enumerate([ 'girth', 'zlead' ]):
        bins = np.linspace(*SUB_RANGE[obs], 21)
        c = 0.5 * (bins[:-1] + bins[1:])
        hist = {}
        for s in ('val', 'jewel'):
            first = next(r for r in res[s].values() if 'sub' in r[0])[0]['sub']
            hist[('truth', s)] = np.histogram(first['hist'][obs][0], bins,
                                              density = True)[0]
            for label in labels:
                runs = res[s].get(label)
                if runs and 'sub' in runs[0]:
                    hist[(label, s)] = [ np.histogram(x['sub']['hist'][obs][1], bins,
                                                      density = True)[0] for x in runs ]
        ax = axes[0][col]
        for (s, ls) in [ ('val', '-'), ('jewel', '--') ]:
            name = 'PYTHIA' if s == 'val' else 'JEWEL'
            ax.step(c, hist[('truth', s)], where = 'mid', color = 'k', ls = ls,
                    lw = 1.6, label = f'real, {name}')
            if ('UVCGAN-S', s) in hist:
                ax.step(c, hist[('UVCGAN-S', s)][0], where = 'mid', lw = 1.1,
                        label = f'UVCGAN-S, {name}', **style('UVCGAN-S', ls = ls))
        ax.set_title(f'{SUB_LABEL[obs]}: distributions')
        ax.grid(alpha = 0.3)
        axr = axes[1][col]
        with np.errstate(all = 'ignore'):
            axr.plot(c, hist[('truth', 'jewel')] / hist[('truth', 'val')], color = 'k',
                     lw = 1.8, label = 'real')
            for label in labels:
                if (label, 'val') not in hist or (label, 'jewel') not in hist:
                    continue
                rr = [ j / v for (j, v) in zip(hist[(label, 'jewel')],
                                               hist[(label, 'val')]) ]
                axr.plot(c, rr[0], lw = 1.0, label = label, **style(label))
                band(axr, c, rr, label)
        axr.axhline(1, color = 'k', lw = 0.6)
        axr.set_yscale('log')
        axr.set_ylim(0.2, 5)
        axr.set_xlabel(SUB_LABEL[obs])
        axr.set_ylabel('JEWEL / PYTHIA')
        axr.grid(alpha = 0.3)
    axes[0][0].legend(fontsize = 5.5)
    axes[1][1].legend(fontsize = 5)
    fig.suptitle('Fig. 7: PYTHIA against JEWEL, R = 0.4, 20 < $p_T^{real}$ < 30 GeV')
    fig.tight_layout()
    fig.savefig(out, dpi = 150)
    plt.close(fig)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Benchmark report')
    parser.add_argument('--models', required = True,
        help = "LABEL=MODEL[+MODEL...],... (MODEL as in bench_jets.py, '+'"
               " joins the seeds of an arm)")
    parser.add_argument('--figure-models', default = None,
        help = 'LABEL,... shown in the figures (default: all)')
    parser.add_argument('--sets', default = 'val,jewel')
    parser.add_argument('--procs', type = int, default = 16)
    parser.add_argument('--out', default = 'docs/flow/bench')
    return parser.parse_args()

def summarise(df):
    """Seeds pooled: mean, half range and the mean statistical error."""
    rows = []
    metrics = [ c for c in df.columns if c not in ('set', 'label', 'model', 'seed', 'R')
                and not c.endswith('_se') ]
    for ((s, label, r), g) in df.groupby([ 'set', 'label', 'R' ], sort = False):
        for m in metrics:
            v = g[m].astype(float)
            if v.isna().all():
                continue
            se = g[f'{m}_se'].astype(float).mean() if f'{m}_se' in g else math.nan
            rows.append({ 'set' : s, 'label' : label, 'R' : r, 'metric' : m,
                          'mean' : v.mean(), 'half_range' : (v.max() - v.min()) / 2,
                          'stat_se' : se, 'n_seeds' : int(v.notna().sum()) })
    return pd.DataFrame(rows)

def mod_fractions(df, labels):
    rows = []
    for label in labels:
        for seed in sorted(df[df.label == label].seed.unique()):
            a = df[(df.set == 'val') & (df.label == label) & (df.seed == seed) & (df.R == 0.4)]
            b = df[(df.set == 'jewel') & (df.label == label) & (df.seed == seed) & (df.R == 0.4)]
            if not (len(a) and len(b)) or 'girth_mean_reco' not in a \
                    or a['girth_mean_reco'].isna().all():
                continue
            row = { 'set' : 'jewel-val', 'label' : label, 'model' : a.model.iloc[0],
                    'seed' : seed, 'R' : 0.4 }
            for obs in SUB_OBS + [ 'ptd' ]:
                dm = float(b[f'{obs}_mean_reco'].iloc[0] - a[f'{obs}_mean_reco'].iloc[0])
                dt = float(b[f'{obs}_mean_truth'].iloc[0] - a[f'{obs}_mean_truth'].iloc[0])
                sm = math.hypot(b[f'{obs}_mean_reco_se'].iloc[0],
                                a[f'{obs}_mean_reco_se'].iloc[0])
                st = math.hypot(b[f'{obs}_mean_truth_se'].iloc[0],
                                a[f'{obs}_mean_truth_se'].iloc[0])
                f = dm / dt if dt else math.nan
                row[f'{obs}_mod_fraction'] = f
                row[f'{obs}_mod_fraction_se'] = abs(f) * math.hypot(
                    sm / dm if dm else 0, st / dt if dt else 0)
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    # pylint: disable=too-many-locals,too-many-branches
    cmdargs = parse_cmdargs()
    models  = {}
    for item in cmdargs.models.split(','):
        (label, spec) = item.split('=', 1)
        models[label] = spec.split('+')
    sets = cmdargs.sets.split(',')
    fig_labels = cmdargs.figure_models.split(',') if cmdargs.figure_models \
        else list(models)
    style_path = os.path.join(os.path.dirname(__file__), 'bench_style.json')
    if os.path.exists(style_path):
        STYLE.update(json.load(open(style_path, encoding = 'utf-8')))
    os.makedirs(cmdargs.out, exist_ok = True)

    tasks = [ (s, label, seed, m) for s in sets for (label, ms) in models.items()
              for (seed, m) in enumerate(ms) ]
    with Pool(cmdargs.procs) as pool:
        done = dict(pool.map(evaluate, tasks))

    res  = { s : {} for s in sets }
    rows = []
    for task in tasks:
        (s, label, seed, m) = task
        r = done[task]
        res[s].setdefault(label, []).append(r)
        for radius in (0.2, 0.4, 0.5):
            # the calibration of the same run on val (PYTHIA) events
            # 10000-19999, frozen: on JEWEL it carries the domain shift
            ref = done.get(('val', label, seed, m))
            cal = calibrated(r['jets'][radius], calibration(ref['jets'][radius])) \
                if ref is not None else {}
            row = { 'set' : s, 'label' : label, 'model' : m, 'seed' : seed,
                    'R' : radius, **r['jets'][radius]['scalars'], **cal }
            if radius == 0.4 and 'sub' in r:
                row.update(r['sub']['scalars'])
            rows.append(row)
    df = pd.DataFrame(rows)
    for s in sets:
        for (k, v) in truth_floor(load(s, 'truth')).items():
            df.loc[(df.set == s) & (df.R == 0.4), k] = v
    if set(sets) >= { 'val', 'jewel' }:
        df = pd.concat([ df, mod_fractions(df, list(models)) ], ignore_index = True)
    df.to_csv(os.path.join(cmdargs.out, 'bench_jets.csv'), index = False)
    summary = summarise(df)
    summary.to_csv(os.path.join(cmdargs.out, 'bench_summary.csv'), index = False)

    sub_labels = [ l for l in fig_labels if models[l] != [ 'area' ] ]
    for s in sets:
        fig3(res, s, fig_labels, os.path.join(cmdargs.out, f'bench_fig3_{s}.png'))
        fig4(res, s, fig_labels, os.path.join(cmdargs.out, f'bench_fig4_{s}.png'))
        fig5(res, s, fig_labels, os.path.join(cmdargs.out, f'bench_fig5_{s}.png'))
        fig5(res, s, fig_labels, os.path.join(cmdargs.out, f'bench_fig5_allR_{s}.png'),
             radii = (0.2, 0.4, 0.5))
        fig6(res, s, sub_labels, os.path.join(cmdargs.out, f'bench_fig6_{s}.png'))
        slide_jets(res, s, fig_labels, os.path.join(cmdargs.out, f'bench_jets_R04_{s}.png'))
    if set(sets) >= { 'val', 'jewel' }:
        fig7(res, sub_labels, os.path.join(cmdargs.out, 'bench_fig7.png'))

    show = [ 'scale_20_30', 'resolution_20_30', 'rmse_gev_20_30', 'resolution_cal_20_30',
             'eff_14_20', 'fake_14_20', 'fake_20_30', 'girth_pair_rmse', 'zg_w1' ]
    t = summary[(summary.R == 0.4) & summary.metric.isin(show)]
    t = t.assign(v = [ f'{m:.3f}+-{h:.3f}' for (m, h) in zip(t['mean'], t.half_range) ])
    with pd.option_context('display.width', 250, 'display.max_columns', 40,
                           'display.max_rows', 200):
        print(t.pivot_table(index = [ 'set', 'label' ], columns = 'metric', values = 'v',
                            aggfunc = 'first', sort = False)
              .reindex(columns = show).to_string())
    print(f'wrote {cmdargs.out}/bench_jets.csv, bench_summary.csv and figures')

if __name__ == '__main__':
    main()
