#!/usr/bin/env python
"""Repeated-sample analyses, reading and appendix tables of the stochastic
translation pilot (FLOW_NOTES.md, "Stochastic conditional FM pilot"), from
the outputs of translation_eval.py --generate (TRANSLATION_OUTPUTS) and the
population report of translation_report.py (--report, the same directory).

    condfm_report.py --models 'CondFM hard OT=CondFM-hard__midpoint128,...' \\
        --reference 'alpha-DSBM=alpha-DSBM__sde30' --det 'OT-FM=OT-CFM__midpoint128' \\
        --solver 'CondFM hard OT=solver_check_hard.csv,...' --report DIR

Repeated samples: the first 1000 PYTHIA test inputs (a random subset of the
test parents), STEM_multi8.npy (8 samples each, seeds 1000-1007), and for
the conditional runs STEM_swap.npy: input i's seed-1000 noise with the
condition of input perm[i] (a fixed cyclic shift, STEM_multi8.json). Every
statistic is bootstrapped over inputs (200 resamples), never over the 8
samples of one input. Shape distances are the normalised-shape EMD of the 53
cone towers (unit energy; translation_report.preference).

Written to --report:
    cf_spread.csv     per model and observable: the within-input sd (pooled
                      over inputs), the sd across inputs of the per-input
                      mean output, the input sd, the rms change from the
                      input, ratios; bootstrap sd
    cf_spread_corr.csv  correlations of the within-input fluctuations
    cf_distances.csv  shape EMD between two samples of one input, from a
                      sample to its own input, to an unrelated input, to the
                      deterministic translator's output; energy and tower
                      rms differences between two samples
    cf_swap.csv       the condition swap at fixed noise
    cf_tails.csv      per-sample tails against JEWEL test's 5% / 95%
                      quantiles (core fraction, z_lead, soft energy,
                      occupancy), single outputs and repeated samples
    cf_verdict.csv    the pre-registered reading, item by item
    cf_samples_*.png, cf_profiles.png, cf_spread.png
    slides/tables/cf_*.tex
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

import fm_common as fc
import translation_eval as te
import translation_report as trp
from closure_data import OFFSET
from jet_fidelity import EMD
from substructure import Geometry

N_MULTI = 1000
Q = [ 'E', 'mass', 'girth', 'ptd', 'zlead', 'zg', 'rg', 'lead', 'core' ]
SHAPE = [ 'girth', 'ptd', 'zlead', 'zg', 'rg', 'core' ]
COLOURS = { 'OT-FM' : '#2a78d6', 'hard' : '#1baf7a', 'soft' : '#eda100',
            'random JEWEL' : '#4a3aa7', 'JEWEL' : '#0b0b0b', 'PYTHIA' : '#898781' }
INK2 = '#52514e'

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Stochastic pilot: repeated samples')
    parser.add_argument('--models', required = True,
        help = "comma separated LABEL=STEM of the conditional runs")
    parser.add_argument('--reference', default = None,
        help = 'LABEL=STEM of another sampler\'s 8 samples (no swaps)')
    parser.add_argument('--det', required = True,
        help = 'LABEL=STEM of the deterministic translator')
    parser.add_argument('--solver', default = '',
        help = "comma separated LABEL=CSV of the models' solver checks (in --report)")
    parser.add_argument('--report', default = 'docs/flow/translation/condfm')
    parser.add_argument('--tables', default = None,
        help = 'LaTeX tables (default: REPORT/slides/tables)')
    parser.add_argument('--boot', type = int, default = 200)
    parser.add_argument('--procs', type = int, default = 30)
    return parser.parse_args()

def specs(text):
    return [ tuple(s.split('=', 1)) for s in text.split(',') if s ]

def arm(label):
    low = label.lower()
    return 'hard' if 'hard' in low else ('soft' if 'soft' in low else None)

# --- shape EMD, in parallel

_EMD = {}

def emd_rows(task):
    (key, start, stop) = task
    trp.use_all_cpus()
    torch.set_num_threads(1)
    (a, b) = _EMD[key]
    geo = Geometry(torch.device('cpu'))
    emd = EMD(geo)
    def unit(x):
        w = torch.as_tensor(x[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9]).float() \
            .clamp(min = 0) * geo.mask
        return w / w.sum((1, 2), keepdim = True).clamp(min = 1e-12)
    return emd(unit(a[start:stop]), unit(b[start:stop]))[0]

def shape_emds(pairs, procs):
    """{key: (A, B)} canvases (N, 16, 16) -> {key: (N,)} shape EMD of row
    pairs."""
    global _EMD            # pylint: disable=global-statement
    _EMD = pairs
    out = {}
    with Pool(procs, initializer = trp.use_all_cpus) as pool:
        for (key, (a, b)) in pairs.items():
            n = len(a)
            step = max(1, n // (procs * 4))
            tasks = [ (key, s, min(s + step, n)) for s in range(0, n, step) ]
            out[key] = np.concatenate(pool.map(emd_rows, tasks))
    print(f'{len(pairs)} EMD sets', flush = True)
    return out

# --- statistics

def boot_mean(per_input, boot, seed = 7):
    """Mean over inputs of a per-input quantity, and its bootstrap sd."""
    x = np.asarray(per_input, float)
    rng = np.random.default_rng(seed)
    bs = [ np.nanmean(x[rng.integers(0, len(x), len(x))]) for _ in range(boot) ]
    return (float(np.nanmean(x)), float(np.std(bs)))

def spread_table(sets, o_in, o_ref, boot):
    """Within-input against across-input variation, and against the size of
    the change; bootstrap over inputs."""
    rows = []
    rng = np.random.default_rng(8)
    n = len(o_in['E'])
    draws = [ rng.integers(0, n, n) for _ in range(boot) ]
    for (label, s) in sets.items():
        for q in Q:
            v = s['obs'][q]                                 # (K, n)
            x = np.asarray(o_in[q], float)
            def stats(sel, v = v, x = x):
                vv = v[:, sel]
                xx = x[sel]
                within = np.sqrt(np.nanmean(np.nanvar(vv, axis = 0, ddof = 1)))
                means  = np.nanmean(vv, axis = 0)
                across = np.nanstd(means)
                change = np.sqrt(np.nanmean((vv - xx[None])**2))
                mchange = np.sqrt(np.nanmean((means - xx)**2))
                return np.array([ within, across, change, mchange, within / across,
                                  within / change ])
            val = stats(np.arange(n))
            bs  = np.array([ stats(i) for i in draws ])
            sd_ref = float(np.nanstd(o_ref[q]))
            rows.append({ 'model' : label, 'observable' : q,
                          'within_sd' : val[0], 'within_sd_sd' : bs[:, 0].std(),
                          'across_sd' : val[1], 'input_sd' : float(np.nanstd(x)),
                          'jewel_sd' : sd_ref, 'within_over_jewel_sd' : val[0] / sd_ref,
                          'rms_change' : val[2], 'rms_mean_change' : val[3],
                          'within_over_across' : val[4], 'within_over_across_sd' : bs[:, 4].std(),
                          'within_over_change' : val[5], 'within_over_change_sd' : bs[:, 5].std() })
    return pd.DataFrame(rows)

def corr_table(sets):
    rows = []
    for (label, s) in sets.items():
        dev = { q : (s['obs'][q] - np.nanmean(s['obs'][q], axis = 0, keepdims = True)).ravel()
                for q in Q }
        c = pd.DataFrame(dev).corr()
        for a in Q:
            for b in Q:
                rows.append({ 'model' : label, 'a' : a, 'b' : b, 'r' : float(c.loc[a, b]) })
    return pd.DataFrame(rows)

def distance_pairs(sets, inp, det, other):
    """The canvas pairs of every distance, with the input of each pair."""
    k = None
    pairs, owners = {}, {}
    for (label, s) in sets.items():
        m = s['multi']
        k = len(m)
        idx = np.tile(np.arange(len(inp)), k)
        a = m.reshape(-1, 16, 16)
        pairs[(label, 'between_samples')] = (m[0::2].reshape(-1, 16, 16),
                                             m[1::2].reshape(-1, 16, 16))
        owners[(label, 'between_samples')] = np.tile(np.arange(len(inp)), k // 2)
        pairs[(label, 'to_own_input')] = (a, np.tile(inp, (k, 1, 1)))
        pairs[(label, 'to_unrelated_input')] = (a, np.tile(inp[other], (k, 1, 1)))
        pairs[(label, 'to_deterministic_output')] = (a, np.tile(det, (k, 1, 1)))
        pairs[(label, 'unrelated_samples')] = (m[0], m[1][other])
        for key in [ 'to_own_input', 'to_unrelated_input', 'to_deterministic_output' ]:
            owners[(label, key)] = idx
        owners[(label, 'unrelated_samples')] = np.arange(len(inp))
        if s.get('swap') is not None:
            perm = s['perm']
            sw = s['swap']
            for (key, b) in [ ('swap_to_condition_sample', m[0][perm]),
                              ('swap_to_noise_mate', m[0]),
                              ('swap_to_condition_input', inp[perm]),
                              ('swap_to_noise_input', inp) ]:
                pairs[(label, key)] = (sw, b)
                owners[(label, key)] = np.arange(len(inp))
    pairs[('deterministic', 'to_own_input')] = (det, inp)
    pairs[('deterministic', 'to_unrelated_input')] = (det, inp[other])
    pairs[('identity', 'to_unrelated_input')] = (inp, inp[other])
    for key in [ ('deterministic', 'to_own_input'), ('deterministic', 'to_unrelated_input'),
                 ('identity', 'to_unrelated_input') ]:
        owners[key] = np.arange(len(inp))
    return (pairs, owners)

def per_input_mean(values, owner, n):
    s = np.bincount(owner, weights = np.nan_to_num(values), minlength = n)
    c = np.bincount(owner, weights = np.isfinite(values).astype(float), minlength = n)
    return s / np.maximum(c, 1)

def distance_table(sets, emds, owners, det_label, n, boot):
    rows = []
    for ((label, kind), d) in emds.items():
        (mean, sd) = boot_mean(per_input_mean(d, owners[(label, kind)], n), boot)
        name = det_label if label == 'deterministic' else label
        rows.append({ 'model' : name, 'distance' : kind, 'shape_emd' : mean,
                      'shape_emd_sd' : sd, 'pairs' : int(np.isfinite(d).sum()) })
    for (label, s) in sets.items():
        m = s['multi']
        (a, b) = (m[0::2], m[1::2])
        e = s['obs']['E']
        de = (e[0::2] - e[1::2])
        rows.append({ 'model' : label, 'distance' : 'two_samples_E_rms_gev',
                      'shape_emd' : float(np.sqrt(np.mean(de**2))), 'shape_emd_sd' : np.nan,
                      'pairs' : int(de.size) })
        rows.append({ 'model' : label, 'distance' : 'two_samples_tower_rms_gev',
                      'shape_emd' : float(np.sqrt(np.mean((a - b)**2))), 'shape_emd_sd' : np.nan,
                      'pairs' : int(de.size) })
    return pd.DataFrame(rows)

def swap_table(sets, emds, o_in, boot):
    rows = []
    for (label, s) in sets.items():
        if s.get('swap') is None:
            continue
        perm = s['perm']
        d = { k : emds[(label, k)] for k in [ 'swap_to_condition_sample', 'swap_to_noise_mate',
                                              'swap_to_condition_input', 'swap_to_noise_input' ] }
        n = len(perm)
        row = { 'model' : label, 'n' : n, 'shift' : int(s['meta']['swap_shift']) }
        for (name, (a, b)) in { 'follows_condition_sample' : ('swap_to_condition_sample',
                                                              'swap_to_noise_mate'),
                                'follows_condition_input' : ('swap_to_condition_input',
                                                             'swap_to_noise_input') }.items():
            win = (d[a] < d[b]).astype(float)
            (p, sd) = boot_mean(win, boot)
            row[f'{name}_rate'] = p
            row[f'{name}_sd'] = sd
        for (k, v) in d.items():
            row[f'{k}_emd'] = float(np.nanmean(v))
        o_sw = s['obs_swap']
        obs = s['obs']
        for q in [ 'E', 'girth', 'zlead', 'ptd' ]:
            row[f'corr_{q}_condition_input'] = float(np.corrcoef(o_sw[q], o_in[q][perm])[0, 1])
            row[f'corr_{q}_noise_input'] = float(np.corrcoef(o_sw[q], o_in[q])[0, 1])
            # does one noise make the same fluctuation in another jet?
            mean = np.nanmean(obs[q], axis = 0)
            f_sw = o_sw[q] - mean[perm]
            f_mate = obs[q][0] - mean
            f_ctrl = obs[q][1][perm] - mean[perm]
            ok = np.isfinite(f_sw) & np.isfinite(f_mate) & np.isfinite(f_ctrl)
            row[f'fluct_corr_{q}_same_noise'] = float(np.corrcoef(f_sw[ok], f_mate[ok])[0, 1])
            row[f'fluct_corr_{q}_other_noise'] = float(np.corrcoef(f_ctrl[ok], f_mate[ok])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)

TAILS = [ ('core', 'low', 0.05), ('zlead', 'low', 0.05), ('soft_0p5', 'high', 0.95),
          ('n_0p01', 'high', 0.95) ]

def tail_table(sets, singles, o_ref, boot):
    rows = []
    cuts = { q : float(np.nanquantile(o_ref[q], p)) for (q, _, p) in TAILS }
    def share(o, q, side):
        v = np.asarray(o[q], float)
        return (v < cuts[q]) if side == 'low' else (v > cuts[q])
    for (label, o) in singles.items():
        row = { 'sample' : label, 'kind' : 'single output (20k)' }
        for (q, side, _) in TAILS:
            row[f'{q}_{side}'] = float(share(o, q, side).mean())
        rows.append(row)
    for (label, s) in sets.items():
        row = { 'sample' : label, 'kind' : '8 samples x 1000 inputs' }
        for (q, side, _) in TAILS:
            per = share(s['obs'], q, side).mean(axis = 0)       # per input
            (row[f'{q}_{side}'], row[f'{q}_{side}_sd']) = boot_mean(per, boot)
        rows.append(row)
    df = pd.DataFrame(rows)
    for (q, side, _) in TAILS:
        df[f'cut_{q}'] = cuts[q]
    return df

# --- the reading

def competitive(pop, label, ref = 'OT-FM'):
    p = pop[pop['sample'] == label].iloc[0]
    r = pop[pop['sample'] == ref].iloc[0]
    worse = [ q for q in te.OBS if p[f'w1_{q}'] - r[f'w1_{q}']
              > 3 * np.sqrt(p[f'w1_{q}_sd']**2 + r[f'w1_{q}_sd']**2) ]
    better = [ q for q in te.OBS if r[f'w1_{q}'] - p[f'w1_{q}']
               > 3 * np.sqrt(p[f'w1_{q}_sd']**2 + r[f'w1_{q}_sd']**2) ]
    return (worse, better)

def verdict(cmdargs, sets, pop, ver, dist, swap, tails, solver):
    rows = []
    for label in sets:
        if sets[label].get('swap') is None:
            continue
        v = ver[ver['model'] == label].iloc[0]
        (worse, better) = competitive(pop, label)
        t = tails[(tails['sample'] == label) & (tails['kind'] != 'single output (20k)')].iloc[0]
        sharp = bool((not v['soft_floor']) and (not v['flattening'])
                     and t['core_low'] <= 0.10 and t['soft_0p5_high'] <= 0.10)
        d = dist[dist['model'] == label].set_index('distance')
        between = d.loc['between_samples']
        sw = swap[swap['model'] == label].iloc[0]
        cond_used = bool(sw['follows_condition_sample_rate']
                         > 0.5 + 3 * sw['follows_condition_sample_sd'])
        sv = solver.get(label)
        if sv is not None:
            div_num = bool(sv['E_rms_over_two_noises'] < 0.1
                           and sv['tower_rms_over_two_noises'] < 0.1)
        else:
            div_num = False
        diverse = bool(div_num and between['shape_emd'] > 3 * between['shape_emd_sd'])
        joint_ok = bool(v['joint_girth_moved'] and v['joint_zlead_moved'])
        rows.append({
            'model' : label, 'marginals_moved' : int(v['n_moved']),
            'joints_moved' : joint_ok,
            'population_rule' : bool(v['n_moved'] >= 4 and joint_ok),
            'worse_than_ot_fm' : ', '.join(worse) or '-',
            'better_than_ot_fm' : ', '.join(better) or '-',
            'competitive_with_ot_fm' : not worse,
            'dependence_beyond_random' : bool(v['dependence_beyond_random']),
            'soft_floor' : bool(v['soft_floor']), 'flattening' : bool(v['flattening']),
            'core_low_share' : float(t['core_low']), 'soft_high_share' : float(t['soft_0p5_high']),
            'sharp_samples' : sharp,
            'solver_adequate' : bool(sv['adequate']) if sv is not None else False,
            'solver_E_over_two_noises' : float(sv['E_rms_over_two_noises']) if sv is not None else np.nan,
            'diversity_above_numerical' : diverse,
            'swap_follows_condition' : float(sw['follows_condition_sample_rate']),
            'conditioning_used' : cond_used,
            'useful' : bool(v['n_moved'] >= 4 and joint_ok and not worse
                            and v['dependence_beyond_random'] and sharp and diverse
                            and cond_used) })
    return pd.DataFrame(rows)

# --- figures

def plt_module():
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({ 'font.size' : 7, 'axes.edgecolor' : INK2, 'axes.labelcolor' : '#0b0b0b',
                          'xtick.color' : INK2, 'ytick.color' : INK2,
                          'axes.spines.top' : False, 'axes.spines.right' : False })
    return plt

def fig_samples(path, inp, det, multi, pick, label, det_label, qs):
    """Every sample of the display inputs, one panel each (no average)."""
    plt = plt_module()
    mask = Geometry(torch.device('cpu')).mask.numpy()
    cols = [ ('PYTHIA input', inp[pick]), (f'{det_label}\n(deterministic)', det[pick]) ]
    cols += [ (f'sample {k + 1}', multi[k][pick]) for k in range(len(multi)) ]
    (fig, axes) = plt.subplots(len(pick), len(cols), figsize = (1.05 * len(cols), 1.38 * len(pick)))
    geo = Geometry(torch.device('cpu'))
    core = (geo.dr < 0.15).numpy()
    for (c, (name, img)) in enumerate(cols):
        for r in range(len(pick)):
            ax = axes[r][c]
            w = img[r][OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * mask
            ax.imshow(np.log10(w + 0.1), cmap = 'viridis', vmin = -1, vmax = 1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            e = w.sum()
            ax.set_title(f'{e:.1f} GeV\nlead {w.max():.1f}, core {(w * core).sum() / max(e, 1e-9):.2f}',
                         fontsize = 4.8, pad = 2)
            if r == 0:
                ax.set_xlabel(name, fontsize = 5.5, labelpad = 12)
                ax.xaxis.set_label_position('top')
            if c == 0:
                ax.set_ylabel(f'q = {qs[r]}', fontsize = 5.5)
    fig.suptitle(f'{label}: each of 8 samples of five fixed held-out PYTHIA jets (energy'
                 ' quantiles q), log10(E + 0.1); no averaging', fontsize = 6.5)
    fig.subplots_adjust(left = 0.03, right = 0.995, top = 0.86, bottom = 0.01, wspace = 0.06,
                        hspace = 0.42)
    fig.savefig(path, dpi = 170)
    plt.close(fig)

def fig_samples_slide(path, inp, det, sets, pick, qs, det_label):
    """The appendix's display: three fixed inputs, OT-FM's output and the
    first three samples of each arm, every sample its own panel."""
    plt = plt_module()
    mask = Geometry(torch.device('cpu')).mask.numpy()
    arms = [ (label, s) for (label, s) in sets.items() if arm(label) ]
    cols = [ ('PYTHIA input', inp[pick], COLOURS['PYTHIA']),
             (det_label, det[pick], COLOURS['OT-FM']) ]
    for (label, s) in arms:
        cols += [ (f'{arm(label)} OT, sample {k + 1}', s['multi'][k][pick], COLOURS[arm(label)])
                  for k in range(3) ]
    (fig, axes) = plt.subplots(len(pick), len(cols), figsize = (0.92 * len(cols), 1.08 * len(pick)))
    for (c, (name, img, col)) in enumerate(cols):
        for r in range(len(pick)):
            ax = axes[r][c]
            w = img[r][OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * mask
            ax.imshow(np.log10(w + 0.1), cmap = 'viridis', vmin = -1, vmax = 1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(col)
                sp.set_linewidth(1.6)
            ax.set_title(f'{w.sum():.0f} GeV, lead {w.max():.1f}', fontsize = 5, pad = 1.5)
            if r == 0:
                ax.set_xlabel(name, fontsize = 5.3, labelpad = 9)
                ax.xaxis.set_label_position('top')
            if c == 0:
                ax.set_ylabel(f'q = {qs[r]}', fontsize = 5.5)
    fig.subplots_adjust(left = 0.035, right = 0.995, top = 0.84, bottom = 0.01, wspace = 0.08,
                        hspace = 0.32)
    fig.savefig(path, dpi = 220)
    plt.close(fig)

def radial(img):
    geo = Geometry(torch.device('cpu'))
    dr = geo.dr.numpy()
    mask = geo.mask.numpy()
    edges = np.array([ 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.42 ])
    w = img[..., OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * mask
    tot = w.sum((-2, -1))
    prof = np.stack([ (w * ((dr >= lo) & (dr < hi))).sum((-2, -1)) for (lo, hi)
                      in zip(edges[:-1], edges[1:]) ], -1) / np.maximum(tot, 1e-9)[..., None]
    return (0.5 * (edges[1:] + edges[:-1]), prof)

def fig_profiles(path, inp, det, sets, pick, qs, o_in, det_obs):
    """Radial profiles of every sample, and the leading tower against the
    core fraction, for the display inputs."""
    # pylint: disable=too-many-locals
    plt = plt_module()
    arms = [ (label, s) for (label, s) in sets.items() if arm(label) ]
    (fig, axes) = plt.subplots(len(arms) + 1, len(pick), figsize = (1.45 * len(pick), 3.9),
                               sharey = 'row')
    for (c, i) in enumerate(pick):
        for (r, (label, s)) in enumerate(arms):
            ax = axes[r][c]
            col = COLOURS[arm(label)]
            (x, p) = radial(s['multi'][:, i])
            for k in range(len(p)):
                ax.plot(x, p[k], color = col, lw = 0.7, alpha = 0.9,
                        ls = '-' if arm(label) == 'hard' else '--')
            (x, pi) = radial(inp[i])
            ax.plot(x, pi, color = COLOURS['PYTHIA'], lw = 1.2, ls = ':')
            (x, pd_) = radial(det[i])
            ax.plot(x, pd_, color = COLOURS['OT-FM'], lw = 1.2)
            ax.set_yscale('log')
            ax.set_ylim(3e-4, 1)
            ax.tick_params(labelsize = 5.5)
            if r == 0:
                ax.set_title(f'q = {qs[c]}: {o_in["E"][i]:.1f} GeV input', fontsize = 6)
            if c == 0:
                ax.set_ylabel(f'{label}\nenergy share per ring', fontsize = 5.5)
            if r == len(arms) - 1:
                ax.set_xlabel(r'$\Delta R$', fontsize = 6)
        ax = axes[-1][c]
        for (label, s) in arms:
            ax.scatter(s['obs']['lead'][:, i], s['obs']['core'][:, i], s = 9,
                       color = COLOURS[arm(label)],
                       marker = 'o' if arm(label) == 'hard' else '^', label = f'{label} samples',
                       edgecolor = 'white', linewidth = 0.4)
        ax.scatter([ o_in['lead'][i] ], [ o_in['core'][i] ], marker = 'x', color = COLOURS['PYTHIA'],
                   s = 18, label = 'PYTHIA input')
        ax.scatter([ det_obs['lead'][i] ], [ det_obs['core'][i] ], marker = 's',
                   color = COLOURS['OT-FM'], s = 12, label = 'OT-FM')
        ax.set_xlabel('leading tower, GeV', fontsize = 6)
        ax.tick_params(labelsize = 5.5)
        if c == 0:
            ax.set_ylabel(r'core fraction ($\Delta R < 0.15$)', fontsize = 5.5)
    (h, l) = axes[-1][0].get_legend_handles_labels()
    fig.legend(h, l, loc = 'lower center', ncol = len(l), fontsize = 5.5, frameon = False)
    fig.suptitle('Every sample separately: radial profiles (top rows; dotted grey: input,'
                 ' blue: OT-FM) and leading tower vs core (bottom)', fontsize = 6.5)
    fig.tight_layout(rect = (0, 0.05, 1, 0.97), h_pad = 0.4)
    fig.savefig(path, dpi = 200)
    plt.close(fig)

def fig_spread(path, spread, dist, sets, det_label):
    """Within-input sd over the across-input sd, per observable; shape
    distances."""
    plt = plt_module()
    (fig, axes) = plt.subplots(1, 2, figsize = (7.0, 2.3), gridspec_kw = { 'width_ratios' : [ 1.45, 1 ] })
    ax = axes[0]
    labels = list(sets)
    width = 0.8 / len(labels)
    for (k, label) in enumerate(labels):
        d = spread[spread['model'] == label].set_index('observable').loc[Q]
        col = COLOURS.get(arm(label), '#898781')
        x = np.arange(len(Q)) + (k - (len(labels) - 1) / 2) * width
        ax.bar(x, d['within_over_across'], width * 0.92, color = col, label = label,
               yerr = d['within_over_across_sd'], error_kw = { 'lw' : 0.6, 'ecolor' : INK2 },
               hatch = None if arm(label) else '////', edgecolor = 'white', linewidth = 0.4)
    ax.set_xticks(np.arange(len(Q)))
    ax.set_xticklabels([ trp.LABELS.get(q, q).replace(', GeV', '').replace(' (cone)', '')
                         for q in Q ], fontsize = 5.5, rotation = 30)
    ax.set_ylabel('within-input sd / sd across inputs', fontsize = 6)
    ax.legend(fontsize = 5.5, frameon = False)
    ax.set_title('Spread of one input\'s 8 samples, relative to the spread across inputs',
                 fontsize = 6.5)
    ax = axes[1]
    kinds = [ ('between_samples', 'two samples,\nsame input'),
              ('to_own_input', 'sample to\nits input'),
              ('to_deterministic_output', 'sample to\nOT-FM output'),
              ('to_unrelated_input', 'sample to an\nunrelated input') ]
    for (k, label) in enumerate(labels):
        d = dist[dist['model'] == label].set_index('distance')
        col = COLOURS.get(arm(label), '#898781')
        ys = [ d.loc[kk, 'shape_emd'] if kk in d.index else np.nan for (kk, _) in kinds ]
        es = [ d.loc[kk, 'shape_emd_sd'] if kk in d.index else np.nan for (kk, _) in kinds ]
        x = np.arange(len(kinds)) + (k - (len(labels) - 1) / 2) * width
        ax.bar(x, ys, width * 0.92, yerr = es, color = col, label = label,
               error_kw = { 'lw' : 0.6, 'ecolor' : INK2 },
               hatch = None if arm(label) else '////', edgecolor = 'white', linewidth = 0.4)
    d = dist[dist['model'] == det_label].set_index('distance')
    ax.axhline(d.loc['to_own_input', 'shape_emd'], color = COLOURS['OT-FM'], lw = 1.0,
               ls = '-', label = f'{det_label} output to its input')
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels([ t for (_, t) in kinds ], fontsize = 5)
    ax.set_ylabel('normalised-shape EMD', fontsize = 6)
    ax.legend(fontsize = 5.2, frameon = False)
    ax.set_title('Shape distances (mean over inputs)', fontsize = 6.5)
    for a in axes:
        a.tick_params(labelsize = 5.5)
    fig.tight_layout()
    fig.savefig(path, dpi = 200)
    plt.close(fig)

# --- tables

def tex_write(path, lines):
    with open(path, 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {path}')

BEGIN = lambda cols, sep = 3: [ r'\scriptsize', r'\setlength{\tabcolsep}{' + str(sep) + 'pt}',  # pylint: disable=unnecessary-lambda-assignment
                                r'\begin{tabular}{' + cols + '}', r'\toprule' ]
END = [ r'\bottomrule', r'\end{tabular}' ]
TEXNAME = { 'JEWEL ref' : 'JEWEL vs JEWEL', 'random JEWEL' : r'\keyrj random JEWEL jet',
            'identity' : 'PYTHIA input', 'OT-FM' : r'\keyot OT-FM (deterministic)',
            'alpha-DSBM' : r'$\alpha$-DSBM ($\epsilon$ 0.25)' }

def texname(s):
    if s in TEXNAME:
        return TEXNAME[s]
    if arm(s) == 'hard':
        return r'\keyhd cond.\ FM, hard OT'
    if arm(s) == 'soft':
        return r'\keysf cond.\ FM, soft OT'
    return s.replace('CycleGAN', 'CycleGAN,')

def fmt(v, d = 3):
    return '--' if (v is None) or (not np.isfinite(v)) else f'{v:.{d}f}'

def population_tex(pop, dep, samples):
    lines = BEGIN('l' + 'r' * 7 + '|rrr') + [
        r' & \multicolumn{7}{c|}{distance to held-out JEWEL, $W_1/\sigma$} & '
        r'\multicolumn{3}{c}{input dependence} \\',
        r' & $p_T$ & mass & girth & $p_T^D$ & $z_\mathrm{lead}$ & $z_g$ & $R_g$'
        r' & $r(p_T)$ & $r$(girth) & own \\', r'\midrule' ]
    for s in samples:
        p = pop[pop['sample'] == s].iloc[0]
        cells = [ fmt(p[f'w1_{q}']) for q in [ 'E', 'mass', 'girth', 'ptd', 'zlead', 'zg', 'rg' ] ]
        if s in set(dep['sample']):
            d = dep[dep['sample'] == s].iloc[0]
            cells += [ f"{d['corr_E']:.2f}", f"{d['corr_girth']:.2f}",
                       f"{d['preference_rate']:.2f}" ]
        else:
            cells += [ '', '', '' ]
        lines.append(texname(s) + ' & ' + ' & '.join(cells) + r' \\')
        if s == 'identity':
            lines.append(r'\midrule')
    return lines + END

def spread_tex(spread, labels, qs = ('E', 'mass', 'girth', 'ptd', 'zlead', 'lead', 'core')):
    names = { 'E' : '$p_T$', 'mass' : 'mass', 'girth' : 'girth', 'ptd' : '$p_T^D$',
              'zlead' : r'$z_\mathrm{lead}$', 'lead' : 'lead', 'core' : 'core', 'zg' : '$z_g$',
              'rg' : '$R_g$' }
    lines = BEGIN('l' + 'r' * len(qs)) + [
        r'sd of one input\textquotesingle s 8 samples / ... & '
        + ' & '.join(names[q] for q in qs) + r' \\', r'\midrule' ]
    for label in labels:
        d = spread[spread['model'] == label].set_index('observable')
        lines.append(r'\multicolumn{' + str(len(qs) + 1) + '}{l}{' + texname(label) + r'} \\')
        lines.append(r'\quad sd across inputs & ' + ' & '.join(
            f"{d.loc[q, 'within_over_across']:.2f}" for q in qs) + r' \\')
        lines.append(r'\quad rms change from input & ' + ' & '.join(
            f"{d.loc[q, 'within_over_change']:.2f}" for q in qs) + r' \\')
    return lines + END

def cost_tex(rep, labels):
    """Training cost and sampling latency: the flows from the report's
    cost.csv, alpha-DSBM from the pilot's, CycleGAN from its outputs' record."""
    def table(path):
        try:
            return pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame({ 'model' : [] })
    cost = table(os.path.join(rep, 'cost.csv'))
    pilot = table(os.path.join(os.path.dirname(os.path.normpath(rep)), 'cost.csv'))
    with open(os.path.join(rep, 'outputs.json'), encoding = 'utf-8') as f:
        info = json.load(f)
    lines = BEGIN('lrrrrrr') + [
        r' & GPU h & updates & updates/s & OT share & network calls & ms / jet \\', r'\midrule' ]
    for label in labels:
        i = info.get(label, {})
        lat = f"{i['ms_per_jet']:.2f}" if 'ms_per_jet' in i else '--'
        nfe = f"{int(i['nfe'])}" if 'nfe' in i else '--'
        c = cost[cost['model'] == label]
        if len(c):
            c = c.iloc[0]
            lines.append(f"{texname(label)} & {c['gpu_h']:.2f} & {c['updates'] / 1e3:.0f}k & "
                         f"{c['train_updates_per_s']:.1f} & {100 * c['coupling_share']:.0f}\\% & "
                         f"{nfe} & {lat}" + r' \\')
        elif (label == 'alpha-DSBM') and (pilot['model'] == 'alpha-DSBM').any():
            c = pilot[pilot['model'] == 'alpha-DSBM'].iloc[0]
            lines.append(f"{texname(label)} & {c['gpu_h']:.2f} & {c['updates'] / 1e3:.0f}k & "
                         f"-- & -- & {nfe} & {lat}" + r' \\')
        elif 'train_time' in i:
            lines.append(f"{texname(label)} & {i['train_time'] / 3600:.2f} & "
                         f"{i['updates'] / 1e3:.0f}k & -- & -- & {nfe} & {lat}" + r' \\')
    return lines + END

def obs_tex(names):
    tex = { 'E' : '$p_T$', 'zlead' : r'$z_\mathrm{lead}$', 'ptd' : '$p_T^D$', 'zg' : '$z_g$',
            'rg' : '$R_g$' }
    return ', '.join(tex.get(q, q) for q in names.split(', '))

def verdict_tex(verd):
    """The pre-registered reading, item by item, one column per arm."""
    yes = lambda v: r'\textbf{yes}' if v else 'no'      # pylint: disable=unnecessary-lambda-assignment
    labels = list(verd['model'])
    items = [
        ('moves toward JEWEL ($\\geq$ 4 of 7, both joints)',
         lambda v: f"{yes(v['population_rule'])} ({v['marginals_moved']} of 7)"),
        ('competitive with OT-FM (none worse by 3 sd)',
         lambda v: yes(v['competitive_with_ot_fm']) + ('' if v['worse_than_ot_fm'] == '-'
                                                       else f" (worse: {obs_tex(v['worse_than_ot_fm'])})")),
        ('input dependence beyond random JEWEL', lambda v: yes(v['dependence_beyond_random'])),
        ('sharp samples (no flag, tails $\\leq$ 10\\%)',
         lambda v: yes(v['sharp_samples']) + f" ({100 * v['core_low_share']:.0f}\\%,"
                                             f" {100 * v['soft_high_share']:.0f}\\%)"),
        ('solver resolved', lambda v: yes(v['solver_adequate'])),
        ('diversity above numerical error', lambda v: yes(v['diversity_above_numerical'])),
        ('conditioning used (swap)',
         lambda v: yes(v['conditioning_used']) + f" ({100 * v['swap_follows_condition']:.0f}\\%)"),
        ('\\textbf{useful (all of the above)}', lambda v: yes(v['useful'])),
    ]
    lines = BEGIN('l' + 'l' * len(labels)) + [
        ' & ' + ' & '.join(texname(l) for l in labels) + r' \\', r'\midrule' ]
    for (name, fn) in items:
        lines.append(name + ' & ' + ' & '.join(fn(verd[verd['model'] == l].iloc[0])
                                                for l in labels) + r' \\')
    return lines + END

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    device = torch.device('cpu')
    torch.set_num_threads(8)
    rep = cmdargs.report
    tables = cmdargs.tables or os.path.join(rep, 'slides', 'tables')
    os.makedirs(tables, exist_ok = True)
    outdir = te.output_dir()
    test = te.load_set('test', 'pythia')
    jt = te.load_set('test', 'jewel')
    n = N_MULTI
    inp = test['canvas'][:n].astype(np.float32)
    rows = test['row'][:n]
    o_in = te.jet_observables(inp, rows, device)
    o_ref = te.jet_observables(jt['canvas'], jt['row'], device)
    (det_label, det_stem) = cmdargs.det.split('=', 1)
    det = te.clip(np.load(os.path.join(outdir, f'{det_stem}.npy')))[:n]
    det_obs = te.jet_observables(det, rows, device)

    sets = {}
    for (label, stem) in specs(cmdargs.models):
        with open(os.path.join(outdir, f'{stem}_multi8.json'), encoding = 'utf-8') as f:
            meta = json.load(f)
        s = { 'multi' : te.clip(np.load(os.path.join(outdir, f'{stem}_multi8.npy'))),
              'swap' : te.clip(np.load(os.path.join(outdir, f'{stem}_swap.npy'))),
              'perm' : (np.arange(n) + meta['swap_shift']) % n, 'meta' : meta }
        sets[label] = s
    if cmdargs.reference:
        (label, stem) = cmdargs.reference.split('=', 1)
        sets[label] = { 'multi' : te.clip(np.load(os.path.join(outdir, f'{stem}_multi8.npy'))),
                        'swap' : None }
    for (label, s) in sets.items():
        per = [ te.jet_observables(x, rows, device) for x in s['multi'] ]
        s['obs'] = { q : np.stack([ np.asarray(p[q], float) for p in per ]) for q in per[0] }
        if s.get('swap') is not None:
            s['obs_swap'] = { q : np.asarray(v, float) for (q, v) in
                              te.jet_observables(s['swap'], rows, device).items() }
        print(f'observables: {label}', flush = True)

    spread = spread_table(sets, o_in, o_ref, cmdargs.boot)
    spread.to_csv(os.path.join(rep, 'cf_spread.csv'), index = False)
    corr_table(sets).to_csv(os.path.join(rep, 'cf_spread_corr.csv'), index = False)

    other = (np.arange(n) + 1 + int(np.random.default_rng(3).integers(n - 1))) % n
    (pairs, owners) = distance_pairs(sets, inp, det, other)
    emds = shape_emds(pairs, cmdargs.procs)
    dist = distance_table(sets, emds, owners, det_label, n, cmdargs.boot)
    dist.to_csv(os.path.join(rep, 'cf_distances.csv'), index = False)
    swap = swap_table(sets, emds, o_in, cmdargs.boot)
    swap.to_csv(os.path.join(rep, 'cf_swap.csv'), index = False)

    store = np.load(os.path.join(fc.translation_root(), 'report',
                                 f'{os.path.basename(os.path.normpath(rep))}.npz'))
    names = sorted({ k.split('|')[1] for k in store.files if k.startswith('crop|') })
    singles = { s : { q : store[f'crop|{s}|{q}'] for (q, _, _) in TAILS } for s in names }
    tails = tail_table(sets, singles, o_ref, cmdargs.boot)
    tails.to_csv(os.path.join(rep, 'cf_tails.csv'), index = False)

    pop = pd.read_csv(os.path.join(rep, 'population.csv'))
    dep = pd.read_csv(os.path.join(rep, 'dependence.csv'))
    ver = pd.read_csv(os.path.join(rep, 'verdict.csv'))
    solver = {}
    for (label, path) in specs(cmdargs.solver):
        sc = pd.read_csv(os.path.join(rep, path))
        rows_ = sc[sc['setting'].astype(str).str.contains('same noise')]
        solver[label] = rows_.iloc[-1] if len(rows_) else None
    verd = verdict(cmdargs, sets, pop, ver, dist, swap, tails, solver)
    verd.to_csv(os.path.join(rep, 'cf_verdict.csv'), index = False)

    qs = (0.1, 0.3, 0.5, 0.7, 0.9)
    pick = trp.display_indices(test, qs = qs)
    for (label, s) in sets.items():
        if arm(label):
            fig_samples(os.path.join(rep, f'cf_samples_{arm(label)}.png'), inp, det, s['multi'],
                        pick, label, det_label, qs)
    fig_profiles(os.path.join(rep, 'cf_profiles.png'), inp, det, sets, pick, qs, o_in, det_obs)
    fig_samples_slide(os.path.join(rep, 'cf_samples_slide.png'), inp, det, sets,
                      [ pick[0], pick[2], pick[4] ], (qs[0], qs[2], qs[4]), det_label)
    fig_spread(os.path.join(rep, 'cf_spread.png'), spread, dist, sets, det_label)

    samples = [ s for s in [ 'JEWEL ref', 'random JEWEL', 'identity', det_label ]
                + [ l for l in sets if arm(l) ] + [ l for l in pop['sample'] if
                                                   ('DSBM' in l or 'CycleGAN' in l) ] ]
    tex_write(os.path.join(tables, 'cf_population.tex'), population_tex(pop, dep, samples))
    tex_write(os.path.join(tables, 'cf_spread.tex'), spread_tex(spread, list(sets)))
    tex_write(os.path.join(tables, 'cf_verdict.tex'), verdict_tex(verd))
    tex_write(os.path.join(tables, 'cf_cost.tex'), cost_tex(rep, [ det_label ]
              + [ l for l in sets if arm(l) ] + [ l for l in pop['sample']
                                                  if ('DSBM' in l or 'CycleGAN' in l) ]))
    with pd.option_context('display.width', 250, 'display.max_columns', 60):
        print(spread.round(3).to_string(index = False))
        print(dist.round(4).to_string(index = False))
        print(swap.round(3).T.to_string())
        print(tails.round(3).to_string(index = False))
        print(verd.T.to_string())

if __name__ == '__main__':
    main()
