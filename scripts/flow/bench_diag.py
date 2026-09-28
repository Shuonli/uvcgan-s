#!/usr/bin/env python
"""Per-tower diagnostics and event displays of the consolidated subtraction
benchmark (FLOW_NOTES.md, "Consolidated benchmark"), from the images of
bench_images.py (raw outputs, nothing clipped unless stated).

Towers (`--towers`, signal readouts; '+' joins the seeds of an arm):
- MAE and RMSE of S_hat - S over all towers, GeV (`_clip0`: S_hat clipped
  at 0 first, a labelled clean-up);
- the event energy bias, mean of sum(S_hat - S), GeV per event;
- the background error against the true signal energy of the tower: the
  implied background B_hat = M - S_hat (for one-panel flows and the joint
  arm's M - B_hat readout exactly its background output) minus B = M - S,
  i.e. S - S_hat. Towers are grouped as S = 0, (0, 0.5], (0.5, 2], (2, 5],
  (5, 10] and > 10 GeV; per group the mean and RMS of B_hat - B, and the
  share of the group's true signal energy put into the background,
  sum(B_hat - B) / sum(S). A positive share at large S: the jet core goes
  into the background.
Consistency (`--consistency`, joint runs): B_hat + S_hat - M of the two
outputs, before any clean-up: per-tower mean, MAE and RMS, and the event
sum.
Displays (`--display`): fixed events, the first two of each set whose
leading truth jet has 25-35 GeV in its R = 0.4 cone (acceptance as the
cone scores). Per event the mixture, the truth and each model's S_hat
(log10(E + 0.1), negative towers shown as 0), and below S_hat - S (GeV),
in a 20-tower window in phi around the jet, the full eta range.
`bench_displays.png` holds the first event of each set,
`bench_displays_more.png` the second.

    bench_diag.py [--display LABEL=STEM,...] [--towers LABEL=STEM+STEM,...]
                  [--consistency LABEL=RUN__SETTING+...,...] [--out docs/flow/bench]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch

import fm_common as fc

ev = fc.ev
S_BINS = [ ('S = 0', 0, 0), ('0-0.5', 0, 0.5), ('0.5-2', 0.5, 2),
           ('2-5', 2, 5), ('5-10', 5, 10), ('>10', 10, 1e9) ]
SET_NAME = { 'val' : 'PYTHIA+HIJING (val)', 'jewel' : 'JEWEL+HIJING' }
CHUNK = 2000
STYLE = {}

plt.rcParams.update({ 'font.size' : 8, 'axes.titlesize' : 8,
                      'axes.labelsize' : 8, 'legend.fontsize' : 6.5,
                      'xtick.labelsize' : 7, 'ytick.labelsize' : 7 })

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Benchmark diagnostics')
    parser.add_argument('--display', default = '',
        help = 'LABEL=SIGNAL_STEM,... for the event displays')
    parser.add_argument('--towers', default = '',
        help = 'LABEL=SIGNAL_STEM[+STEM...],... for the tower scores')
    parser.add_argument('--consistency', default = '',
        help = 'LABEL=RUN__SETTING[+...],... of joint runs')
    parser.add_argument('--figure-labels', default = None,
        help = 'LABEL,... of the background-error figure (default: those'
               " without a '[...]' suffix)")
    parser.add_argument('--sets', default = 'val,jewel')
    parser.add_argument('--n-events', type = int, default = 20000)
    parser.add_argument('--out', default = 'docs/flow/bench')
    return parser.parse_args()

def parse_list(spec):
    out = {}
    for item in filter(None, spec.split(',')):
        (label, stems) = item.split('=', 1)
        out[label] = stems.split('+')
    return out

def images(set_name, stem, n):
    path = os.path.join(fc.out_root(), 'bench', 'images', set_name, f'{stem}.npy')
    return np.load(path, mmap_mode = 'r')[:n]

def chunks(n):
    for start in range(0, n, CHUNK):
        yield slice(start, min(start + CHUNK, n))

def tower_scores(set_name, stem, n):
    m = images(set_name, 'mixture', n)
    s = images(set_name, 'truth', n)
    y = images(set_name, stem, n)
    acc = { 'ae' : 0.0, 'se' : 0.0, 'ae0' : 0.0, 'se0' : 0.0, 'cnt' : 0,
            'ev_sum' : [] }
    bins = { k : [ 0.0, 0.0, 0.0, 0 ] for (k, _, _) in S_BINS }
    for sl in chunks(n):
        ss  = np.asarray(s[sl], np.float64)
        yy  = np.asarray(y[sl], np.float64)
        d   = yy - ss
        d0  = np.clip(yy, 0, None) - ss
        acc['ae']  += np.abs(d).sum()
        acc['se']  += (d**2).sum()
        acc['ae0'] += np.abs(d0).sum()
        acc['se0'] += (d0**2).sum()
        acc['cnt'] += d.size
        acc['ev_sum'].append(d.sum(axis = (1, 2)))
        err = -d                       # (M - S_hat) - (M - S)
        for (k, lo, hi) in S_BINS:
            sel = (ss == 0) if k == 'S = 0' else ((ss > lo) & (ss <= hi))
            e = err[sel]
            bins[k][0] += e.sum()
            bins[k][1] += (e**2).sum()
            bins[k][2] += ss[sel].sum()
            bins[k][3] += e.size
    del m
    ev_sum = np.concatenate(acc['ev_sum'])
    row = { 'mae' : acc['ae'] / acc['cnt'], 'rmse' : np.sqrt(acc['se'] / acc['cnt']),
            'mae_clip0' : acc['ae0'] / acc['cnt'],
            'rmse_clip0' : np.sqrt(acc['se0'] / acc['cnt']),
            'event_bias_gev' : ev_sum.mean(),
            'event_bias_gev_se' : ev_sum.std() / np.sqrt(len(ev_sum)),
            'event_rms_gev' : np.sqrt((ev_sum**2).mean()) }
    for (k, _, _) in S_BINS:
        (se, se2, ssum, cnt) = bins[k]
        row[f'bkgerr_mean[{k}]'] = se / max(cnt, 1)
        row[f'bkgerr_rms[{k}]'] = np.sqrt(se2 / max(cnt, 1))
        row[f'bkgerr_share[{k}]'] = se / ssum if ssum > 0 else np.nan
        row[f'towers[{k}]'] = cnt
    return row

def consistency(set_name, run_setting, n):
    m = images(set_name, 'mixture', n)
    b = images(set_name, f'{run_setting}__bkg', n)
    s = images(set_name, f'{run_setting}__sigdirect', n)
    (tot, ae, se, cnt, ev_sum) = (0.0, 0.0, 0.0, 0, [])
    for sl in chunks(n):
        r = (np.asarray(b[sl], np.float64) + np.asarray(s[sl], np.float64)
             - np.asarray(m[sl], np.float64))
        tot += r.sum()
        ae  += np.abs(r).sum()
        se  += (r**2).sum()
        cnt += r.size
        ev_sum.append(r.sum(axis = (1, 2)))
    ev_sum = np.concatenate(ev_sum)
    return { 'cons_mean' : tot / cnt, 'cons_mae' : ae / cnt,
             'cons_rms' : np.sqrt(se / cnt), 'cons_event_mean_gev' : ev_sum.mean(),
             'cons_event_rms_gev' : np.sqrt((ev_sum**2).mean()) }

def pick_events(set_name, n):
    (embed, signal) = ev.load_pairs(os.environ.get('UVCGAN_S_DATA', 'data'),
                                    20000, 0, truth = set_name)
    truth = ev.Truth(embed[:n], signal[:n], ev.cone_kernel(ev.R_JET), 10.0,
                     torch.device('cpu'))
    e = truth.e_true.numpy()
    ok = np.nonzero(truth.jets.numpy() & (e >= 25) & (e < 35))[0][:2]
    return [ (int(i), int(truth.row[i]), int(truth.col[i]), float(e[i])) for i in ok ]

def plot_displays(events, stems, n, out):
    # pylint: disable=too-many-locals
    cols = [ ('mixture M', 'mixture'), ('truth S', 'truth') ] + list(stems.items())
    (fig, axes) = plt.subplots(2 * len(events), len(cols),
                               figsize = (0.95 * len(cols) + 0.3, 2.3 * len(events)),
                               squeeze = False)
    for (r, (s, i, _row, col, e_true)) in enumerate(events):
        phis  = np.arange(col - 10, col + 10) % 64
        truth = np.asarray(images(s, 'truth', n)[i])[:, phis]
        for (c, (name, stem)) in enumerate(cols):
            img = np.asarray(images(s, stem, n)[i])[:, phis]
            ax  = axes[2 * r][c]
            im1 = ax.imshow(np.log10(np.clip(img, 0, None) + 0.1), vmin = -1,
                            vmax = 1.5, cmap = 'viridis', origin = 'lower',
                            aspect = 'auto')
            if r == 0:
                ax.set_title(name.replace(': ', ':\n'), fontsize = 6.5)
            ax2 = axes[2 * r + 1][c]
            if c >= 2:
                ax2.imshow(img - truth, vmin = -2, vmax = 2, cmap = 'RdBu_r',
                                 origin = 'lower', aspect = 'auto')
                ax2.text(0.5, -0.12, f'sum {np.sum(img - truth):+.1f} GeV',
                         transform = ax2.transAxes, ha = 'center', va = 'top',
                         fontsize = 5.5)
            for a in (ax, ax2):
                a.set_xticks([])
                a.set_yticks([])
            if c == 0:
                ax.set_ylabel(f'{"PYTHIA" if s == "val" else "JEWEL"} #{i}\n'
                              f'cone {e_true:.0f} GeV', fontsize = 6.5)
                fig.colorbar(im1, ax = ax2, fraction = 0.9, aspect = 12,
                             orientation = 'horizontal', label = 'log10(E / GeV + 0.1)')
                ax2.axis('off')
            if c == 1:
                fig.colorbar(ScalarMappable(Normalize(-2, 2), 'RdBu_r'), ax = ax2,
                             fraction = 0.9, aspect = 12, orientation = 'horizontal',
                             label = '$\\hat S - S$, GeV')
                ax2.axis('off')
            if c == 2:
                ax2.set_ylabel('$\\hat S - S$', fontsize = 6.5)
    fig.suptitle('Event displays (raw outputs): 20 towers in $\\phi$ around the jet,'
                 ' full $\\eta$ (24 towers)', fontsize = 7.5)
    fig.tight_layout()
    fig.savefig(out, dpi = 170)
    plt.close(fig)

def style(label):
    st = dict(STYLE.get(label.split(' [')[0], {}))
    if label.endswith('[mid32]'):
        st['ls'] = ':'
    return st

def plot_bkg(df, sets, out, labels = None):
    # pylint: disable=too-many-locals
    order  = [ k for (k, _, _) in S_BINS ]
    # the main readouts; the others are in the tables
    labels = [ l for l in dict.fromkeys(df.label)
               if (l in labels if labels else '[' not in l) ]
    (fig, axes) = plt.subplots(2, len(sets), figsize = (max(3.3 * len(sets), 5.0), 4.2),
                               squeeze = False, sharex = True)
    w = 0.8 / len(labels)
    for (col, s) in enumerate(sets):
        d = df[df.set == s]
        for (row, what) in enumerate([ 'bkgerr_mean', 'bkgerr_rms' ]):
            ax = axes[row][col]
            for (k, label) in enumerate(labels):
                g = d[d.label == label]
                v = np.array([ [ g[f'{what}[{b}]'].iloc[j] for b in order ]
                               for j in range(len(g)) ])
                x = np.arange(len(order)) + (k - len(labels) / 2 + 0.5) * w
                st = style(label)
                ax.bar(x, v.mean(axis = 0), w, color = st.get('color'),
                       hatch = { '--' : '///', ':' : '...' }.get(st.get('ls')),
                       edgecolor = 'white' if st.get('ls') in ('--', ':') else None,
                       label = label)
                if len(v) > 1:
                    ax.errorbar(x, v.mean(axis = 0),
                                yerr = [ v.mean(axis = 0) - v.min(axis = 0),
                                         v.max(axis = 0) - v.mean(axis = 0) ],
                                fmt = 'none', ecolor = 'k', lw = 0.7, capsize = 1.5)
            ax.axhline(0, color = 'k', lw = 0.7)
            ax.grid(alpha = 0.3, axis = 'y')
            if row == 0:
                ax.set_title(SET_NAME[s])
                ax.set_ylabel('mean $\\hat B - B$, GeV')
            else:
                ax.set_ylabel('RMS $\\hat B - B$, GeV')
                ax.set_xticks(np.arange(len(order)))
                ax.set_xticklabels(order)
                ax.set_xlabel('true signal energy S of the tower, GeV')
    axes[0][0].legend(fontsize = 5.5)
    fig.suptitle('Background error by true signal energy of the tower\n'
                 '($\\hat B = M - \\hat S$; seed mean, whiskers: seed range)',
                 fontsize = 7.5)
    fig.tight_layout()
    fig.savefig(out, dpi = 150)
    plt.close(fig)

def summarise(df, keys):
    rows = []
    metrics = [ c for c in df.columns if c not in keys + [ 'seed', 'stem' ] ]
    for (k, g) in df.groupby(keys, sort = False):
        row = dict(zip(keys, k))
        row['n_seeds'] = len(g)
        for m in metrics:
            v = g[m].astype(float)
            row[m] = v.mean()
            row[f'{m}_half_range'] = (v.max() - v.min()) / 2
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    cmdargs = parse_cmdargs()
    os.makedirs(cmdargs.out, exist_ok = True)
    style_path = os.path.join(os.path.dirname(__file__), 'bench_style.json')
    STYLE.update(json.load(open(style_path, encoding = 'utf-8')))
    sets = cmdargs.sets.split(',')

    display = { k : v[0] for (k, v) in parse_list(cmdargs.display).items() }
    if display:
        picked = { s : pick_events(s, cmdargs.n_events) for s in sets }
        print('displayed events (set, index, row, col, cone GeV):', picked)
        for (k, name) in enumerate([ 'bench_displays.png', 'bench_displays_more.png' ]):
            events = [ (s, *picked[s][k]) for s in sets if len(picked[s]) > k ]
            plot_displays(events, display, cmdargs.n_events,
                          os.path.join(cmdargs.out, name))

    towers = parse_list(cmdargs.towers)
    if towers:
        rows = []
        for s in sets:
            for (label, stems) in towers.items():
                for (seed, stem) in enumerate(stems):
                    rows.append({ 'set' : s, 'label' : label, 'seed' : seed,
                                  'stem' : stem,
                                  **tower_scores(s, stem, cmdargs.n_events) })
                print(f'{s}: {label} towers scored', flush = True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(cmdargs.out, 'bench_towers.csv'), index = False)
        summ = summarise(df, [ 'set', 'label' ])
        summ.to_csv(os.path.join(cmdargs.out, 'bench_towers_summary.csv'), index = False)
        plot_bkg(df, sets, os.path.join(cmdargs.out, 'bench_bkgerr.png'),
                 cmdargs.figure_labels.split(',') if cmdargs.figure_labels else None)
        cols = [ 'mae', 'rmse', 'mae_clip0', 'event_bias_gev', 'bkgerr_mean[S = 0]',
                 'bkgerr_share[5-10]', 'bkgerr_share[>10]' ]
        with pd.option_context('display.width', 250, 'display.max_columns', 30):
            print(summ[[ 'set', 'label' ] + cols].round(4).to_string(index = False))

    cons = parse_list(cmdargs.consistency)
    if cons:
        rows = []
        for s in sets:
            for (label, runs) in cons.items():
                for (seed, run) in enumerate(runs):
                    rows.append({ 'set' : s, 'label' : label, 'seed' : seed,
                                  'stem' : run,
                                  **consistency(s, run, cmdargs.n_events) })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(cmdargs.out, 'bench_consistency.csv'), index = False)
        with pd.option_context('display.width', 250):
            print(summarise(df, [ 'set', 'label' ]).round(4).to_string(index = False))

if __name__ == '__main__':
    main()
