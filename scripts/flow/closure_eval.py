#!/usr/bin/env python
"""Event-level fidelity of the jet -> jet flow on the closure test
(FLOW_NOTES.md, "Closure test").

The flow F was trained unpaired on source jets A and modified jets T(B)
of different parent events (closure_data.py). Here it is applied to held-out
source jets J and compared, event by event, with their known modified
versions T(J):

    energy      the raw response E(F(J)) / E(J), whose target is the
                modification's 0.8, not 1 (no calibration); bias and RMSE of
                E(F(J)) - E(T(J)) in GeV and of log(E(F(J)) / E(T(J)))
    shape       normalised-shape EMD between F(J) and T(J), both scaled to
                unit energy (dR / 0.4 per unit energy, as jet_fidelity.EMD)
    observables mass, girth, p_T^D, core, z_lead, n1, z_g, R_g of the cone
                (substructure.observables): bias and RMSE of O(F(J)) - O(T(J))
                in units of sigma(O(T(J))); the modification recovered,
                Delta_pred = O(F(J)) - O(J) against Delta_true = O(T(J)) -
                O(J): mean Delta_pred / mean Delta_true and their per-jet
                correlation
    marginals   W1 / sigma between the distributions of O(F(J)) and O(T(J))
                (energy included); joint: the largest difference between the
                two correlation matrices of (log E, observables)

Reference outputs: `identity` (F(J) = J) and `random target` (F(J) = T(J')
for an unrelated J' of the same pool: the right marginals, no event
correspondence).

Modes:
    --select   score every checkpoint of each run on the validation pairs
               (EMA network, the selection setting of 4 Euler steps) and
               mark the one of lowest mean full EMD(F(J), T(J)) (GeV), fixed
               as the selection metric before scoring
    default    score each run's selected checkpoint on the test pairs, the
               references alongside, and on a fixed subset (--resolved-n)
               also with a better-resolved solve (--resolved-nfe, midpoint)

    --figure   test jets at four energy quantiles: J, T(J) and F(J) of each
               run at both solves, with each jet's normalised-shape EMD to T(J)

    --setting NFE:SOLVER   the inference setting of --select, --control and
               --figure (default: the selection setting, 4 Euler steps); its
               selections go to evals/closure_val_<NFE><SOLVER>.csv, with the
               same scores on training pairs (--train-n) in
               evals/closure_train_<NFE><SOLVER>.csv
    --control  the positive-control report (FLOW_NOTES.md, "Paired positive
               control"): identity and each run at --setting on the test
               pairs, raw outputs clipped at 0 as the mapper does; energy
               response against both denominators (E_out / E_in, whose truth is
               0.8, and E_out / E(T(J)), whose truth is 1), bias and RMSE of the
               mass and girth modification (O(F(J)) - O(J) against O(T(J)) -
               O(J), physical units), the energy of the cone towers empty in
               T(J) before and after the clip, and the per-tower error by true
               tower energy
    --truth identity       the target of every jet is the jet itself (the
               null test)

    closure_eval.py RUN [RUN ...] --select [--setting 32:midpoint --train-n 5000]
    closure_eval.py RUN [RUN ...] [--out PREFIX]
    closure_eval.py RUN [RUN ...] --control --setting 32:midpoint [--truth identity]
    closure_eval.py RUN [RUN ...] --figure PNG [--setting 32:midpoint]
"""

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

import fm_common as fc
from closure_data import OFFSET, cone_mask
from jet_fidelity import EMD
from substructure import Geometry, observables, OBSERVABLES

ev = fc.ev

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Closure-test fidelity')
    parser.add_argument('runs', nargs = '+')
    parser.add_argument('--select', action = 'store_true')
    parser.add_argument('--n-val', type = int, default = 5000)
    parser.add_argument('--resolved-n', type = int, default = 2000)
    parser.add_argument('--resolved-nfe', type = int, default = 32)
    parser.add_argument('--batch', type = int, default = 2000)
    parser.add_argument('--out', default = os.path.join(
        fc.out_root(), 'closure_eval'))
    parser.add_argument('--figure', default = None)
    parser.add_argument('--setting', default = None,
        help = 'NFE:SOLVER, e.g. 32:midpoint')
    parser.add_argument('--train-n', type = int, default = 0,
        help = 'with --select: also score this many training pairs')
    parser.add_argument('--control', action = 'store_true')
    parser.add_argument('--truth', default = 'modified',
        choices = [ 'modified', 'identity' ])
    return parser.parse_args()

def setting_of(cmdargs):
    if cmdargs.setting:
        (nfe, solver) = cmdargs.setting.split(':')
        return (int(nfe), solver)
    (nfe, solver, _, _) = fc.SELECTION['jetflow']
    return (nfe, solver)

def tag_of(cmdargs):
    return '' if not cmdargs.setting else '_' + cmdargs.setting.replace(':', '')

def load_pairs(name, n = None, truth = 'modified'):
    """Pairs (src = J, tgt = T(J), row, ...) of the validation, test or
    training pool (training: the first n source jets A of the cache, T
    applied here); `truth = 'identity'`: tgt = J."""
    cache = os.path.dirname(fc.cache_path('signal'))
    if name == 'train':
        # pylint: disable=import-outside-toplevel
        from closure_data import Modification
        src  = np.load(fc.cache_path('closure_src'), mmap_mode = 'r')[:n] \
            .astype(np.float32)
        meta = np.load(os.path.join(cache, 'closure_meta.npz'))
        d = { 'src' : src, 'row' : meta['src_row'][:n],
              'col' : meta['src_col'][:n], 'parent' : meta['src_parent'][:n],
              'tgt' : Modification(torch.device('cpu'))(
                  torch.from_numpy(src)).numpy() }
    else:
        f = np.load(os.path.join(cache, f'closure_{name}.npz'))
        d = { k : f[k][slice(0, n)] for k in f.files }
    if truth == 'identity':
        d['tgt'] = d['src'].copy()
    return d

class Jets:
    """Observables and EMD of canvases in the jet frame of `rows`."""

    def __init__(self, rows, device):
        self.geo  = Geometry(device)
        self.emd  = EMD(self.geo)
        self.mask = torch.from_numpy(cone_mask()).to(device)
        self.row  = torch.as_tensor(rows, device = device)
        self.device = device

    def window(self, canvas):
        c = torch.as_tensor(canvas, device = self.device).float()
        return c[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9].clamp(min = 0) \
            * self.mask

    def observables(self, canvas):
        w   = self.window(canvas)
        idx = torch.arange(len(w), device = self.device)
        o   = observables(w, self, idx, self.geo)
        o['E'] = w.sum((1, 2)).cpu().numpy()
        return o

    def emds(self, a, b):
        """(full EMD in GeV, normalised-shape EMD) between canvases."""
        (wa, wb) = (self.window(a), self.window(b))
        (full, _) = self.emd(wa, wb)
        na = wa / wa.sum((1, 2), keepdim = True).clamp(min = 1e-12)
        nb = wb / wb.sum((1, 2), keepdim = True).clamp(min = 1e-12)
        (shape, _) = self.emd(na, nb)
        return (full, shape)

@torch.no_grad()
def apply(mapper, src, batch, device):
    out = []
    for start in range(0, len(src), batch):
        j = torch.as_tensor(src[start:start + batch], device = device).float()
        out.append(mapper(j).cpu())
    return torch.cat(out).numpy()

def scores(pred, pairs, jets, name):
    # pylint: disable=too-many-locals
    o_f = jets.observables(pred)
    o_t = jets.observables(pairs['tgt'])
    o_j = jets.observables(pairs['src'])
    (full, shape) = jets.emds(pred, pairs['tgt'])

    (e_f, e_t, e_j) = (o_f['E'], o_t['E'], o_j['E'])
    row = {
        'model' : name, 'n' : len(e_f),
        'response_mean' : float(np.mean(e_f / e_j)),
        'response_sd' : float(np.std(e_f / e_j)),
        'E_bias_gev' : float(np.mean(e_f - e_t)),
        'E_rmse_gev' : float(np.sqrt(np.mean((e_f - e_t)**2))),
        'logE_bias' : float(np.mean(np.log(e_f / e_t))),
        'logE_rmse' : float(np.sqrt(np.mean(np.log(e_f / e_t)**2))),
        'emd_gev' : float(full.mean()),
        'shape_emd' : float(np.nanmean(shape)),
        'shape_emd_median' : float(np.nanmedian(shape)),
    }
    rows = []
    for q in [ 'E' ] + OBSERVABLES:
        (f, t, j) = (o_f[q], o_t[q], o_j[q])
        ok = np.isfinite(f) & np.isfinite(t) & np.isfinite(j)
        (f, t, j) = (f[ok], t[ok], j[ok])
        sd = t.std()
        (d_pred, d_true) = (f - j, t - j)
        rows.append({
            'model' : name, 'observable' : q,
            'rel_bias' : float((f - t).mean() / sd),
            'rel_rmse' : float(np.sqrt(np.mean((f - t)**2)) / sd),
            'delta_true' : float(d_true.mean()),
            'delta_pred' : float(d_pred.mean()),
            'delta_frac' : float(d_pred.mean() / d_true.mean())
                if abs(d_true.mean()) > 1e-9 else np.nan,
            'delta_corr' : float(np.corrcoef(d_pred, d_true)[0, 1])
                if d_pred.std() > 0 and d_true.std() > 0 else np.nan,
            'w1_sigma' : float(wasserstein_distance(f, t) / sd),
        })

    keys = [ 'E' ] + [ q for q in OBSERVABLES if q not in ('zg', 'rg') ]
    def corr(o):
        m = np.stack([ np.log(o['E']) if q == 'E' else o[q] for q in keys ])
        return np.corrcoef(m)
    row['joint_corr_maxdiff'] = float(np.abs(corr(o_f) - corr(o_t)).max())
    return (row, rows)

def checkpoints(run):
    return fc.list_checkpoints(run)

def ckpt_step(path):
    return int(re.search(r'step_(\d+)\.pt$', path).group(1))

def select(cmdargs, device):
    # pylint: disable=too-many-locals
    (nfe, solver) = setting_of(cmdargs)
    sets = { 'val' : load_pairs('val', cmdargs.n_val, cmdargs.truth) }
    if cmdargs.train_n > 0:
        sets['train'] = load_pairs('train', cmdargs.train_n, cmdargs.truth)
    jets = { k : Jets(v['row'], device) for (k, v) in sets.items() }

    for run in cmdargs.runs:
        rows = { k : [] for k in sets }
        for ckpt in checkpoints(run):
            (method, net, state, config) = fc.load_run(run, ckpt, device, 'ema')
            mapper = fc.JetMapper(method, net, nfe, solver)
            for (k, pairs) in sets.items():
                pred = apply(mapper, pairs['src'], cmdargs.batch, device)
                (row, _) = scores(pred, pairs, jets[k], config['label'])
                row.update({ 'step' : ckpt_step(ckpt),
                             'train_time' : state['stats']['train_time'] })
                rows[k].append(row)
                print(f"{config['label']} {k:5s} step {row['step']:7d}"
                      f" {row['train_time'] / 60:6.1f} min  emd"
                      f" {row['emd_gev']:.3f}  shape {row['shape_emd']:.4f}"
                      f"  response {row['response_mean']:.3f}", flush = True)
        df = pd.DataFrame(rows['val'])
        best = int(df.sort_values('emd_gev').step.iloc[0])
        df['selected'] = df.step == best
        os.makedirs(os.path.join(run, 'evals'), exist_ok = True)
        df.to_csv(os.path.join(run, 'evals', f'closure_val{tag_of(cmdargs)}.csv'),
                  index = False)
        if 'train' in rows:
            pd.DataFrame(rows['train']).to_csv(os.path.join(
                run, 'evals', f'closure_train{tag_of(cmdargs)}.csv'),
                index = False)
        print(f'{run}: selected step {best}', flush = True)

def selected_step(run, tag = ''):
    df = pd.read_csv(os.path.join(run, 'evals', f'closure_val{tag}.csv'))
    return int(df[df.selected].step.iloc[0])

TOWER_CLASSES = [ ('=0', 0, 0), ('<0.2', 1e-9, 0.2), ('0.2-1', 0.2, 1),
                  ('1-5', 1, 5), ('>5', 5, 1e9) ]

def control_scores(raw, pairs, jets, name):
    """The positive-control report of one output (raw: before the clip)."""
    # pylint: disable=too-many-locals
    (row, _) = scores(np.clip(raw, 0, None), pairs, jets, name)
    o_f = jets.observables(np.clip(raw, 0, None))
    o_t = jets.observables(pairs['tgt'])
    o_j = jets.observables(pairs['src'])
    (e_f, e_t, e_j) = (o_f['E'], o_t['E'], o_j['E'])
    row.update({
        'E_out_over_E_in' : float(np.mean(e_f / e_j)),
        'E_out_over_E_in_sd' : float(np.std(e_f / e_j)),
        'E_out_over_E_true' : float(np.mean(e_f / e_t)),
        'E_out_over_E_true_rmse' : float(np.sqrt(np.mean((e_f / e_t - 1)**2))),
    })
    for q in [ 'mass', 'girth' ]:
        d_true = o_t[q] - o_j[q]
        err    = (o_f[q] - o_j[q]) - d_true
        ok     = np.isfinite(err)
        row.update({
            f'd{q}_true_mean' : float(np.mean(d_true[ok])),
            f'd{q}_true_sd' : float(np.std(d_true[ok])),
            f'd{q}_bias' : float(np.mean(err[ok])),
            f'd{q}_rmse' : float(np.sqrt(np.mean(err[ok]**2))),
        })

    mask = jets.mask
    w_raw = torch.as_tensor(raw, device = jets.device).float()[
        :, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * mask
    w_t   = jets.window(pairs['tgt'])
    empty = (w_t == 0) & mask
    row.update({
        'empty_towers' : float(empty.sum((1, 2)).float().mean()),
        'halo_signed_gev' : float((w_raw * empty).sum((1, 2)).mean()),
        'halo_clipped_gev' : float((w_raw.clamp(min = 0) * empty)
                                   .sum((1, 2)).mean()),
    })
    err = w_raw.clamp(min = 0) - w_t
    for (label, lo, hi) in TOWER_CLASSES:
        sel = empty if label == '=0' else ((w_t >= lo) & (w_t < hi) & mask
                                           & (w_t > 0))
        e = err[sel]
        row[f'tower{label}_mean'] = float(e.mean())
        row[f'tower{label}_rms'] = float(e.pow(2).mean().sqrt())
    return row

def control(cmdargs, device):
    # pylint: disable=too-many-locals
    (nfe, solver) = setting_of(cmdargs)
    pairs = load_pairs('test', None, cmdargs.truth)
    jets  = Jets(pairs['row'], device)
    rows  = [ control_scores(pairs['src'], pairs, jets, 'identity F(J) = J') ]
    for run in cmdargs.runs:
        step = selected_step(run, tag_of(cmdargs))
        ckpt = [ c for c in checkpoints(run) if ckpt_step(c) == step ][0]
        (method, net, state, config) = fc.load_run(run, ckpt, device, 'ema')
        mapper = fc.JetMapper(method, net, nfe, solver, clip = False)
        torch.cuda.synchronize()
        t0  = time.perf_counter()
        raw = apply(mapper, pairs['src'], cmdargs.batch, device)
        torch.cuda.synchronize()
        dt  = time.perf_counter() - t0
        row = control_scores(raw, pairs, jets, config['label'])
        row.update({
            'setting' : f'{solver}{nfe}', 'step' : step,
            'train_time_min' : state['stats']['train_time'] / 60,
            'updates' : state['stats']['step'],
            'pairing' : config.get('pairing', 'unpaired'),
            'cost' : config.get('cost', 'l2'),
            'ms_per_jet' : 1000 * dt / len(pairs['src']),
        })
        rows.append(row)
        print(f"{config['label']}: done", flush = True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{cmdargs.out}.csv', index = False)
    show = [ [ 'model', 'n', 'shape_emd', 'shape_emd_median', 'emd_gev',
               'E_out_over_E_in', 'E_out_over_E_in_sd', 'E_out_over_E_true',
               'E_bias_gev', 'E_rmse_gev' ],
             [ 'model', 'dmass_true_mean', 'dmass_true_sd', 'dmass_bias',
               'dmass_rmse', 'dgirth_true_mean', 'dgirth_true_sd',
               'dgirth_bias', 'dgirth_rmse' ],
             [ 'model', 'empty_towers', 'halo_signed_gev', 'halo_clipped_gev' ]
             + [ f'tower{c[0]}_{k}' for c in TOWER_CLASSES
                 for k in ('mean', 'rms') ] ]
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        for cols in show:
            print(df[cols].round(4).to_string(index = False))
            print()
    print(f'wrote {cmdargs.out}.csv')

def evaluate(cmdargs, device):
    # pylint: disable=too-many-locals
    pairs = load_pairs('test')
    jets  = Jets(pairs['row'], device)
    rng   = np.random.default_rng(0)
    table = []
    obs   = []

    for (name, pred) in [
        ('identity', pairs['src']),
        ('random target', pairs['tgt'][rng.permutation(len(pairs['tgt']))]),
    ]:
        (row, rows) = scores(pred, pairs, jets, name)
        table.append(row)
        obs += rows

    sub = { k : v[:cmdargs.resolved_n] for (k, v) in pairs.items() }
    jets_sub = Jets(sub['row'], device)

    for run in cmdargs.runs:
        step = selected_step(run)
        ckpt = [ c for c in checkpoints(run) if ckpt_step(c) == step ][0]
        (method, net, state, config) = fc.load_run(run, ckpt, device, 'ema')
        (nfe, solver, _, _) = fc.SELECTION['jetflow']

        for (setting, n_fe, solv, data, j) in [
            (f'euler{nfe}', nfe, solver, pairs, jets),
            (f'euler{nfe} (subset)', nfe, solver, sub, jets_sub),
            (f'midpoint{cmdargs.resolved_nfe} (subset)', cmdargs.resolved_nfe,
             'midpoint', sub, jets_sub),
        ]:
            mapper = fc.JetMapper(method, net, n_fe, solv)
            torch.cuda.synchronize()
            t0   = time.perf_counter()
            pred = apply(mapper, data['src'], cmdargs.batch, device)
            torch.cuda.synchronize()
            dt   = time.perf_counter() - t0
            label = f"{config['label']} {setting}"
            (row, rows) = scores(pred, data, j, label)
            row.update({
                'run' : config['label'], 'setting' : setting, 'step' : step,
                'train_time_min' : state['stats']['train_time'] / 60,
                'updates' : state['stats']['step'],
                'cost' : config.get('cost', 'l2'),
                'cost_lambda' : config.get('cost_lambda'),
                'backbone' : config.get('backbone', 'unet'),
                'ms_per_jet' : 1000 * dt / len(data['src']),
            })
            table.append(row)
            obs += rows
            print(f'{label}: done', flush = True)

    df  = pd.DataFrame(table)
    odf = pd.DataFrame(obs)
    df.to_csv(f'{cmdargs.out}.csv', index = False)
    odf.to_csv(f'{cmdargs.out}_observables.csv', index = False)

    with pd.option_context('display.width', 250, 'display.max_rows', 300,
                           'display.max_columns', 30):
        print(df[[ 'model', 'n', 'response_mean', 'response_sd', 'E_bias_gev',
                   'E_rmse_gev', 'logE_rmse', 'emd_gev', 'shape_emd',
                   'joint_corr_maxdiff' ]].round(4).to_string(index = False))
        for (value, title) in [
            ('rel_rmse', 'RMSE of O(F(J)) - O(T(J)) / sigma(O(T(J)))'),
            ('rel_bias', 'bias / sigma'),
            ('delta_frac', 'mean modification recovered, mean Delta_pred /'
                           ' mean Delta_true'),
            ('delta_corr', 'per-jet correlation of Delta_pred and Delta_true'),
            ('w1_sigma', 'W1 / sigma of the marginals, F(J) against T(J)'),
        ]:
            print(f'\n{title}:')
            print(odf.pivot_table(index = 'model', columns = 'observable',
                                  values = value, sort = False)
                  [[ 'E' ] + OBSERVABLES].round(3).to_string())
    print(f'wrote {cmdargs.out}.csv, {cmdargs.out}_observables.csv')

def figure(cmdargs, device):
    """Representative test jets, J -> T(J) and each run's F(J)."""
    # pylint: disable=import-outside-toplevel,too-many-locals
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pairs = load_pairs('test', cmdargs.resolved_n, cmdargs.truth)
    jets  = Jets(pairs['row'], device)
    e     = pairs['src'][:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9].sum((1, 2))
    pick  = [ int(np.argmin(np.abs(e - np.quantile(e, q))))
              for q in (0.25, 0.5, 0.75, 0.95) ]
    sel   = { k : v[pick] for (k, v) in pairs.items() }
    jsel  = Jets(sel['row'], device)

    target  = 'J' if cmdargs.truth == 'identity' else 'T(J)'
    columns = [ ('J', sel['src']) ] + (
        [] if cmdargs.truth == 'identity' else [ ('T(J)', sel['tgt']) ])
    settings = [ setting_of(cmdargs) ] if cmdargs.setting else \
        [ (4, 'euler'), (cmdargs.resolved_nfe, 'midpoint') ]
    for run in cmdargs.runs:
        step = selected_step(run, tag_of(cmdargs))
        ckpt = [ c for c in checkpoints(run) if ckpt_step(c) == step ][0]
        (method, net, _, config) = fc.load_run(run, ckpt, device, 'ema')
        for (nfe, solver) in settings:
            mapper = fc.JetMapper(method, net, nfe, solver)
            columns.append((f"{config['label']}\n{solver} {nfe}",
                            apply(mapper, sel['src'], 16, device)))

    (fig, axes) = plt.subplots(len(pick), len(columns),
                               figsize = (2.1 * len(columns), 2.3 * len(pick)))
    for (c, (name, img)) in enumerate(columns):
        w = jsel.window(img).cpu().numpy()
        (_, shape) = jsel.emds(img, sel['tgt'])
        for r in range(len(pick)):
            ax = axes[r][c]
            ax.imshow(np.log10(w[r] + 0.1), cmap = 'viridis', vmin = -1,
                      vmax = 1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            title = f'{w[r].sum():.1f} GeV'
            if c > 0:
                title += f', shape EMD {shape[r]:.3f}'
            ax.set_title(title, fontsize = 6)
            if r == 0:
                ax.set_xlabel(name, fontsize = 7)
                ax.xaxis.set_label_position('top')
    fig.suptitle('Closure test jets (cone towers, log10(E + 0.1)); shape EMD'
                 f' to {target}, unit-energy jets', fontsize = 8)
    fig.tight_layout()
    fig.savefig(cmdargs.figure, dpi = 110)
    print(f'wrote {cmdargs.figure}')

def main():
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    cmdargs.runs = [ r if os.path.isabs(r) or os.path.exists(r)
                     else os.path.join(fc.out_root(), r) for r in cmdargs.runs ]
    if cmdargs.select:
        select(cmdargs, device)
    elif cmdargs.control:
        control(cmdargs, device)
    elif cmdargs.figure:
        figure(cmdargs, device)
    else:
        evaluate(cmdargs, device)

if __name__ == '__main__':
    main()
