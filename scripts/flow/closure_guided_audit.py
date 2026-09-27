#!/usr/bin/env python
"""Audit of the teacher-guided coupling before any training (FLOW_NOTES.md,
"Teacher-guided coupling").

Fixed validation source jets x with hidden truth T(x) are matched against
real unpaired targets y, drawn from the training target pool T(B): other
events, so no true counterpart is present. Batches are the training
matching pool, 256 sources x 256 targets, solved by the training solver
(POT emd, uniform weights, default iterations) and read as the plan's
permutation. The references a source is matched from:

    original   d(x, y)                  the unpaired coupling so far
    guided     d(F_teacher(x), y)       the frozen 2k-pair teacher's endpoint
    oracle     d(T(x), y)               a perfect teacher (hidden truth, audit
                                        only): the best a guided coupling can do
    oracle NN  argmin_y d(T(x), y)      the same without the one-to-one
                                        constraint of the plan: how close the
                                        pool's nearest real jet is to T(x)

d is fm_common.ShapeEnergyCost (lambda 1, its frozen scales). For every
source and matched target, against T(x): normalised-shape EMD, energy
error, and tower residuals by true tower energy. The teacher's own endpoint
F_teacher(x) is scored alongside. Every source is matched against R
independent target batches, which shows how much the selected target
changes with the minibatch. Hidden truth T(x) is used only to score, and
to build the oracle references.

    closure_guided_audit.py [--sources 1024] [--repeats 4]
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import ot as pot

import fm_common as fc
from closure_data import OFFSET
from closure_eval import load_pairs, Jets, TOWER_CLASSES
from closure_teacher import cache_paths

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Guided-coupling audit')
    parser.add_argument('--teacher', default = 'closure_pk1_s0')
    parser.add_argument('--sources', type = int, default = 1024)
    parser.add_argument('--pool', type = int, default = 256)
    parser.add_argument('--repeats', type = int, default = 4)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--out', default = os.path.join(
        fc.out_root(), 'closure_guided_audit'))
    parser.add_argument('--figure', default = 'docs/flow/closure_guided_audit.png')
    return parser.parse_args()

def permutation(cost):
    n = cost.shape[0]
    plan = pot.emd(pot.unif(n), pot.unif(n), cost.double().cpu().numpy())
    return plan.argmax(axis = 1)

def tower_residuals(jets, pred, truth):
    """Mean and rms of pred - truth per class of true tower energy."""
    wp = jets.window(pred)
    wt = jets.window(truth)
    out = {}
    for (label, lo, hi) in TOWER_CLASSES:
        sel = (wt == 0) if label == '=0' else ((wt >= lo) & (wt < hi) & (wt > 0))
        sel = sel & jets.mask
        e = (wp - wt)[sel]
        out[f'tower{label}_mean'] = float(e.mean())
        out[f'tower{label}_rms'] = float(e.pow(2).mean().sqrt())
    return out

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    rng     = np.random.default_rng(cmdargs.seed)
    cost    = fc.ShapeEnergyCost.load(1.0)

    val   = load_pairs('val', cmdargs.sources)
    teach = np.load(cache_paths(cmdargs.teacher)['val'])[:cmdargs.sources]
    tgt_pool = np.load(fc.cache_path('closure_tgt'), mmap_mode = 'r')

    rows    = []
    matched = { k : [] for k in [ 'original', 'guided', 'oracle', 'oracle NN' ] }
    first   = {}
    n_src   = cmdargs.sources // cmdargs.pool * cmdargs.pool

    for rep in range(cmdargs.repeats):
        chosen = { k : np.empty((n_src, 16, 16), np.float32) for k in matched }
        for start in range(0, n_src, cmdargs.pool):
            sl = slice(start, start + cmdargs.pool)
            x  = torch.from_numpy(val['src'][sl]).to(device)
            tx = torch.from_numpy(val['tgt'][sl]).to(device)
            fx = torch.from_numpy(teach[sl]).to(device)
            idx = np.sort(rng.choice(len(tgt_pool), cmdargs.pool, replace = False))
            y  = torch.from_numpy(tgt_pool[idx].astype(np.float32)).to(device)

            c_oracle = cost(tx, y)
            picks = {
                'original'  : permutation(cost(x, y)),
                'guided'    : permutation(cost(fx, y)),
                'oracle'    : permutation(c_oracle),
                'oracle NN' : c_oracle.argmin(dim = 1).cpu().numpy(),
            }
            for (k, j) in picks.items():
                chosen[k][sl] = y[torch.as_tensor(j, device = device)].cpu().numpy()

        for k in matched:
            matched[k].append(chosen[k])
            if rep == 0:
                first[k] = chosen[k]

    pairs = { k : v[:n_src] for (k, v) in val.items() }
    jets  = Jets(pairs['row'], device)
    e_t   = jets.observables(pairs['tgt'])['E']

    def score(name, pred, rep):
        (full, shape) = jets.emds(pred, pairs['tgt'])
        e = jets.observables(pred)['E']
        row = {
            'reference' : name, 'repeat' : rep,
            'shape_emd_mean' : float(np.nanmean(shape)),
            'shape_emd_median' : float(np.nanmedian(shape)),
            'shape_emd_p90' : float(np.nanquantile(shape, 0.9)),
            'emd_gev' : float(full.mean()),
            'E_bias_gev' : float(np.mean(e - e_t)),
            'E_rmse_gev' : float(np.sqrt(np.mean((e - e_t)**2))),
        }
        row.update(tower_residuals(jets, pred, pairs['tgt']))
        return row

    rows.append(score('teacher F(x)', teach[:n_src], 0))
    rows.append(score('identity x', pairs['src'], 0))
    for k in matched:
        for (rep, pred) in enumerate(matched[k]):
            rows.append(score(f'matched, {k}', pred, rep))

    # how much the selected target changes between target batches
    spread = []
    for k in matched:
        d = []
        for (a, b) in [ (0, 1), (1, 2), (2, 3) ][:cmdargs.repeats - 1]:
            (_, s) = jets.emds(matched[k][a], matched[k][b])
            d.append(np.nanmean(s))
        spread.append({ 'reference' : f'matched, {k}',
                        'shape_emd_between_repeats' : float(np.mean(d)) })

    df = pd.DataFrame(rows)
    df.to_csv(f'{cmdargs.out}.csv', index = False)
    agg = df.groupby('reference', sort = False).mean(numeric_only = True) \
        .drop(columns = 'repeat')
    agg = agg.join(pd.DataFrame(spread).set_index('reference'))
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(agg[[ 'shape_emd_mean', 'shape_emd_median', 'shape_emd_p90',
                    'emd_gev', 'E_bias_gev', 'E_rmse_gev',
                    'shape_emd_between_repeats' ]].round(4).to_string())
        print()
        print(agg[[ c for c in agg.columns if c.startswith('tower') ]]
              .round(3).to_string())
    print(f'wrote {cmdargs.out}.csv')

    plot(pairs, teach[:n_src], first, jets, cmdargs.figure)

def plot(pairs, teach, first, jets, out):
    e = pairs['src'][:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9].sum((1, 2))
    pick = [ int(np.argmin(np.abs(e - np.quantile(e, q))))
             for q in (0.25, 0.5, 0.75) ]
    cols = [ ('x', pairs['src']), ('T(x), hidden', pairs['tgt']),
             ('teacher F(x)', teach) ] + [
             (f'matched: {k}', v) for (k, v) in first.items() ]
    (fig, axes) = plt.subplots(len(pick), len(cols),
                               figsize = (2.0 * len(cols), 2.3 * len(pick)))
    for (c, (name, img)) in enumerate(cols):
        sel = img[pick]
        w = sel[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * jets.mask.cpu().numpy()
        for (r, i) in enumerate(pick):
            ax = axes[r][c]
            ax.imshow(np.log10(np.clip(w[r], 0, None) + 0.1), cmap = 'viridis',
                      vmin = -1, vmax = 1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{w[r].sum():.1f} GeV', fontsize = 6)
            if r == 0:
                ax.set_xlabel(name, fontsize = 7)
                ax.xaxis.set_label_position('top')
    fig.suptitle('Guided-coupling audit: what each reference selects from 256'
                 ' real targets (true counterpart absent)', fontsize = 8)
    fig.tight_layout()
    fig.savefig(out, dpi = 110)
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
