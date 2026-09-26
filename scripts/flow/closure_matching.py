#!/usr/bin/env python
"""What drives the minibatch matching of the closure test, before training
(FLOW_NOTES.md, "Closure test").

Fixed unpaired minibatches as in training: n source jets A and n target
jets T(B) of the closure pools (closure_data.py), different parent events.
Costs, each solved by the same exact OT (POT emd, uniform weights) and read
as the plan's permutation:

    full          squared L2 of log(E + 0.1) of the full 24 x 64 events
                  (vac_med_coupling.py); a target's full event is its parent
                  event with the cone replaced by T(B)
    centred       squared L2 of log(E + 0.1) of the jet-centred canvases,
                  i.e. of the cone towers (vac_med_coupling.py; the training
                  baseline, TorchCFM's cost on the standardised states)
    shape+E l=0   fm_common.ShapeEnergyCost, lambda = 0 (shape only)
    shape+E l=1   the same, lambda = 1 (the candidate)
    random        a random permutation (reference)

Reported per cost: rank correlation of source and matched target energies
(the known modification scales every energy by 0.8, a monotone map), the
median target / source energy ratio, the mean normalised-shape EMD between
matched jets (a distance none of the costs optimises), the mean distance
between the matched jets' axes in the full events, and figures of matched
energies and of representative pairs. Checks: the calibration of the
candidate's scales (a_s, a_E, from the first 2048 jets of each training
pool, frozen to OUTDIR/sphenix/flow/closure_cost_calib.json), and that
independent periodic phi shifts of full events, followed by re-finding the
axis and re-centring, leave the centred costs unchanged while the full-image
cost changes. A permutation keeps the target marginal whatever the cost:
none of this validates a correspondence.

    closure_matching.py [--batches 256,1024] [--repeats 4]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import ot as pot

import fm_common as fc
from closure_data import OFFSET, cone_mask
from jet_fidelity import EMD
from substructure import Geometry

ev = fc.ev

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Closure matching audit')
    parser.add_argument('--batches', default = '256,1024')
    parser.add_argument('--repeats', type = int, default = 4)
    parser.add_argument('--calib', type = int, default = 2048)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--figure', default = 'docs/flow/closure_matching.png')
    return parser.parse_args()

def cache(name):
    return os.path.join(os.path.dirname(fc.cache_path('signal')), name)

def permutation(cost):
    n = cost.shape[0]
    plan = pot.emd(pot.unif(n), pot.unif(n), cost.double().cpu().numpy(),
                   numItermax = 10_000_000)
    return plan.argmax(axis = 1)

def l2_log(a, b):
    return torch.cdist(torch.log(a.flatten(1) + fc.BIAS),
                       torch.log(b.flatten(1) + fc.BIAS))**2

class ShapeEMD:
    """Mean EMD between unit-energy jets (dimensionless, dR / 0.4)."""

    def __init__(self, device):
        self.geo  = Geometry(device)
        self.emd  = EMD(self.geo)
        self.mask = torch.from_numpy(cone_mask()).to(device)

    def window(self, canvas):
        return canvas[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * self.mask

    def __call__(self, a, b):
        wa = self.window(a)
        wb = self.window(b)
        wa = wa / wa.sum((1, 2), keepdim = True)
        wb = wb / wb.sum((1, 2), keepdim = True)
        (full, _) = self.emd(wa, wb)
        return float(full.mean())

def full_events(parents, rows, cols, canvases, data, device):
    """Parent events with their cone replaced by the given canvases."""
    img = torch.from_numpy(data[np.sort(parents)].astype(np.float32))
    img = img[np.argsort(np.argsort(parents))].to(device)
    off = torch.arange(-4, 5, device = device)
    r = torch.as_tensor(rows, device = device)[:, None, None] + off[None, :, None]
    c = (torch.as_tensor(cols, device = device)[:, None, None]
         + off[None, None, :]) % fc.SHAPE[1]
    n = torch.arange(len(parents), device = device)[:, None, None]
    mask = torch.from_numpy(cone_mask()).to(device)
    win  = canvases[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9].to(device)
    img[n, r, c] = torch.where(mask, win, img[n, r, c])
    return img

def recentre(full, kernel, mask):
    """Axis (the leading R = 0.4 cone) and centred canvas of full events."""
    sums = ev.cone_sums(full, kernel)
    flat = sums.flatten(1).argmax(dim = 1)
    row  = flat // sums.shape[2]
    col  = flat %  sums.shape[2]
    safe = row.clamp(4, fc.SHAPE[0] - 5)
    off  = torch.arange(-4, 5, device = full.device)
    r = safe[:, None, None] + off[None, :, None]
    c = (col[:, None, None] + off[None, None, :]) % fc.SHAPE[1]
    n = torch.arange(len(full), device = full.device)[:, None, None]
    canvas = torch.zeros((len(full), 16, 16), device = full.device)
    canvas[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] = \
        full[n, r, c].clamp(min = 0) * mask
    return (row.cpu().numpy(), col.cpu().numpy(), canvas)

def phi_check(fa, fb, a, b, meta, ia, ib, costs, rng, device):
    """Independent periodic phi shifts of the full events, the axis found
    again and the jets re-centred: relative change of each cost matrix on
    the events whose axis is found at the shifted original tower (and the
    share of such events; another cone can lead once a target jet has lost
    energy)."""
    # pylint: disable=too-many-arguments,too-many-locals
    kernel = ev.cone_kernel(ev.R_JET).to(device)
    mask   = torch.from_numpy(cone_mask()).to(device)
    out    = {}
    same   = []
    cans   = []
    for (full, row, col) in [
        (fa, meta['src_row'][ia], meta['src_col'][ia]),
        (fb, meta['tgt_row'][ib], meta['tgt_col'][ib]),
    ]:
        shift = rng.integers(0, fc.SHAPE[1], size = len(full))
        moved = torch.stack([ torch.roll(x, int(k), dims = 1)
                              for (x, k) in zip(full, shift) ])
        (r2, c2, canvas) = recentre(moved, kernel, mask)
        same.append((r2 == row) & (c2 == (col + shift) % fc.SHAPE[1]))
        cans.append(canvas)
    (sa, sb) = (torch.from_numpy(same[0]), torch.from_numpy(same[1]))
    out['same_axis_src'] = float(sa.float().mean())
    out['same_axis_tgt'] = float(sb.float().mean())

    def change(c0, c1):
        c0 = c0[sa][:, sb]
        c1 = c1[sa][:, sb]
        return float((c1 - c0).abs().max() / c0.abs().max())

    out['centred'] = change(l2_log(a, b), l2_log(cans[0], cans[1]))
    for (name, c) in costs.items():
        out[name] = change(c(a, b), c(cans[0], cans[1]))

    # the full-image cost under the same kind of shifts, on all events
    ga = torch.stack([ torch.roll(x, int(k), dims = 1) for (x, k) in
                       zip(fa, rng.integers(0, fc.SHAPE[1], size = len(fa))) ])
    gb = torch.stack([ torch.roll(x, int(k), dims = 1) for (x, k) in
                       zip(fb, rng.integers(0, fc.SHAPE[1], size = len(fb))) ])
    c0 = l2_log(fa, fb)
    out['full'] = float((l2_log(ga, gb) - c0).abs().max() / c0.abs().max())
    return out

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    rng     = np.random.default_rng(cmdargs.seed)

    src  = np.load(cache('train_closure_src.npy'), mmap_mode = 'r')
    tgt  = np.load(cache('train_closure_tgt.npy'), mmap_mode = 'r')
    meta = np.load(cache('closure_meta.npz'))
    data = np.load(fc.cache_path('signal'), mmap_mode = 'r')

    # the candidate's scales, from a fixed training-only subset, frozen
    cost = fc.ShapeEnergyCost()
    calib = cost.calibrate(
        torch.from_numpy(src[:cmdargs.calib].astype(np.float32)).to(device),
        torch.from_numpy(tgt[:cmdargs.calib].astype(np.float32)).to(device),
    )
    calib['subset'] = (f'first {cmdargs.calib} jets of train_closure_src and'
                       ' of train_closure_tgt, all source-target pairs')
    with open(fc.ShapeEnergyCost.calib_path(), 'w', encoding = 'utf-8') as f:
        json.dump(calib, f, indent = 4)
    print('calibration:', json.dumps(calib), flush = True)

    costs = { 'shape+E l=0' : fc.ShapeEnergyCost.load(0.0),
              'shape+E l=1' : fc.ShapeEnergyCost.load(1.0) }
    semd  = ShapeEMD(device)
    rows  = []
    shows = {}

    for n in [ int(x) for x in cmdargs.batches.split(',') ]:
        for rep in range(cmdargs.repeats):
            ia = np.sort(rng.choice(len(src) - cmdargs.calib, n,
                                    replace = False) + cmdargs.calib)
            ib = np.sort(rng.choice(len(tgt) - cmdargs.calib, n,
                                    replace = False) + cmdargs.calib)
            a = torch.from_numpy(src[ia].astype(np.float32)).to(device)
            b = torch.from_numpy(tgt[ib].astype(np.float32)).to(device)
            ea = a.sum((1, 2)).cpu().numpy()
            eb = b.sum((1, 2)).cpu().numpy()

            fa = full_events(meta['src_parent'][ia], meta['src_row'][ia],
                             meta['src_col'][ia], a, data, device)
            fb = full_events(meta['tgt_parent'][ib], meta['tgt_row'][ib],
                             meta['tgt_col'][ib], b, data, device)

            perms = {
                'random'   : rng.permutation(n),
                'full'     : permutation(l2_log(fa, fb)),
                'centred'  : permutation(l2_log(a, b)),
            }
            for (name, c) in costs.items():
                perms[name] = permutation(c(a, b))

            for (name, j) in perms.items():
                deta = (meta['src_row'][ia] - meta['tgt_row'][ib][j]) * ev.DETA
                dcol = np.abs(meta['src_col'][ia] - meta['tgt_col'][ib][j])
                dphi = np.minimum(dcol, fc.SHAPE[1] - dcol) * ev.DPHI
                rows.append({
                    'batch' : n, 'repeat' : rep, 'cost' : name,
                    'energy_rank_corr' : float(spearmanr(ea, eb[j])[0]),
                    'median_ratio' : float(np.median(eb[j] / ea)),
                    'ratio_iqr' : float(np.subtract(*np.quantile(
                        eb[j] / ea, [ 0.75, 0.25 ]))),
                    'shape_emd' : semd(a, b[j]),
                    'axis_dr' : float(np.sqrt(deta**2 + dphi**2).mean()),
                })
                if (n == 256) and (rep == 0):
                    shows[name] = (ea, eb[j], a.cpu(), b[j].cpu())

            if (n == 256) and (rep == 0):
                print('phi shifts + re-centring:', json.dumps(phi_check(
                    fa, fb, a, b, meta, ia, ib, costs, rng, device)),
                    flush = True)

            print(f'batch {n} repeat {rep} done', flush = True)

    df  = pd.DataFrame(rows)
    out = os.path.join(fc.out_root(), 'closure_matching.csv')
    df.to_csv(out, index = False)
    with pd.option_context('display.width', 200):
        print(df.groupby([ 'batch', 'cost' ], sort = False)
                [[ 'energy_rank_corr', 'median_ratio', 'ratio_iqr',
                   'shape_emd', 'axis_dr' ]].mean().round(3).to_string())
    print(f'wrote {out}')

    plot(shows, cmdargs.figure)

def plot(shows, out):
    # pylint: disable=too-many-locals
    names = [ 'random', 'full', 'centred', 'shape+E l=0', 'shape+E l=1' ]
    (fig, axes) = plt.subplots(4, len(names), figsize = (3.2 * len(names), 12))
    for (k, name) in enumerate(names):
        (ea, eb, a, b) = shows[name]
        ax = axes[0][k]
        ax.scatter(ea, eb, s = 4, alpha = 0.5)
        lim = [ 0, max(ea.max(), eb.max()) ]
        ax.plot(lim, [ 0.8 * x for x in lim ], 'r--', lw = 1,
                label = 'T: 0.8 E')
        ax.set_title(f'{name}: rank corr {spearmanr(ea, eb)[0]:.2f}',
                     fontsize = 8)
        ax.set_xlabel('source E, GeV', fontsize = 7)
        ax.set_ylabel('matched target E, GeV', fontsize = 7)
        ax.legend(fontsize = 6)
        # representative pairs: the median-energy source jets
        order = np.argsort(np.abs(ea - np.median(ea)))[:3]
        for (r, i) in enumerate(order[:3]):
            ax = axes[1 + r][k]
            pair = np.concatenate([
                a[i, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9].numpy(),
                np.full((9, 1), np.nan),
                b[i, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9].numpy() ], axis = 1)
            ax.imshow(np.log10(pair + 0.1), cmap = 'viridis', vmin = -1,
                      vmax = 1.5)
            ax.set_title(f'source {ea[i]:.0f} GeV | matched {eb[i]:.0f} GeV',
                         fontsize = 7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('Closure test: what each cost matches (batch 256, unpaired'
                 ' A vs T(B); log10(E + 0.1) of the cone towers)', fontsize = 9)
    fig.tight_layout()
    fig.savefig(out, dpi = 100)
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
