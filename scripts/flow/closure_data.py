#!/usr/bin/env python
"""Jet-centred PYTHIA jets and a known modification, for the unpaired
closure test of the jet-to-jet flow (FLOW_NOTES.md, "Closure test").

Jets. PYTHIA events of the signal training cache (clean, no background),
a fixed random subset read in index order. The axis is the tower whose
R = 0.4 cone holds the most energy (as the jet scores,
eval_val_truth.Truth). A jet is kept if the 9 x 9 window around the axis
lies inside the acceptance (axis rows 4-19, |eta| < 0.73) and its cone holds
>= 10 GeV. The jet image J is the towers of the R = 0.4 cone (53 of the
9 x 9 window, substructure.Geometry), in GeV of tower E_T as the images
hold them, placed on a 16 x 16 canvas (the window at rows and columns 4-12,
the axis at 8, 8) so that the backbones can downsample it three times; the
rest of the canvas is zero. Energy observable: E = sum of J.

Modification. T(J) = 0.8 [0.8 J + 0.2 K(J)]. K moves each tower's energy
to its neighbours inside the cone (3 x 3 minus the centre), with weights
exp(-dR^2 / (2 x 0.1^2)) normalised over those neighbours, so that K
preserves the sum. T keeps J >= 0, broadens the jet gently, and gives
E(T(J)) = 0.8 E(J) exactly. It is a method-validation toy, not a quenching
model.

Pools, disjoint by parent event, drawn from a random permutation of the
selected jets:
- train source A (200k);
- train target B (200k), stored as T(B);
- validation (10k pairs J, T(J));
- test (20k pairs).

Selection acts on J before the modification; nothing is reselected after
it. Written to OUTDIR/sphenix/flow/cache/:

    train_closure_src.npy   A     (N, 16, 16) float16, GeV
    train_closure_tgt.npy   T(B)
    closure_meta.npz        parent events, axis rows and columns of A and B
    closure_val.npz, closure_test.npz
                            src = J, tgt = T(J) (float32), parent, row, col
    closure_manifest.json   selection, counts, checks

and the flow's normalisation, psi = log(E + 0.1) standardised over the
training canvases of A and T(B) together: OUTDIR/sphenix/flow/norm_closure.json.

    closure_data.py [--n-train 200000] [--n-val 10000] [--n-test 20000]

--null rebuilds the train target pool B unmodified, from closure_meta.npz,
for the unpaired null test (FLOW_NOTES.md): train_closure_null_tgt.npy,
checked against the stored T(B).

    closure_data.py --null
"""

import argparse
import json
import os

import numpy as np
import torch

import fm_common as fc

ev = fc.ev

CANVAS = 16
OFFSET = 4          # the 9 x 9 window sits at [OFFSET, OFFSET + 9)
HALF   = 4
SIGMA  = 0.1        # width of the broadening kernel, in dR
SCALE  = 0.8        # overall energy factor
KEEP   = 0.8        # share of each tower's energy kept in place

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Closure-test jets')
    parser.add_argument('--n-train', type = int, default = 200000)
    parser.add_argument('--n-val', type = int, default = 10000)
    parser.add_argument('--n-test', type = int, default = 20000)
    parser.add_argument('--min-jet', type = float, default = 10.0)
    parser.add_argument('--read', type = int, default = 600000,
        help = 'events read from the signal cache')
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--chunk', type = int, default = 20000)
    parser.add_argument('--null', action = 'store_true',
        help = 'rebuild the train target pool B unmodified (null test)')
    return parser.parse_args()

def cone_mask():
    """(9, 9) bool: the towers of the R = 0.4 cone (substructure.Geometry)."""
    return (ev.cone_kernel(ev.R_JET) > 0.5).cpu().numpy()

def broadening_matrix(mask):
    """(81, 81) K[i, j]: share of window tower j's energy that goes to i."""
    off  = np.arange(-HALF, HALF + 1)
    rows = np.repeat(off, 9)
    cols = np.tile(off, 9)
    inside = mask.flatten()

    kmat = np.zeros((81, 81))
    for j in np.nonzero(inside)[0]:
        dr = rows - rows[j]
        dc = cols - cols[j]
        near = (np.abs(dr) <= 1) & (np.abs(dc) <= 1) & ((dr != 0) | (dc != 0))
        near &= inside
        w = np.exp(-((dr * ev.DETA)**2 + (dc * ev.DPHI)**2) / (2 * SIGMA**2))
        w = np.where(near, w, 0.0)
        assert w.sum() > 0
        kmat[:, j] = w / w.sum()
    return kmat

class Modification:
    """T(J) = SCALE [KEEP J + (1 - KEEP) K(J)] on the canvas."""

    def __init__(self, device):
        self.mask = torch.from_numpy(cone_mask()).to(device)
        self.kmat = torch.from_numpy(broadening_matrix(cone_mask())) \
            .float().to(device)

    def __call__(self, canvas):
        win  = canvas[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9].reshape(-1, 81)
        moved = win @ self.kmat.T
        new  = SCALE * (KEEP * win + (1 - KEEP) * moved)
        out  = canvas.clone()
        out[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] = new.reshape(-1, 9, 9)
        return out

def select(images, kernel, mask, min_jet):
    """Axes, cone energies and canvases of the jets kept among (N, 24, 64)
    images (GeV, on the GPU)."""
    sums = ev.cone_sums(images, kernel)
    flat = sums.flatten(1).argmax(dim = 1)
    row  = flat // sums.shape[2]
    col  = flat %  sums.shape[2]
    n    = torch.arange(len(images), device = images.device)
    e    = sums[n, row, col]
    keep = (row >= HALF) & (row < fc.SHAPE[0] - HALF) & (e >= min_jet)

    idx = torch.nonzero(keep).flatten()
    off = torch.arange(-HALF, HALF + 1, device = images.device)
    r = row[idx][:, None, None] + off[None, :, None]
    c = (col[idx][:, None, None] + off[None, None, :]) % fc.SHAPE[1]
    k = idx[:, None, None]
    win = images[k, r, c].clamp(min = 0) * mask

    canvas = torch.zeros((len(idx), CANVAS, CANVAS), device = images.device)
    canvas[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] = win
    return (idx, row[idx], col[idx], canvas)

def windows(images, rows, cols, mask):
    """Canvases of the cone around known axes of (N, 24, 64) images."""
    off = torch.arange(-HALF, HALF + 1, device = images.device)
    r = rows[:, None, None] + off[None, :, None]
    c = (cols[:, None, None] + off[None, None, :]) % fc.SHAPE[1]
    k = torch.arange(len(images), device = images.device)[:, None, None]
    canvas = torch.zeros((len(images), CANVAS, CANVAS), device = images.device)
    canvas[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] = \
        images[k, r, c].clamp(min = 0) * mask
    return canvas

def build_null(cmdargs, device):
    cache  = os.path.dirname(fc.cache_path('signal'))
    meta   = np.load(os.path.join(cache, 'closure_meta.npz'))
    data   = np.load(fc.cache_path('signal'), mmap_mode = 'r')
    stored = np.load(os.path.join(cache, 'train_closure_tgt.npy'),
                     mmap_mode = 'r')
    mask   = torch.from_numpy(cone_mask()).to(device)
    modify = Modification(device)

    parent = meta['tgt_parent']
    order  = np.argsort(parent)
    out    = np.zeros((len(parent), CANVAS, CANVAS), dtype = np.float16)
    worst  = 0.0
    for start in range(0, len(order), cmdargs.chunk):
        sel = order[start:start + cmdargs.chunk]
        img = torch.from_numpy(data[parent[sel]].astype(np.float32)).to(device)
        can = windows(img, torch.as_tensor(meta['tgt_row'][sel], device = device),
                      torch.as_tensor(meta['tgt_col'][sel], device = device),
                      mask)
        out[sel] = can.cpu().numpy().astype(np.float16)
        ref = torch.from_numpy(stored[sel].astype(np.float32)).to(device)
        diff = (modify(can) - ref).abs().max() / ref.abs().max()
        worst = max(worst, float(diff))
    np.save(os.path.join(cache, 'train_closure_null_tgt.npy'), out)
    print(f'wrote train_closure_null_tgt.npy ({len(out)} jets); T of the'
          f' rebuilt jets against the stored T(B): max relative difference'
          f' {worst:.2e} (float16 storage)')

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    if cmdargs.null:
        build_null(cmdargs, device)
        return
    rng     = np.random.default_rng(cmdargs.seed)
    kernel  = ev.cone_kernel(ev.R_JET).to(device)
    mask    = torch.from_numpy(cone_mask()).to(device)
    modify  = Modification(device)

    data   = np.load(fc.cache_path('signal'), mmap_mode = 'r')
    events = np.sort(rng.choice(len(data), cmdargs.read, replace = False))

    parts = { 'parent' : [], 'row' : [], 'col' : [], 'canvas' : [] }
    for start in range(0, len(events), cmdargs.chunk):
        ev_idx = events[start:start + cmdargs.chunk]
        images = torch.from_numpy(data[ev_idx].astype(np.float32)).to(device)
        (idx, row, col, canvas) = select(images, kernel, mask,
                                         cmdargs.min_jet)
        parts['parent'].append(ev_idx[idx.cpu().numpy()])
        parts['row'].append(row.cpu().numpy())
        parts['col'].append(col.cpu().numpy())
        parts['canvas'].append(canvas.cpu())
        print(f'read {start + len(ev_idx)} events', flush = True)

    parent = np.concatenate(parts['parent'])
    row    = np.concatenate(parts['row'])
    col    = np.concatenate(parts['col'])
    canvas = torch.cat(parts['canvas'])
    n_sel  = len(parent)

    need = 2 * cmdargs.n_train + cmdargs.n_val + cmdargs.n_test
    assert n_sel >= need, (n_sel, need)
    order  = rng.permutation(n_sel)
    bounds = np.cumsum([ 0, cmdargs.n_train, cmdargs.n_train,
                         cmdargs.n_val, cmdargs.n_test ])
    pools  = dict(zip([ 'src', 'tgt', 'val', 'test' ], [
        order[bounds[k]:bounds[k + 1]] for k in range(4) ]))

    # parent events are disjoint by construction; check anyway
    seen = np.concatenate([ parent[p] for p in pools.values() ])
    assert len(np.unique(seen)) == len(seen)

    cache = os.path.dirname(fc.cache_path('signal'))
    checks = {}

    def modified(p):
        out = []
        for start in range(0, len(p), cmdargs.chunk):
            j = canvas[p[start:start + cmdargs.chunk]].to(device)
            out.append(modify(j).cpu())
        return torch.cat(out)

    src = canvas[pools['src']]
    tgt = modified(pools['tgt'])
    np.save(os.path.join(cache, 'train_closure_src.npy'),
            src.numpy().astype(np.float16))
    np.save(os.path.join(cache, 'train_closure_tgt.npy'),
            tgt.numpy().astype(np.float16))
    np.savez(os.path.join(cache, 'closure_meta.npz'),
             src_parent = parent[pools['src']], src_row = row[pools['src']],
             src_col = col[pools['src']], tgt_parent = parent[pools['tgt']],
             tgt_row = row[pools['tgt']], tgt_col = col[pools['tgt']])

    for name in [ 'val', 'test' ]:
        p = pools[name]
        (j, tj) = (canvas[p], modified(p))
        ratio = (tj.sum((1, 2)) / j.sum((1, 2))).numpy()
        checks[name] = {
            'n' : int(len(p)),
            'min_tower_T' : float(tj.min()),
            'energy_ratio_min' : float(ratio.min()),
            'energy_ratio_max' : float(ratio.max()),
            'outside_cone_max' : float(
                (tj * (1 - torch.nn.functional.pad(
                    mask.cpu().float(), (OFFSET, CANVAS - OFFSET - 9,
                                         OFFSET, CANVAS - OFFSET - 9)))).abs().max()),
            'median_E_J' : float(np.median(j.sum((1, 2)).numpy())),
        }
        np.savez(os.path.join(cache, f'closure_{name}.npz'),
                 src = j.numpy(), tgt = tj.numpy(), parent = parent[p],
                 row = row[p], col = col[p])

    ratio16 = (torch.from_numpy(np.load(os.path.join(
        cache, 'train_closure_tgt.npy')).astype(np.float32)).sum((1, 2))
        / canvas[pools['tgt']].sum((1, 2))).numpy()
    checks['train_tgt_float16'] = {
        'energy_ratio_min' : float(ratio16.min()),
        'energy_ratio_max' : float(ratio16.max()),
        'min_tower_T' : float(tgt.min()),
    }

    # the flow's normalisation: psi = log(E + 0.1) over A and T(B)
    both = torch.cat([ src, tgt ]).double()
    psi  = torch.log(both + fc.BIAS)
    norm = { 'jet' : (float(psi.mean()), float(psi.std())), '_bias' : fc.BIAS }
    with open(os.path.join(fc.out_root(), 'norm_closure.json'), 'w',
              encoding = 'utf-8') as f:
        json.dump(norm, f, indent = 4)

    manifest = {
        'read_events' : int(cmdargs.read), 'selected' : int(n_sel),
        'selection' : f'leading cone (R = 0.4) >= {cmdargs.min_jet} GeV,'
                      ' axis rows 4-19 (9 x 9 window inside the acceptance)',
        'canvas' : f'{CANVAS} x {CANVAS}, 9 x 9 window at {OFFSET}..{OFFSET + 8},'
                   ' R = 0.4 cone towers only, GeV',
        'modification' : f'T(J) = {SCALE} [{KEEP} J + {1 - KEEP:.1f} K(J)],'
                         f' K: 8 in-cone neighbours, exp(-dR^2 / 2 {SIGMA}^2),'
                         ' normalised per source tower',
        'pools' : { k : int(len(v)) for (k, v) in pools.items() },
        'seed' : cmdargs.seed, 'checks' : checks, 'norm' : norm,
    }
    with open(os.path.join(cache, 'closure_manifest.json'), 'w',
              encoding = 'utf-8') as f:
        json.dump(manifest, f, indent = 4)
    print(json.dumps(manifest, indent = 2))

if __name__ == '__main__':
    main()
