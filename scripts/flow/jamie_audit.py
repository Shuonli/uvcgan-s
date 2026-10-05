#!/usr/bin/env python
"""The toy study's matching audit (FLOW_NOTES.md, "Jamie's toy exercises
with OT flow matching"), on training pools only, before any training.

For each unpaired problem, 16 fixed batches (seed 20261005 + k) of 256
source and 256 target training images, paired as the trainer pairs them
(fm_common.RowCoupling on the squared L2 of the standardised states, every
tower; one target per source row), against random pairs:
- the cost and its share from towers near a jet (R < 1.0 of the source axis
  or of the target axis) against the far region;
- total and cone (R < 0.4 around the source axis) energy differences;
- the jet-axis displacement between source and target (clean and UE
  translation); the girth and cone-energy correlations;
- whether the UE of the pair matches: correlations of the event
  multiplicity m, of v2 and of the event-plane angle (cos 2 dpsi) between
  the source's UE and the target's (subtraction: the mixture's UE against the
  UE target).
Also the entropic regularisation of the soft plan: bisection to a median row
effective support exp(-sum q log q) of 4 targets (accepted 3-5), on the
clean-translation batches, rounded to two digits and frozen.

    jamie_audit.py [--out docs/flow/jamie_otfm]
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch

import fm_common as fc
import jamie_methods as jm
import jamie_obs as jo

AUDIT_SEED = 20261005
PROBLEMS = {
    # name: (source pool, target pool, transform)
    'E1-U: mixture -> UE' : ('e1_mix', 'e1_ue', 'sub'),
    'E3: vacuum -> quenched' : ('e3_vac', 'e3_med', 'clean'),
    'E3g: vacuum -> quenched (girth rule)' : ('e3_vac', 'e3g_med', 'clean'),
    'E4: vacuum + UE -> quenched + UE' : ('e4_src', 'e4_tgt', 'ue'),
    'hybrid: vacuum + UE data -> UE' : ('e4_src', 'e1_ue', 'sub'),
    'hybrid: quenched + UE data -> UE' : ('e4_tgt', 'e1_ue', 'sub'),
}

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Toy matching audit')
    parser.add_argument('--batches', type = int, default = 16)
    parser.add_argument('--batch', type = int, default = 256)
    parser.add_argument('--support', type = float, default = 4.0)
    parser.add_argument('--out', default = 'docs/flow/jamie_otfm')
    return parser.parse_args()

def pool(name):
    d = os.path.join(jm.jamie_root(), 'cache')
    x = np.load(os.path.join(d, f'{name}.npy'), mmap_mode = 'r')
    with np.load(os.path.join(d, f'{name}_meta.npz')) as f:
        meta = { k : f[k] for k in f.files }
    return (x, meta)

def batches(src, tgt, k, n):
    rng = np.random.default_rng(AUDIT_SEED + k)
    i = np.sort(rng.choice(len(src[0]), n, replace = False))
    j = np.sort(rng.choice(len(tgt[0]), n, replace = False))
    return (i, j)

def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan

def audit_problem(name, spec, cmdargs, norm, device):
    # pylint: disable=too-many-locals
    (sname, tname, kind) = spec
    (src, tgt) = (pool(sname), pool(tname))
    gen = torch.Generator(device = device).manual_seed(AUDIT_SEED)
    exact = fc.RowCoupling('exact')
    rows = { 'OT (exact)' : [], 'random' : [] }
    for k in range(cmdargs.batches):
        (i, j) = batches(src, tgt, k, cmdargs.batch)
        xs = torch.as_tensor(np.asarray(src[0][i], np.float32), device = device)
        xt = torch.as_tensor(np.asarray(tgt[0][j], np.float32), device = device)
        zs = norm.z(xs, kind).flatten(1)
        zt = norm.z(xt, kind).flatten(1)
        ax_s = torch.tensor(np.stack([ src[1]['eta0'][i], src[1]['phi0'][i] ], 1),
                            dtype = torch.float32, device = device)
        ax_t = torch.tensor(np.stack([ tgt[1]['eta0'][j], tgt[1]['phi0'][j] ], 1),
                            dtype = torch.float32, device = device)
        jsel = { 'OT (exact)' : exact(zs, zt, gen)[2],
                 'random' : torch.randperm(cmdargs.batch, device = device, generator = gen) }
        for (lab, jj) in jsel.items():
            d2 = (zs - zt[jj])**2
            has_s = torch.isfinite(ax_s[:, 0])
            has_t = torch.isfinite(ax_t[jj, 0])
            near = torch.zeros_like(d2, dtype = torch.bool)
            if has_s.any():
                (_, _, dr_s) = jm.offsets(torch.nan_to_num(ax_s))
                near |= ((dr_s < 1.0) & has_s[:, None, None]).flatten(1)
            if has_t.any():
                (_, _, dr_t) = jm.offsets(torch.nan_to_num(ax_t[jj]))
                near |= ((dr_t < 1.0) & has_t[:, None, None]).flatten(1)
            cost = d2.sum(1)
            e_s = xs.sum((1, 2))
            e_t = xt[jj].sum((1, 2))
            r = { 'cost' : cost.cpu().numpy(), 'near_share' : ((d2 * near).sum(1) / cost).cpu().numpy(),
                  'dE_total' : (e_t - e_s).cpu().numpy() }
            if has_s.any():
                (_, _, dr_s) = jm.offsets(torch.nan_to_num(ax_s))
                cone = dr_s < 0.4
                r['dE_cone_src_axis'] = ((xt[jj] - xs) * cone).sum((1, 2)).cpu().numpy()
            if has_s.any() and has_t.any():
                a1 = ax_s.cpu().numpy()
                a2 = ax_t[jj].cpu().numpy()
                dphi = np.angle(np.exp(1j * (a2[:, 1] - a1[:, 1])))
                r['axis_dr'] = np.sqrt((a2[:, 0] - a1[:, 0])**2 + dphi**2)
            r['jj'] = jj.cpu().numpy()
            r['i'] = i
            r['j'] = j
            rows[lab].append(r)
    out = []
    for (lab, rs) in rows.items():
        cat = { k : np.concatenate([ r[k] for r in rs ]) for k in rs[0] if k not in ('jj', 'i', 'j') }
        si = np.concatenate([ r['i'] for r in rs ])
        tj = np.concatenate([ r['j'][r['jj']] for r in rs ])
        row = { 'problem' : name, 'pairs' : lab, 'n' : len(si), 'source' : sname, 'target' : tname,
                'transform' : kind, 'cost_mean' : float(cat['cost'].mean()),
                'near_share_of_cost' : float(cat['near_share'].mean()),
                'dE_total_mean' : float(cat['dE_total'].mean()),
                'dE_total_rms' : float(np.sqrt(np.mean(cat['dE_total']**2))) }
        if 'dE_cone_src_axis' in cat:
            row['dE_cone_rms'] = float(np.sqrt(np.mean(cat['dE_cone_src_axis']**2)))
        if 'axis_dr' in cat:
            row['axis_dr_median'] = float(np.median(cat['axis_dr']))
            row['axis_dr_lt_0p2'] = float(np.mean(cat['axis_dr'] < 0.2))
            row['corr_E'] = corr(src[1]['E'][si], tgt[1]['E'][tj])
            row['corr_g'] = corr(src[1]['g'][si], tgt[1]['g'][tj])
        for q in ('m', 'v2'):
            if np.isfinite(src[1][q]).any() and np.isfinite(tgt[1][q]).any():
                row[f'corr_ue_{q}'] = corr(src[1][q][si], tgt[1][q][tj])
        if np.isfinite(src[1]['psi']).any() and np.isfinite(tgt[1]['psi']).any():
            row['mean_cos2dpsi'] = float(np.nanmean(np.cos(2 * (src[1]['psi'][si] - tgt[1]['psi'][tj]))))
        out.append(row)
        print(row, flush = True)
    return out

def entropic_reg(cmdargs, norm, device, spec):
    """The soft plan's regularisation: median row support of 4."""
    (sname, tname, kind) = spec
    (src, tgt) = (pool(sname), pool(tname))
    states = []
    for k in range(cmdargs.batches):
        (i, j) = batches(src, tgt, k, cmdargs.batch)
        zs = norm.z(torch.as_tensor(np.asarray(src[0][i], np.float32), device = device), kind)
        zt = norm.z(torch.as_tensor(np.asarray(tgt[0][j], np.float32), device = device), kind)
        states.append((zs.unsqueeze(1), zt.unsqueeze(1)))
    costs = torch.stack([ torch.cdist(a.flatten(1), b.flatten(1))**2 for (a, b) in states ])
    def median(reg):
        cpl = fc.RowCoupling('entropic', reg, 1e-4)
        return float(torch.cat([ fc.effective_support(cpl.log_plan(a, b)) for (a, b) in states ]).median())
    grid = []
    for reg in [ 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ]:
        grid.append({ 'reg' : reg, 'median_support' : median(reg) })
        print(grid[-1], flush = True)
    lo = max(g['reg'] for g in grid if g['median_support'] < cmdargs.support)
    hi = min(g['reg'] for g in grid if g['median_support'] > cmdargs.support)
    for _ in range(30):
        mid = math.sqrt(lo * hi)
        (lo, hi) = (mid, hi) if median(mid) < cmdargs.support else (lo, mid)
        if hi / lo < 1.001:
            break
    reg = float(f'{math.sqrt(lo * hi):.2g}')
    cpl = fc.RowCoupling('entropic', reg, 1e-4)
    supp = torch.cat([ fc.effective_support(cpl.log_plan(a, b)) for (a, b) in states ]).cpu().numpy()
    q = np.quantile(supp, [ 0.05, 0.25, 0.5, 0.75, 0.95 ])
    return { 'problem' : f'{sname} -> {tname}', 'reg' : reg, 'accepted' : bool(3 <= q[2] <= 5),
             'support_quantiles_5_25_50_75_95' : q.tolist(),
             'cost_median' : float(costs.median()), 'grid' : grid }

def main():
    cmdargs = parse_cmdargs()
    device = torch.device('cuda')
    norm = jm.load_norm()
    rows = []
    for (name, spec) in PROBLEMS.items():
        rows += audit_problem(name, spec, cmdargs, norm, device)
    os.makedirs(cmdargs.out, exist_ok = True)
    pd.DataFrame(rows).to_csv(os.path.join(cmdargs.out, 'matching_audit.csv'), index = False)
    soft = { 'E3' : entropic_reg(cmdargs, norm, device, PROBLEMS['E3: vacuum -> quenched']) }
    with open(os.path.join(cmdargs.out, 'entropic_reg.json'), 'w', encoding = 'utf-8') as f:
        json.dump(soft, f, indent = 4)
    print(json.dumps(soft, indent = 4))

if __name__ == '__main__':
    main()
