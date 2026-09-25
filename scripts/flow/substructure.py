#!/usr/bin/env python
"""Jet substructure of the extracted signal, against truth.

Each jet is the R = 0.4 cone (53 towers) around the true leading-jet axis
of a val (PYTHIA) or JEWEL mixed event, as in eval_val_truth.py; the same
towers are read from the truth image and from every extracted image, so
the comparison isolates the extraction, not the jet finding. Towers are
massless constituents with transverse energy = tower value (negative
towers of a subtraction are dropped). Observables:

    et       cone transverse energy (as studied so far)
    mass     invariant mass of the constituents
    girth    sum(et_i dR_i) / et, dR to the jet axis
    ptd      sqrt(sum et_i^2) / et
    core     fraction of et in the central 3 x 3 towers (dR < 0.15)
    zlead    leading tower fraction
    n1       towers above 1 GeV
    zg, rg   soft drop (C/A reclustering, zcut 0.1, beta 0, R0 0.4)

For every model and observable: the relative shift of the mean, the
Wasserstein-1 distance between the model's and the true distributions in
units of the true standard deviation, the per-jet correlation and the
per-jet resolution sigma(model - truth) / sigma(truth). Read-outs with a
tower threshold are also compared to the truth thresholded the same way.
And the quenching signal: the JEWEL-minus-PYTHIA difference of each mean
for jets of true cone energy 20-30 GeV, from the model against from truth.

    substructure.py [--n-events 10000] [--condcfm-samples 16]
                    [--extra LABEL=RUN:READOUT ...] [--skip-condcfm]

`--extra` adds models: RUN is a run directory under OUTDIR/sphenix/flow,
READOUT either a read-out of readout_test.py (needs its cached backgrounds:
run readout_test.py on that run first) or `direct` / `mixture`, the run's
decoding at its selection setting (fm_common.SELECTION).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

import fm_common as fc
from readout_test import readout, reference_images, fc_best_step

ev = fc.ev

OBSERVABLES = [ 'et', 'mass', 'girth', 'ptd', 'core', 'zlead', 'n1',
                'zg', 'rg' ]
OT_READOUTS = [ 'coarse', 'converged', 'coarse_thr0.5', 'seeded10_zero',
                'seeded8_zero_thr0.5', 'seeded10_zero_thr0.7' ]
WINDOW = (20.0, 30.0)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Jet substructure')
    parser.add_argument('--n-events', type = int, default = 10000)
    parser.add_argument('--batch', type = int, default = 500)
    parser.add_argument('--condcfm-samples', type = int, default = 16)
    parser.add_argument('--otcfm', default = 'pilot_otcfm_s0')
    parser.add_argument('--condcfm', default = 'ext_condcfm_s0')
    parser.add_argument('--out', default = None)
    parser.add_argument('--extra', nargs = '*', default = [],
        help = 'LABEL=RUN:READOUT models to add')
    parser.add_argument('--skip-condcfm', action = 'store_true')
    return parser.parse_args()

# --- soft drop: C/A reclustering of up to 53 towers, compiled if possible

def _soft_drop(pt, y, p, zcut, r0):
    n = len(pt)
    if n < 2:
        return (np.nan, np.nan)

    m    = 2 * n - 1
    npt  = np.zeros(m)
    ny   = np.zeros(m)
    nphi = np.zeros(m)
    left = -np.ones(m, np.int64)
    rght = -np.ones(m, np.int64)
    ndr  = np.zeros(m)
    act  = np.zeros(m, np.bool_)

    for i in range(n):
        npt[i]  = pt[i]
        ny[i]   = y[i]
        nphi[i] = p[i]
        act[i]  = True

    nxt = n
    for _ in range(n - 1):
        best = 1e30
        (ia, ib) = (-1, -1)
        for a in range(nxt):
            if not act[a]:
                continue
            for b in range(a + 1, nxt):
                if not act[b]:
                    continue
                d = (ny[a] - ny[b])**2 + (nphi[a] - nphi[b])**2
                if d < best:
                    best = d
                    (ia, ib) = (a, b)

        s = npt[ia] + npt[ib]
        npt[nxt]  = s
        ny[nxt]   = (npt[ia] * ny[ia] + npt[ib] * ny[ib]) / s
        nphi[nxt] = (npt[ia] * nphi[ia] + npt[ib] * nphi[ib]) / s
        left[nxt] = ia
        rght[nxt] = ib
        ndr[nxt]  = np.sqrt(best)
        act[ia]   = False
        act[ib]   = False
        act[nxt]  = True
        nxt += 1

    node = nxt - 1
    while left[node] >= 0:
        (a, b) = (left[node], rght[node])
        z = min(npt[a], npt[b]) / (npt[a] + npt[b])
        if z > zcut:
            return (z, ndr[node])
        node = a if npt[a] >= npt[b] else b

    return (np.nan, np.nan)

try:
    import numba
    _soft_drop = numba.njit(cache = True)(_soft_drop)
except ImportError:                                  # pragma: no cover
    pass

def soft_drop(w, deta, dphi, zcut = 0.1, r0 = 0.4):
    """(zg, rg) of each jet; nan when no splitting passes."""
    out = np.full((len(w), 2), np.nan)
    for (k, wk) in enumerate(w):
        sel = wk > 0
        out[k] = _soft_drop(wk[sel], deta[sel], dphi[sel], zcut, r0)
    return out

# --- observables

class Geometry:
    """The 9 x 9 towers around a jet axis and their offsets."""

    def __init__(self, device):
        self.mask = ev.cone_kernel(ev.R_JET).to(device) > 0.5      # (9, 9)
        half = self.mask.shape[0] // 2
        off  = torch.arange(-half, half + 1, device = device)
        self.off  = off
        self.deta = (off[:, None] * ev.DETA).expand(9, 9).float()
        self.dphi = (off[None, :] * ev.DPHI).expand(9, 9).float()
        self.dr   = (self.deta**2 + self.dphi**2).sqrt()

    def patches(self, images, truth, idx):
        """(N, 9, 9) towers around the true axes of the events idx."""
        rows = truth.row[idx]
        cols = truth.col[idx]
        r = rows[:, None, None] + self.off[None, :, None]
        c = (cols[:, None, None] + self.off[None, None, :]) % fc.SHAPE[1]
        n = torch.arange(len(idx), device = rows.device)[:, None, None]
        return images[idx.cpu()].to(rows.device)[n, r, c]

def observables(w, truth, idx, geo):
    """Per-jet observables of (N, 9, 9) constituent patches."""
    w  = w.clamp(min = 0) * geo.mask
    et = w.sum((1, 2))
    safe = et.clamp(min = 1e-6)

    eta0 = -1.1 + (truth.row[idx].float() + 0.5) * ev.DETA
    eta  = eta0[:, None, None] + geo.deta
    e  = (w * torch.cosh(eta)).sum((1, 2))
    px = (w * torch.cos(geo.dphi)).sum((1, 2))
    py = (w * torch.sin(geo.dphi)).sum((1, 2))
    pz = (w * torch.sinh(eta)).sum((1, 2))
    mass = (e**2 - px**2 - py**2 - pz**2).clamp(min = 0).sqrt()

    result = {
        'et'    : et,
        'mass'  : mass,
        'girth' : (w * geo.dr).sum((1, 2)) / safe,
        'ptd'   : (w**2).sum((1, 2)).sqrt() / safe,
        'core'  : (w * (geo.dr < 0.15)).sum((1, 2)) / safe,
        'zlead' : w.flatten(1).amax(1) / safe,
        'n1'    : (w > 1.0).sum((1, 2)).float(),
    }
    result = { k : v.cpu().numpy() for (k, v) in result.items() }

    sd = soft_drop(
        w.flatten(1).cpu().numpy().astype(np.float64),
        geo.deta.flatten().cpu().numpy().astype(np.float64),
        geo.dphi.flatten().cpu().numpy().astype(np.float64),
    )
    result['zg'] = sd[:, 0]
    result['rg'] = sd[:, 1]

    return result

def jet_observables(images, truth, geo):
    idx = torch.nonzero(truth.jets).flatten()
    return observables(geo.patches(images, truth, idx), truth, idx, geo)

# --- the models' signal images

@torch.no_grad()
def model_images(cmdargs, embed, truth_name, device):
    kernel = ev.cone_kernel(ev.R_JET).to(device)
    images = {}

    for (label, img) in reference_images(embed, cmdargs.batch, device).items():
        images[label] = img

    for name in OT_READOUTS:
        images[f'OT-CFM {name}'] = readout_images(
            cmdargs.otcfm, name, embed, truth_name, cmdargs.batch, device,
            kernel
        )

    for spec in cmdargs.extra:
        (label, rest) = spec.split('=', 1)
        (run, name)   = rest.split(':', 1)
        if name in ('direct', 'mixture'):
            images[label] = decoded_images(
                run, name, embed, cmdargs.batch, device
            )
        else:
            images[label] = readout_images(
                run, name, embed, truth_name, cmdargs.batch, device, kernel
            )

    if cmdargs.skip_condcfm:
        return images

    run  = os.path.join(fc.out_root(), cmdargs.condcfm)
    step = fc_best_step(run)
    ckpt = [ c for c in fc.list_checkpoints(run)
             if c.endswith(f'step_{step:08d}.pt') ][0]
    (method, net, _, _) = fc.load_run(run, ckpt, device, 'ema')

    for samples in [ 1, cmdargs.condcfm_samples ]:
        dec = fc.Decomposer(method, net, nfe = 8, samples = samples)
        out = []
        for start in range(0, len(embed), cmdargs.batch):
            m = torch.from_numpy(embed[start:start + cmdargs.batch])
            out.append(dec(m.to(device).float().unsqueeze(1))[:, 1].cpu())
        label = '1 sample' if samples == 1 else f'mean of {samples}'
        images[f'cond. CFM, {label}'] = torch.cat(out)

    return images

def readout_images(run, name, embed, truth_name, batch, device, kernel):
    """A read-out of readout_test.py from its cached backgrounds."""
    run  = os.path.join(fc.out_root(), run)
    step = fc_best_step(run)
    bkg  = torch.load(os.path.join(
        run, 'evals', f'readout_{truth_name}_s{step:08d}_n20000.pt'
    ))
    out = []
    for start in range(0, len(embed), batch):
        m  = torch.from_numpy(embed[start:start + batch]).to(device).float()
        b4 = bkg['b4'][start:start + len(m)].to(device)
        bc = bkg['bc'][start:start + len(m)].to(device)
        out.append(readout(name, m, b4, bc, kernel)[0].cpu())
    return torch.cat(out)

@torch.no_grad()
def decoded_images(run, decode, embed, batch, device):
    """The run's signal at its selection setting, read `direct` or as
    `mixture` - background."""
    run  = os.path.join(fc.out_root(), run)
    step = fc_best_step(run)
    ckpt = [ c for c in fc.list_checkpoints(run)
             if c.endswith(f'step_{step:08d}.pt') ][0]
    (method, net, _, _) = fc.load_run(run, ckpt, device, 'ema')
    (nfe, solver, _, samples) = fc.SELECTION[method.name]
    dec = fc.Decomposer(method, net, nfe = nfe, solver = solver,
                        decode = decode, samples = samples)
    out = []
    for start in range(0, len(embed), batch):
        m = torch.from_numpy(embed[start:start + batch])
        out.append(dec(m.to(device).float().unsqueeze(1))[:, 1].cpu())
    return torch.cat(out)

# --- comparison

def compare(x, t):
    ok = np.isfinite(x) & np.isfinite(t)
    (x, t) = (x[ok], t[ok])
    sd = t.std()
    return {
        'n'          : int(ok.sum()),
        'mean_model' : float(x.mean()),
        'mean_truth' : float(t.mean()),
        'rel_shift'  : float((x.mean() - t.mean()) / abs(t.mean())),
        'w1_sigma'   : float(wasserstein_distance(x, t) / sd),
        'corr'       : float(np.corrcoef(x, t)[0, 1]),
        'rel_res'    : float((x - t).std() / sd),
    }

def reference_truth(name):
    for thr in ('0.5', '0.7'):
        if f'thr{thr}' in name:
            return f'truth, towers > {thr} GeV'
    return 'truth'

def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    geo     = Geometry(device)
    rows    = []
    means   = []
    dists   = {}

    for truth_name in [ 'val', 'jewel' ]:
        (embed, signal) = ev.load_pairs(
            os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0,
            truth = truth_name
        )
        (embed, signal) = (embed[:cmdargs.n_events], signal[:cmdargs.n_events])
        truth = ev.Truth(embed, signal, ev.cone_kernel(ev.R_JET), 10.0, device)

        sig = torch.from_numpy(signal).float()
        images = {
            'truth' : sig,
            'truth, towers > 0.5 GeV' : torch.where(sig > 0.5, sig, 0 * sig),
            'truth, towers > 0.7 GeV' : torch.where(sig > 0.7, sig, 0 * sig),
        }
        images.update(model_images(cmdargs, embed, truth_name, device))

        obs = { k : jet_observables(v, truth, geo) for (k, v) in images.items() }
        dists[truth_name] = obs
        et_true = obs['truth']['et']
        window  = (et_true >= WINDOW[0]) & (et_true < WINDOW[1])

        for (name, o) in obs.items():
            for q in OBSERVABLES:
                x = o[q]
                means.append({
                    'truth_set' : truth_name, 'model' : name,
                    'observable' : q,
                    'mean_window' : float(np.nanmean(x[window])),
                    'sd_fail' : float(np.mean(np.isnan(o['zg']))),
                })
                if name.startswith('truth'):
                    continue
                for ref in sorted({ 'truth', reference_truth(name) }):
                    rows.append({
                        'truth_set' : truth_name, 'model' : name,
                        'reference' : ref, 'observable' : q,
                        **compare(x, obs[ref][q]),
                    })

            print(f'{truth_name}: {name} done', flush = True)

    df  = pd.DataFrame(rows)
    mod = pd.DataFrame(means)
    out = cmdargs.out or os.path.join(fc.out_root(), 'substructure')
    df.to_csv(f'{out}.csv', index = False)

    # the quenching signal: JEWEL minus PYTHIA at the same true energy
    pv = mod.pivot_table(index = [ 'model', 'observable' ],
                         columns = 'truth_set', values = 'mean_window')
    pv['delta'] = pv['jewel'] - pv['val']
    truth_delta = pv.loc['truth', 'delta']
    pv['delta_over_truth'] = pv.apply(
        lambda r: r['delta'] / truth_delta[r.name[1]], axis = 1
    )
    pv.reset_index().to_csv(f'{out}_modification.csv', index = False)

    np.savez_compressed(f'{out}_distributions.npz', **{
        f'{t}|{m}|{q}' : v for (t, o) in dists.items()
            for (m, oo) in o.items() for (q, v) in oo.items()
    })

    show = df[df.reference == 'truth'].pivot_table(
        index = [ 'truth_set', 'model' ], columns = 'observable',
        values = 'w1_sigma'
    )[OBSERVABLES]
    with pd.option_context('display.width', 250):
        print('\nW1 / sigma(truth), against the full truth:')
        print(show.round(3).to_string())
        print('\nper-jet resolution / sigma(truth), against the full truth:')
        print(df[df.reference == 'truth'].pivot_table(
            index = [ 'truth_set', 'model' ], columns = 'observable',
            values = 'rel_res')[OBSERVABLES].round(3).to_string())
        print('\nJEWEL - PYTHIA, true cone energy 20-30 GeV, as a fraction'
              ' of the true difference:')
        print(pv['delta_over_truth'].unstack()[OBSERVABLES].round(2).to_string())
    print(f'wrote {out}.csv, {out}_modification.csv, {out}_distributions.npz')

if __name__ == '__main__':
    main()
