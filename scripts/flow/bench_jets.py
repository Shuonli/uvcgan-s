#!/usr/bin/env python
"""Reconstructed-jet analysis of the consolidated subtraction benchmark,
after the UVCGAN-S paper (arXiv:2510.23717v2; FLOW_NOTES.md, "Consolidated
benchmark").

The paper's scripts are not public (LS4GAN/uvcgan-s holds training code
only), so this is the minimal documented analysis. Settings from the paper:
- FastJet anti-kT on calorimeter towers, R = 0.2, 0.4, 0.5; jets within
  |eta| < 0.6;
- truth ("real") jets from the detector-level signal image alone;
- matching within dR < 0.75 R, all jets (not only the leading one), in the
  pT range 14-50 GeV;
- soft drop with z_cut 0.1 and beta 0;
- substructure for R = 0.4 at 20 < pT_real < 30 GeV.

Choices the paper leaves open, fixed here:
- a tower is a massless constituent at its centre (eta = -1.1 + (row + 0.5)
  x 2.2/24, phi = (col + 0.5) x 2 pi/64), with the tower value as E_T, as in
  all our earlier analyses;
- towers <= 0 GeV are dropped before clustering, for every method;
- jets above 5 GeV are kept for matching;
- matching is one-to-one, greedily closest in dR first;
- the fake rate is binned in the reconstructed pT;
- C/A reclustering for soft drop, following the harder branch; jets without
  a passing split get no z_g / r_g;
- xi = ln(pT_jet / pT_constituent) over every constituent;
- in addition to the paper's set, pT_D = sqrt(sum pT_i^2) / sum pT_i.

A model `STEM@thrX` (a labelled clean-up, never the primary result) drops
the output's towers below X GeV before clustering; the truth is unchanged.

The Area baseline:
- anti-kT jets of the mixture with active area (ghost area 0.01, |y| <
  1.1);
- rho = median pT/A of the kT (R 0.4) jets within |y| < 0.7, without the
  two hardest;
- pT_sub = pT - rho A; the jet axis is the unsubtracted one;
- no substructure (it does not subtract constituents).
ICS (fjcontrib) is not available here.

    bench_jets.py --set val|jewel --models truth,area,uvcgan__sig,<run>__<setting>__<readout> [--procs 32]

Writes OUTDIR/sphenix/flow/bench/jets/<set>/<model>.npz (every jet above
5 GeV, per R: event, pt, eta, phi, and for R = 0.4 girth, mass, zlead, ptd,
zg, rg, and the xi of its constituents).
"""

import argparse
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.expanduser('~/pyext/jets'))
import fastjet._swig as fj        # pylint: disable=wrong-import-position

RADII  = (0.2, 0.4, 0.5)
DETA   = 2.2 / 24
DPHI   = 2 * math.pi / 64
ETA    = -1.1 + (np.arange(24) + 0.5) * DETA
PHI    = (np.arange(64) + 0.5) * DPHI
PT_MIN = 5.0
ZCUT   = 0.1
COLUMNS = [ 'event', 'pt', 'eta', 'phi', 'girth', 'mass', 'zlead', 'ptd', 'zg',
            'rg' ]

def image_dir(name):
    return os.path.join(os.environ.get('UVCGAN_S_OUTDIR', 'outdir'), 'sphenix',
                        'flow', 'bench', 'images', name)

def jet_dir(name):
    path = os.path.join(os.environ.get('UVCGAN_S_OUTDIR', 'outdir'), 'sphenix',
                        'flow', 'bench', 'jets', name)
    os.makedirs(path, exist_ok = True)
    return path

def particles(img):
    rows, cols = np.nonzero(img > 0)
    out = []
    for (r, c) in zip(rows, cols):
        pt  = float(img[r, c])
        eta = ETA[r]
        phi = PHI[c]
        out.append(fj.PseudoJet(pt * math.cos(phi), pt * math.sin(phi),
                                pt * math.sinh(eta), pt * math.cosh(eta)))
    return out

def soft_drop(jet):
    """(z_g, r_g) of a jet by C/A declustering, or (nan, nan)."""
    cons = jet.constituents()
    if len(cons) < 2:
        return (math.nan, math.nan)
    # C/A with a radius large enough to recluster all constituents
    cs   = fj.ClusterSequence(cons, fj.JetDefinition(fj.cambridge_algorithm, 10.0))
    top  = fj.sorted_by_pt(cs.inclusive_jets())[0]
    (p1, p2) = (fj.PseudoJet(), fj.PseudoJet())
    while top.has_parents(p1, p2):
        (a, b) = (p1, p2) if p1.pt() >= p2.pt() else (p2, p1)
        z = b.pt() / (a.pt() + b.pt())
        if z > ZCUT:
            return (z, a.delta_R(b))
        top = a
        (p1, p2) = (fj.PseudoJet(), fj.PseudoJet())
    return (math.nan, math.nan)

def substructure(jet, radius):
    cons = jet.constituents()
    pts  = np.array([ c.pt() for c in cons ])
    dr   = np.array([ c.delta_R(jet) for c in cons ])
    pt   = jet.pt()
    (zg, rg) = soft_drop(jet)
    return {
        'girth' : float((pts * dr).sum() / pt),
        'mass' : float(jet.m()) if jet.m2() > 0 else 0.0,
        'zlead' : float(pts.max() / pt),
        'ptd' : float(np.sqrt((pts**2).sum()) / pts.sum()),
        'zg' : zg, 'rg' : rg,
        'xi' : np.log(pt / pts),
    }

def area_jets(img, radius):
    """Jets of the mixture, pT - rho A (the Area baseline)."""
    parts = particles(img)
    spec  = fj.GhostedAreaSpec(1.1, 1, 0.01)
    adef  = fj.AreaDefinition(fj.active_area_explicit_ghosts, spec)
    cs    = fj.ClusterSequenceArea(parts, fj.JetDefinition(fj.antikt_algorithm,
                                                           radius), adef)
    sel   = fj.SelectorAbsRapMax(0.7) * (~fj.SelectorNHardest(2))
    bge   = fj.JetMedianBackgroundEstimator(sel, fj.JetDefinition(
        fj.kt_algorithm, 0.4), adef)
    bge.set_particles(parts)
    rho   = bge.rho()
    out   = []
    for j in cs.inclusive_jets(PT_MIN):
        pt = j.pt() - rho * j.area()
        if pt > PT_MIN:
            out.append((pt, j.eta(), j.phi_std()))
    return out

def cluster_chunk(args):
    """Jets of events [start, stop) of one model's images."""
    # pylint: disable=too-many-locals
    (path, model, start, stop) = args
    (stem, thr) = (model.split('@thr')[0], float(model.split('@thr')[1])) \
        if '@thr' in model else (model, None)
    if model == 'area':
        imgs = np.load(os.path.join(path, 'mixture.npy'), mmap_mode = 'r')
    else:
        imgs = np.load(os.path.join(path, f'{stem}.npy'), mmap_mode = 'r')
    rows = { r : [] for r in RADII }
    xis  = []
    for ev_i in range(start, stop):
        img = np.asarray(imgs[ev_i], dtype = np.float64)
        if thr is not None:
            img = np.where(img >= thr, img, 0.0)
        if model == 'area':
            for r in RADII:
                for (pt, eta, phi) in area_jets(img, r):
                    rows[r].append((ev_i, pt, eta, phi) + (math.nan,) * 6)
            continue
        parts = particles(img)
        for r in RADII:
            cs = fj.ClusterSequence(parts, fj.JetDefinition(fj.antikt_algorithm, r))
            for j in fj.sorted_by_pt(cs.inclusive_jets(PT_MIN)):
                rec = (ev_i, j.pt(), j.eta(), j.phi_std())
                if r == 0.4:
                    s = substructure(j, r)
                    rec += (s['girth'], s['mass'], s['zlead'], s['ptd'], s['zg'],
                            s['rg'])
                    xis.append(np.stack([ np.full(len(s['xi']), len(rows[r])),
                                          s['xi'] ], axis = 1))
                else:
                    rec += (math.nan,) * 6
                rows[r].append(rec)
    return (rows, xis)

def cluster(cmdargs):
    path = image_dir(cmdargs.set)
    n    = len(np.load(os.path.join(path, 'truth.npy'), mmap_mode = 'r'))
    n    = min(n, cmdargs.n_events)
    step = max(1, n // (cmdargs.procs * 4))
    for model in cmdargs.models.split(','):
        out_path = os.path.join(jet_dir(cmdargs.set), f'{model}.npz')
        if os.path.exists(out_path) and not cmdargs.force:
            print(f'{out_path} exists, skipped', flush = True)
            continue
        tasks = [ (path, model, s, min(s + step, n)) for s in range(0, n, step) ]
        with Pool(cmdargs.procs) as pool:
            results = pool.map(cluster_chunk, tasks)
        out = {}
        for r in RADII:
            recs = []
            offset = 0
            xi = []
            for (rows, xis) in results:
                recs += rows[r]
                if r == 0.4:
                    for x in xis:
                        x = x.copy()
                        x[:, 0] += offset
                        xi.append(x)
                    offset += len(rows[r])
            a = np.array(recs, dtype = np.float64).reshape(-1, len(COLUMNS))
            key = f'R{int(r * 10)}'
            for (k, col) in zip(COLUMNS, a.T):
                out[f'{key}_{k}'] = col
            if r == 0.4:
                out['R4_xi'] = np.concatenate(xi) if xi else np.zeros((0, 2))
        np.savez_compressed(out_path, **out)
        print(f'{cmdargs.set}: {model} clustered ({len(out["R4_pt"])} R=0.4 jets)',
              flush = True)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Benchmark jet analysis')
    parser.add_argument('--set', default = 'val')
    parser.add_argument('--models', default = 'truth')
    parser.add_argument('--n-events', type = int, default = 20000)
    parser.add_argument('--procs', type = int, default = 32)
    parser.add_argument('--force', action = 'store_true')
    return parser.parse_args()

if __name__ == '__main__':
    cluster(parse_cmdargs())
