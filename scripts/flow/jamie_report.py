#!/usr/bin/env python
"""CPU side of the toy study (FLOW_NOTES.md, "Jamie's toy exercises with OT
flow matching"): every metric, table and figure from the frozen-solver
outputs of jamie_eval.py --generate. Hidden truth is used here for scoring
only.

    jamie_report.py --exercise 1 --models 'E1-U=E1-U_s0,E1-P=E1-P_s0,...'
    (exercises 1, 2, 3, 3g, 4, 22 (the prior/data matrix); --out DIR)

Regions are by the distance of a tower centre to the generator's jet axis
(jamie_obs.py); energies are signed sums of the raw outputs; substructure
uses the image clipped at 0. Bootstrap: 200 resamples of events (of parents
for the banks).
"""

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

import fm_common as fc
import jamie_methods as jm
import jamie_obs as jo
import toycalo as tc

sys.path.insert(0, os.path.expanduser('~/pyext/jets'))
import fastjet._swig as fj            # pylint: disable=wrong-import-position,wrong-import-order

DEV = torch.device('cpu')
BOOT = 200
E_BINS = [ 10, 20, 25, 30, 40, 50, 80 ]
POP_OBS = [ 'total', 'cone', 'ring', 'far', 'mass', 'girth', 'ptd', 'zlead', 'zg', 'rg' ]
SHAPE_OBS = [ 'cone', 'girth', 'mass', 'ptd', 'zlead', 'ring' ]
BANK_OBS = [ 'cone', 'ring', 'girth' ]
# the second-seed confirmations: the same inputs' outputs of the two seeds
SEED_PAIRS = { '3' : [ ('D', 'D seed 1') ],
               '4' : [ ('D', 'D seed 1'), ('D+far', 'D seed 1+far') ] }

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Toy study report')
    parser.add_argument('--exercise', required = True,
                        choices = [ '1', '2', '3', '3g', '4', '22' ])
    parser.add_argument('--models', required = True,
                        help = 'comma separated LABEL=OUTPUT_DIR (outputs/<dir>)')
    parser.add_argument('--out', default = 'docs/flow/jamie_otfm')
    parser.add_argument('--procs', type = int, default = 30)
    return parser.parse_args()

# --- data

def cache(name):
    with np.load(os.path.join(jm.jamie_root(), 'cache', f'{name}.npz')) as f:
        return { k : f[k] for k in f.files }

def output(label, key):
    path = os.path.join(jm.jamie_root(), 'outputs', label, f'{key}.npy')
    return np.load(path).astype(np.float32) if os.path.exists(path) else None

def meta(label):
    path = os.path.join(jm.jamie_root(), 'outputs', label, 'meta.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding = 'utf-8') as f:
        return json.load(f)

def axes(d):
    return np.stack([ d['eta0'], d['phi0'] ], 1).astype(np.float32)

def dr_of(ax):
    return jo.offsets(ax, DEV)[2].numpy()

def obs(images, ax, sub = True):
    return jo.observables(images, ax, DEV, substructure = sub)

def specs(text):
    return [ tuple(s.split('=', 1)) for s in text.split(',') if s ]

def boot_sd(fn, n, seed = 1):
    rng = np.random.default_rng(seed)
    return float(np.std([ fn(rng.integers(0, n, n)) for _ in range(BOOT) ]))

def w1s(a, b, sd = None, boot = True, seed = 2):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    sd = sd or max(float(b.std()), 1e-9)
    w = wasserstein_distance(a, b) / sd
    if not boot:
        return (w, np.nan)
    rng = np.random.default_rng(seed)
    bs = [ wasserstein_distance(rng.choice(a, len(a)), rng.choice(b, len(b))) / sd
           for _ in range(BOOT) ]
    return (w, float(np.std(bs)))

def write(df, out, name):
    os.makedirs(out, exist_ok = True)
    df.to_csv(os.path.join(out, name), index = False)
    with pd.option_context('display.width', 250, 'display.max_columns', 80):
        print(f'== {name}\n' + df.round(4).to_string(index = False), flush = True)

# --- energy readouts

def rho_far(images, dr, rmin = 1.2):
    """The image's own UE level: mean tower energy at R > rmin (towers in
    the acceptance), signed."""
    far = dr > rmin
    return (images * far).sum((1, 2)) / far.sum((1, 2))

def region_sum(images, dr, lo, hi):
    return (images * ((dr >= lo) & (dr < hi))).sum((1, 2))

def region_area(dr, lo, hi):
    return ((dr >= lo) & (dr < hi)).sum((1, 2))

# --- anti-kT on positive towers

def akt_chunk(task):
    (imgs, axs) = task
    out = []
    centre_eta = tc.ETA_C
    centre_phi = tc.PHI_C
    for (img, ax) in zip(imgs, axs):
        parts = []
        for (r, c) in zip(*np.nonzero(img > 0)):
            pt = float(img[r, c])
            (eta, phi) = (centre_eta[r], centre_phi[c])
            parts.append(fj.PseudoJet(pt * math.cos(phi), pt * math.sin(phi),
                                      pt * math.sinh(eta), pt * math.cosh(eta)))
        if not parts:
            out.append((0.0, 0.0, 0))
            continue
        cs = fj.ClusterSequence(parts, fj.JetDefinition(fj.antikt_algorithm, 0.4))
        jets = fj.sorted_by_pt(cs.inclusive_jets(0.0))
        lead = jets[0].pt()
        near = 0.0
        if ax is not None and np.isfinite(ax[0]):
            for j in jets:
                dphi = math.atan2(math.sin(j.phi() - ax[1]), math.cos(j.phi() - ax[1]))
                if math.hypot(j.eta() - ax[0], dphi) < 0.4:
                    near = max(near, j.pt())
        out.append((lead, near, sum(1 for j in jets if j.pt() > 10.0 and abs(j.eta()) < 0.7)))
    return out

def antikt(images, ax, procs):
    """(leading jet pT, pT of the hardest jet within 0.4 of the axis, number
    of jets above 10 GeV at |eta| < 0.7), anti-kT R = 0.4 on the towers above
    0 (FastJet)."""
    n = len(images)
    if ax is None:
        ax = np.full((n, 2), np.nan, np.float32)
    step = max(1, n // (procs * 4))
    tasks = [ (images[s:s + step], ax[s:s + step]) for s in range(0, n, step) ]
    with Pool(procs) as p:
        res = sum(p.map(akt_chunk, tasks), [])
    return np.array(res, dtype = float)

# --- Exercise 1: subtraction and the halo

def resolution(e_hat, e_true):
    d = e_hat - e_true
    (q25, q75) = np.percentile(d, [ 25, 75 ])
    b = np.cov(e_true, e_hat)[0, 1] / np.var(e_true)
    a = e_hat.mean() - b * e_true.mean()
    return { 'offset' : float(d.mean()), 'rms' : float(d.std()), 'rms_iqr' : float((q75 - q25) / 1.349),
             'resp_slope' : float(b), 'rms_calibrated' : float((e_hat - a - b * e_true).std() / b) }

def ex1_scores(label, jhat, jet, ax, dr):
    # pylint: disable=too-many-locals
    cone = dr < 0.4
    e_t = (jet * cone).sum((1, 2))
    e_h = (jhat * cone).sum((1, 2))
    row = { 'model' : label, 'n' : len(jet), **resolution(e_h, e_t) }
    row['offset_sd'] = boot_sd(lambda i: (e_h[i] - e_t[i]).mean(), len(e_t))
    row['rms_sd'] = boot_sd(lambda i: (e_h[i] - e_t[i]).std(), len(e_t))
    row['neg_tower_share'] = float((jhat < 0).mean())
    row['neg_energy_event'] = float(jhat.clip(max = 0).sum((1, 2)).mean())
    row['neg_energy_cone'] = float((jhat.clip(max = 0) * cone).sum((1, 2)).mean())
    empty = jet == 0
    for (name, (lo, hi)) in jo.REGIONS.items():
        m = empty & (dr >= lo) & (dr < hi)
        row[f'halo_{name}_signed'] = float((jhat * m).sum((1, 2)).mean())
        row[f'halo_{name}_positive'] = float((jhat.clip(min = 0) * m).sum((1, 2)).mean())
    hard = jet >= 1.0
    soft = (jet > 0) & (jet < 1.0)
    row['core_loss_hard'] = float(((jhat - jet) * hard).sum((1, 2)).mean())
    row['core_loss_hard_rel'] = float(((jhat - jet) * hard).sum((1, 2)).sum() / (jet * hard).sum())
    row['soft_true_excess'] = float(((jhat - jet) * soft).sum((1, 2)).mean())
    row['occ_cone_hat'] = float(((jhat > 0.05) & cone).sum((1, 2)).mean())
    row['occ_cone_true'] = float(((jet > 0.05) & cone).sum((1, 2)).mean())
    lead_h = (jhat * cone).reshape(len(jet), -1).max(1)
    lead_t = (jet * cone).reshape(len(jet), -1).max(1)
    row['lead_error_mean'] = float((lead_h - lead_t).mean())
    row['lead_error_rms'] = float((lead_h - lead_t).std())
    for (lo, hi) in zip(jo.RINGS[:-1], jo.RINGS[1:]):
        m = (dr >= lo) & (dr < hi)
        tag = f'{lo:g}_{hi:g}'
        row[f'ring_{tag}_true'] = float((jet * m).sum((1, 2)).mean())
        row[f'ring_{tag}_on_true'] = float((jhat * m * (jet > 0)).sum((1, 2)).mean())
        row[f'ring_{tag}_on_empty'] = float((jhat * m * empty).sum((1, 2)).mean())
        row[f'ring_{tag}_sum'] = float((jhat * m).sum((1, 2)).mean())
    m = (dr >= 0.4) & (dr < 1.0)
    row['recoil_ring_true'] = float((jet * m).sum((1, 2)).mean())
    row['recoil_ring_hat'] = float((jhat * m).sum((1, 2)).mean())
    for (lo, hi) in zip(E_BINS[:-1], E_BINS[1:]):
        s = (e_t >= lo) & (e_t < hi)
        if s.sum() > 20:
            row[f'offset_{lo}_{hi}'] = float((e_h[s] - e_t[s]).mean())
            row[f'rms_{lo}_{hi}'] = float((e_h[s] - e_t[s]).std())
    return row

def rho_readouts(mix, dr):
    """The standard methods: rho from the image's R > 1.2; rho x A on the
    cone, and rho subtracted per tower (signed, and with negative towers
    dropped)."""
    rho = rho_far(mix, dr)
    signed = mix - rho[:, None, None]
    return { 'rho per tower (signed)' : signed,
             'rho per tower, negatives dropped' : signed.clip(min = 0) }

def ex1(models, cmdargs):
    # pylint: disable=too-many-locals,too-many-statements
    out = os.path.join(cmdargs.out, 'ex1')
    rows, slices, fakes, refound = [], [], [], []
    for (setname, main) in [ ('t1_mix', True), ('t1_low', False), ('t1_high', False),
                             ('t1_beyond', False) ]:
        d = cache(setname)
        (jet, ue) = (d['jet'], d['ue'])
        mix = jet + ue
        ax = axes(d)
        dr = dr_of(ax)
        cands = { lab : mix - output(dirn, f'{setname}_mix') for (lab, dirn) in models
                  if output(dirn, f'{setname}_mix') is not None }
        cands.update(rho_readouts(mix, dr))
        truth_akt = antikt(jet, ax, cmdargs.procs) if main else None
        for (lab, jhat) in cands.items():
            if main:
                rows.append(ex1_scores(lab, jhat, jet, ax, dr))
                if not lab.startswith('rho'):
                    (_, near_h, _) = antikt(jhat, ax, cmdargs.procs).T
                    refound.append({ 'model' : lab, **{ f'akt_{k}' : v for (k, v) in
                                     resolution(near_h, truth_akt[:, 1]).items() },
                                     'akt_found' : float(np.mean(near_h > 5)) })
            else:
                cone = dr < 0.4
                r = resolution((jhat * cone).sum((1, 2)), (jet * cone).sum((1, 2)))
                slices.append({ 'set' : setname, 'model' : lab, **r })
        # rho x A on the cone (not an image)
        rho = rho_far(mix, dr)
        e_ra = (mix * (dr < 0.4)).sum((1, 2)) - rho * (dr < 0.4).sum((1, 2))
        r = resolution(e_ra, (jet * (dr < 0.4)).sum((1, 2)))
        if main:
            rows.append({ 'model' : 'rho x A (cone)', 'n' : len(jet), **r })
        else:
            slices.append({ 'set' : setname, 'model' : 'rho x A (cone)', **r })
    write(pd.DataFrame(rows), out, 'ex1_subtraction.csv')
    write(pd.DataFrame(slices), out, 'ex1_slices.csv')
    write(pd.DataFrame(refound), out, 'ex1_refound.csv')
    # UE-only events: fake jets
    u = cache('t1_ue')['ue']
    cands = { lab : u - output(dirn, 't1_ue_ue') for (lab, dirn) in models
              if output(dirn, 't1_ue_ue') is not None }
    rho_u = u.mean((1, 2))
    a_cone = float(fc.ev.cone_kernel(0.4).sum())
    inner = np.abs(tc.ETA_C) < 0.7
    for (lab, jhat) in list(cands.items()) + [ ('rho x A (flat rho, event mean)', None) ]:
        if jhat is None:
            cs = jo.cone_sums_everywhere(u, DEV).numpy() - rho_u[:, None, None] * a_cone
            akt = antikt((u - rho_u[:, None, None]).clip(min = 0), None, cmdargs.procs)
        else:
            cs = jo.cone_sums_everywhere(jhat, DEV).numpy()
            akt = antikt(jhat.clip(min = 0), None, cmdargs.procs)
        big = cs[:, inner].max((1, 2))
        fakes.append({ 'model' : lab, 'n' : len(u),
                       'jamie_largest_cone_gt10' : float(np.mean(big > 10)),
                       'largest_cone_median' : float(np.median(big)),
                       'akt_jet_gt10_share' : float(np.mean(akt[:, 2] > 0)),
                       'akt_leading_median' : float(np.median(akt[:, 0])) })
    write(pd.DataFrame(fakes), out, 'ex1_fakes.csv')
    return rows

# --- Exercise 2 and the prior/data matrix: a frozen subtractor on pairs

def shape_obs(images, ax, dr):
    o = obs(images, ax)
    o['ring'] = region_sum(images, dr, 0.4, 1.0)
    return o

def ex2(models, cmdargs, tag = 'ex2'):
    # pylint: disable=too-many-locals,too-many-statements
    out = os.path.join(cmdargs.out, tag)
    d = cache('t2_pair')
    (jv, jmed, ue) = (d['jet'], d['med'], d['ue'])
    ax = axes(d)
    dr = dr_of(ax)
    (mv, mm) = (jv + ue, jmed + ue)
    tv = shape_obs(jv, ax, dr)
    tq = shape_obs(jmed, ax, dr)
    cands = {}
    for (lab, dirn) in models:
        (bv, bm) = (output(dirn, 't2_pair_vac'), output(dirn, 't2_pair_med'))
        if bv is not None and bm is not None:
            cands[lab] = (mv - bv, mm - bm, bv, bm)
    rv = rho_far(mv, dr)
    rm = rho_far(mm, dr)
    cands['rho per tower (signed)'] = (mv - rv[:, None, None], mm - rm[:, None, None], None, None)
    rows, errs = [], []
    n = len(jv)
    for (lab, (hv, hm, bv, bm)) in cands.items():
        ov = shape_obs(hv, ax, dr)
        oq = shape_obs(hm, ax, dr)
        row = { 'model' : lab, 'n' : n }
        for q in SHAPE_OBS:
            t_shift = tq[q] - tv[q]
            h_shift = oq[q] - ov[q]
            ok = np.isfinite(t_shift) & np.isfinite(h_shift)
            ts = float(t_shift[ok].mean())
            ts_se = float(t_shift[ok].std() / np.sqrt(ok.sum()))
            hs = float(h_shift[ok].mean())
            row[f'{q}_truth_shift'] = ts
            row[f'{q}_shift'] = hs
            row[f'{q}_shift_sd'] = boot_sd(lambda i, h = h_shift[ok]: h[i].mean(), int(ok.sum()))
            row[f'{q}_recovered'] = hs / ts if abs(ts) > 5 * ts_se else np.nan
            row[f'{q}_abs_error'] = hs - ts
            for (ver, o, t) in [ ('vac', ov, tv), ('med', oq, tq) ]:
                e = o[q] - t[q]
                e = e[np.isfinite(e)]
                errs.append({ 'model' : lab, 'version' : ver, 'observable' : q,
                              'error_mean' : float(e.mean()), 'error_sd' : float(e.std()),
                              'error_p16' : float(np.percentile(e, 16)),
                              'error_p84' : float(np.percentile(e, 84)) })
        row['lost_vac'] = float(np.mean((ov['cone'] < 10) & (tv['cone'] >= 10)))
        row['lost_med'] = float(np.mean((oq['cone'] < 10) & (tq['cone'] >= 10)))
        hard_v = jv >= 1.0
        hard_m = jmed >= 1.0
        row['hard_loss_vac'] = float(((hv - jv) * hard_v).sum((1, 2)).mean())
        row['hard_loss_med'] = float(((hm - jmed) * hard_m).sum((1, 2)).mean())
        row['hard_loss_rel_vac'] = float(((hv - jv) * hard_v).sum() / (jv * hard_v).sum())
        row['hard_loss_rel_med'] = float(((hm - jmed) * hard_m).sum() / (jmed * hard_m).sum())
        row['vac_cone_offset'] = float((ov['cone'] - tv['cone']).mean())
        row['vac_cone_rms'] = float((ov['cone'] - tv['cone']).std())
        row['vac_ring_hat'] = float(ov['ring'].mean())
        row['vac_ring_true'] = float(tv['ring'].mean())
        row['vac_ring_excess_sd'] = boot_sd(lambda i: (ov['ring'][i] - tv['ring'][i]).mean(), n)
        if bv is not None:
            ring = (dr >= 0.4) & (dr < 1.0)
            row['recoil_in_bhat'] = float((((bm - ue) - (bv - ue)) * ring).sum((1, 2)).mean())
        # inclusive (no selection) first; survivors (both versions >= 10 GeV) beside it
        surv = (ov['cone'] >= 10) & (oq['cone'] >= 10)
        row['survivors'] = float(surv.mean())
        row['cone_shift_survivors'] = float((oq['cone'] - ov['cone'])[surv].mean())
        row['cone_truth_shift_survivors'] = float((tq['cone'] - tv['cone'])[surv].mean())
        rows.append(row)
    write(pd.DataFrame(rows), out, f'{tag}_shifts.csv')
    write(pd.DataFrame(errs), out, f'{tag}_errors.csv')
    if tag == 'prior_data':
        prior_data_effects(rows, out)
    return rows

def prior_data_effects(rows, out):
    """The 2 x 2's pre-registered readings: per prior, quenched-data minus
    vacuum-data cell in the recovered cone, girth and ring shifts, in
    combined bootstrap sd ('data matter' beyond 3); each cell's ring in
    vacuum jets against the truth's (beyond 3 sd: recoil invented), and,
    since the vacuum-prior cells already leave a ring halo, the broad minus
    the vacuum prior at the same data."""
    by = { r['model'] : r for r in rows }
    eff = []
    for prior in ('vac', 'broad'):
        (q, v) = (by.get(f'{prior} prior x quench data'), by.get(f'{prior} prior x vac data'))
        syn = by.get(f'{prior} prior: syn')
        if q is None or v is None:
            continue
        row = { 'prior' : prior }
        for k in ('cone', 'girth', 'ring'):
            d = q[f'{k}_shift'] - v[f'{k}_shift']
            row[f'{k}_quench_minus_vac_data'] = d
            row[f'{k}_z'] = d / np.hypot(q[f'{k}_shift_sd'], v[f'{k}_shift_sd'])
            if syn is not None:
                row[f'{k}_quench_data_minus_syn'] = q[f'{k}_shift'] - syn[f'{k}_shift']
                row[f'{k}_vac_data_minus_syn'] = v[f'{k}_shift'] - syn[f'{k}_shift']
        row['data_matter'] = bool(max(abs(row[f'{k}_z']) for k in ('cone', 'girth', 'ring')) > 3)
        for (lab, r) in (('vac_data', v), ('quench_data', q)):
            ex = r['vac_ring_hat'] - r['vac_ring_true']
            row[f'vac_ring_excess_{lab}'] = ex
            row[f'invents_recoil_{lab}'] = bool(ex > 3 * r['vac_ring_excess_sd'])
        eff.append(row)
    for data in ('vac', 'quench'):
        (b, v) = (by.get(f'broad prior x {data} data'), by.get(f'vac prior x {data} data'))
        if b is not None and v is not None:
            eff.append({ 'prior' : f'broad minus vac, {data} data',
                         'vac_ring_broad_minus_vac' : b['vac_ring_hat'] - v['vac_ring_hat'],
                         'vac_ring_broad_minus_vac_z' : (b['vac_ring_hat'] - v['vac_ring_hat'])
                         / np.hypot(b['vac_ring_excess_sd'], v['vac_ring_excess_sd']),
                         **{ f'{k}_broad_minus_vac' : b[f'{k}_shift'] - v[f'{k}_shift']
                             for k in ('cone', 'girth', 'ring') } })
    if eff:
        write(pd.DataFrame(eff), out, 'prior_data_effects.csv')

# --- Exercise 3 (clean) and 4 (with UE): translation

def patches(images, ax):
    """(N, 9, 9) towers around the axis tower (phi wraps)."""
    r0 = np.clip(np.floor((ax[:, 0] - tc.ETA_MIN) / tc.DETA).astype(int), 4, 19)
    c0 = np.floor(np.mod(ax[:, 1], 2 * np.pi) / tc.DPHI).astype(int)
    off = np.arange(-4, 5)
    rows = r0[:, None, None] + off[None, :, None]
    cols = (c0[:, None, None] + off[None, None, :]) % tc.NPHI
    k = np.arange(len(images))[:, None, None]
    return images[k, rows, cols]

_EMD = {}

def emd_chunk(task):
    (key, s, e) = task
    # pylint: disable=import-outside-toplevel
    from jet_fidelity import EMD
    from substructure import Geometry
    torch.set_num_threads(1)
    (a, b) = _EMD[key]
    geo = Geometry(torch.device('cpu'))
    emd = EMD(geo)
    def unit(x):
        w = torch.as_tensor(x[s:e]).float().clamp(min = 0) * geo.mask
        return w / w.sum((1, 2), keepdim = True).clamp(min = 1e-12)
    return emd(unit(a), unit(b))[0]

def shape_emds(pairs, procs):
    global _EMD            # pylint: disable=global-statement
    _EMD = pairs
    out = {}
    with Pool(procs) as p:
        for (key, (a, _)) in pairs.items():
            step = max(1, len(a) // (procs * 4))
            out[key] = np.concatenate(p.map(emd_chunk, [ (key, s, min(s + step, len(a)))
                                                        for s in range(0, len(a), step) ]))
    return out

def ue_readout(images, dr):
    """Cone and ring energy minus the image's own rho (R > 1.2) times the
    regions' tower counts; signed."""
    rho = rho_far(images, dr)
    return { 'cone_sub' : region_sum(images, dr, 0.0, 0.4) - rho * region_area(dr, 0.0, 0.4),
             'ring_sub' : region_sum(images, dr, 0.4, 1.0) - rho * region_area(dr, 0.4, 1.0),
             'rho' : rho }

def translation(models, cmdargs, ex):
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    girth = ex == '3g'
    ue_case = ex == '4'
    tag = f'ex{ex}'
    out = os.path.join(cmdargs.out, tag)
    setname = { '3' : 't3_pair', '3g' : 't3g_pair', '4' : 't4_pair' }[ex]
    refname = { '3' : 't3_ref', '3g' : 't3g_ref', '4' : 't4_ref' }[ex]
    bankname = { '3' : 't3_bank', '3g' : 't3g_bank', '4' : 't4_bank' }[ex]
    d = cache(setname)
    ax = axes(d)
    dr = dr_of(ax)
    if ue_case:
        (inp, truth, ue_a) = (d['jet'] + d['ue'], d['med'] + d['ue'], d['ue'])
    else:
        (inp, truth) = (d['jet'], d['med'])
    r = cache(refname)
    ref = r['med'] + r['ue'] if ue_case else r['med']
    ax_ref = axes(r)
    dr_ref = dr_of(ax_ref)
    def full_obs(images, a, drr):
        o = obs(images, a)
        if ue_case:
            o.update(ue_readout(images, drr))
        return o
    o_in = full_obs(inp, ax, dr)
    o_tr = full_obs(truth, ax, dr)
    o_ref = full_obs(ref, ax_ref, dr_ref)
    cands = { lab : output(dirn, f'{setname}_vac') for (lab, dirn) in models }
    cands = { k : v for (k, v) in cands.items() if v is not None }
    o_mod = { lab : full_obs(x, ax, dr) for (lab, x) in cands.items() }
    keys = POP_OBS + ([ 'cone_sub', 'ring_sub' ] if ue_case else [])
    pop, dep = [], []
    rows_src = [ ('truth reference (independent sample)', o_ref), ('identity (input)', o_in) ] \
        + list(o_mod.items())
    for (lab, o) in rows_src:
        row = { 'model' : lab }
        for q in keys:
            (w, e) = w1s(o[q], o_tr[q])
            row[f'w1_{q}'] = w
            row[f'w1_{q}_sd'] = e
            row[f'mean_{q}'] = float(np.nanmean(o[q]))
        row['valid_cone_ge_10'] = float(np.mean(o['cone'] >= 10))
        pop.append(row)
    shift_keys = [ 'cone', 'ring', 'girth' ] + ([ 'cone_sub', 'ring_sub' ] if ue_case else [])
    for (lab, o) in [ ('identity (input)', o_in) ] + list(o_mod.items()):
        row = { 'model' : lab }
        for q in shift_keys:
            ts = float(np.nanmean(o_tr[q]) - np.nanmean(o_in[q]))
            row[f'{q}_shift_recovered'] = (float(np.nanmean(o[q]) - np.nanmean(o_in[q])) / ts
                                           if abs(ts) > 1e-9 else np.nan)
            ok = np.isfinite(o[q]) & np.isfinite(o_in[q])
            row[f'corr_{q}_input'] = float(np.corrcoef(o[q][ok], o_in[q][ok])[0, 1])
            okt = ok & np.isfinite(o_tr[q])
            row[f'corr_{q}_truth_draw'] = float(np.corrcoef(o[q][okt], o_tr[q][okt])[0, 1])
        # the ring's energy by tower size: recoil-like towers against a haze
        x = cands.get(lab, inp)
        ring = (dr >= 0.4) & (dr < 1.0)
        for (lo, hi) in [ (-1e9, 0.05), (0.05, 0.5), (0.5, 1e9) ]:
            m = ring & (x >= lo) & (x < hi)
            mt = ring & (truth >= lo) & (truth < hi)
            row[f'ring_towers_{lo:g}_{hi:g}_energy'] = float((x * m).sum((1, 2)).mean())
            row[f'ring_towers_{lo:g}_{hi:g}_energy_truth'] = float((truth * mt).sum((1, 2)).mean())
        if ue_case:
            far = dr >= 1.0
            xf = x[far]
            inf = inp[far]
            row['far_tower_corr_input'] = float(np.corrcoef(xf, inf)[0, 1])
            row['far_rms_change_over_ue_sd'] = float(np.sqrt(np.mean((xf - inf)**2)) / inf.std())
            row['far_mean_change'] = float((xf - inf).mean())
            tf = truth[far]
            row['truth_far_tower_corr_input'] = float(np.corrcoef(tf, inf)[0, 1])
            row['truth_far_rms_change_over_ue_sd'] = float(np.sqrt(np.mean((tf - inf)**2)) / inf.std())
            other = np.roll(inp, 1, axis = 0)[far]
            row['other_event_far_corr'] = float(np.corrcoef(xf, other)[0, 1])
            if lab in cands:
                orc = cands[lab] - ue_a
                row['oracle_cone'] = float(region_sum(orc, dr, 0, 0.4).mean())
                row['oracle_ring'] = float(region_sum(orc, dr, 0.4, 1.0).mean())
        dep.append(row)
    trc = { 'oracle_cone' : float(region_sum(d['med'], dr, 0, 0.4).mean()),
            'oracle_ring' : float(region_sum(d['med'], dr, 0.4, 1.0).mean()) } if ue_case else {}
    write(pd.DataFrame(pop), out, f'{tag}_population.csv')
    dep = pd.DataFrame(dep)
    for (k, v) in trc.items():
        dep[f'truth_{k}'] = v
    write(dep, out, f'{tag}_dependence.csv')
    # own-input shape EMD against another input (9 x 9 cone patches)
    shift = 1 + int(np.random.default_rng(2).integers(len(inp) - 1))
    perm = (np.arange(len(inp)) + shift) % len(inp)
    pin = patches(inp - (ue_a if ue_case else 0), ax)
    pairs = {}
    for (lab, x) in [ ('identity (input)', inp), ('truth (one draw)', truth) ] + list(cands.items()):
        px = patches(x - (ue_a if ue_case else 0), ax)
        pairs[(lab, 'own')] = (px, pin)
        pairs[(lab, 'other')] = (px, pin[perm])
    em = shape_emds(pairs, cmdargs.procs)
    emd_rows = []
    for lab in dict.fromkeys(k[0] for k in pairs):
        (own, oth) = (em[(lab, 'own')], em[(lab, 'other')])
        ok = np.isfinite(own) & np.isfinite(oth)
        emd_rows.append({ 'model' : lab, 'emd_own_input' : float(own[ok].mean()),
                          'emd_other_input' : float(oth[ok].mean()),
                          'own_preferred' : float(np.mean(own[ok] < oth[ok])) })
    write(pd.DataFrame(emd_rows), out, f'{tag}_emd.csv')
    seeds(ex, o_in, o_tr, o_mod, cands, inp, dr, out)
    bank(models, cmdargs, ex, d, ax, bankname, out)
    swaps(models, cmdargs, ex, d, ax, out)
    if girth:
        girth_rule(o_in, o_tr, o_mod, d, out)
    return pop

def seeds(ex, o_in, o_tr, o_mod, cands, inp, dr, out):
    """Second seed: per input, the two seeds' outputs against each other, as
    changes from the input (observables) and tower by tower by region; and
    each seed's change against the true change of that jet (one draw)."""
    # pylint: disable=too-many-locals
    rows = []
    keys = [ 'cone', 'ring', 'far', 'girth' ] + ([ 'cone_sub', 'ring_sub' ] if ex == '4' else [])
    for (a, b) in SEED_PAIRS.get(ex, []):
        if a not in o_mod or b not in o_mod:
            continue
        row = { 'pair' : f'{a} vs {b}' }
        for q in keys:
            (ca, cb) = (o_mod[a][q] - o_in[q], o_mod[b][q] - o_in[q])
            ct = o_tr[q] - o_in[q]
            ok = np.isfinite(ca) & np.isfinite(cb) & np.isfinite(ct)
            (ca, cb, ct) = (ca[ok], cb[ok], ct[ok])
            row[f'{q}_corr_change_truth_a'] = float(np.corrcoef(ca, ct)[0, 1])
            row[f'{q}_corr_change_truth_b'] = float(np.corrcoef(cb, ct)[0, 1])
            row[f'{q}_mean_change_a'] = float(ca.mean())
            row[f'{q}_mean_change_b'] = float(cb.mean())
            row[f'{q}_rms_change_a'] = float(np.sqrt(np.mean(ca**2)))
            row[f'{q}_rms_seed_diff'] = float(np.sqrt(np.mean((ca - cb)**2)))
            row[f'{q}_corr_changes'] = float(np.corrcoef(ca, cb)[0, 1])
        for (name, lo, hi) in [ ('cone', 0.0, 0.4), ('ring', 0.4, 1.0), ('far', 1.0, 1e9) ]:
            m = (dr >= lo) & (dr < hi)
            row[f'towers_{name}_rms_seed_diff'] = float(np.sqrt(((cands[a] - cands[b])**2)[m].mean()))
            row[f'towers_{name}_rms_change_a'] = float(np.sqrt(((cands[a] - inp)**2)[m].mean()))
        rows.append(row)
    if rows:
        write(pd.DataFrame(rows), out, f'ex{ex}_seeds.csv')

def draw_spread(x, dr):
    """(P, n, 24, 64) draws: the mean over parents of the draw-to-draw sd of
    the near (R < 1) and far (R >= 1) energy sums and of single towers there."""
    out = {}
    for (name, m) in [ ('near', dr < 1.0), ('far', dr >= 1.0) ]:
        mm = m[:, None]
        out[f'{name}_sum_draw_sd'] = float(np.mean(np.std((x * mm).sum((2, 3)), axis = 1)))
        sd = np.std(x, axis = 1)
        out[f'{name}_tower_draw_sd'] = float((sd * m).sum() / m.sum())
    return out

def bank(models, cmdargs, ex, d, ax, bankname, out):
    """Conditional distributions: 128 model draws per parent against 128 true
    quenchings (the other 128: the true-vs-true reference)."""
    # pylint: disable=too-many-locals,unused-argument
    ue_case = ex == '4'
    b = cache(bankname)
    parents = b['parents']
    a = ax[parents]
    reps = b['reps'].astype(np.float32)            # (P, 256, 24, 64)
    if ue_case:
        reps = reps + d['ue'][parents][:, None]
    (p, k) = reps.shape[:2]
    def per_draw(x):
        n = x.shape[1]
        flat = x.reshape(-1, 24, 64)
        aa = np.repeat(a, n, axis = 0)
        drr = dr_of(aa)
        o = obs(flat, aa, sub = False)
        if ue_case:
            o.update(ue_readout(flat, drr))
            o['cone'] = o['cone_sub']
            o['ring'] = o['ring_sub']
        return { q : np.asarray(o[q]).reshape(p, n) for q in BANK_OBS }
    tr = per_draw(reps)
    half = k // 2
    tr_a = { q : v[:, :half] for (q, v) in tr.items() }
    tr_b = { q : v[:, half:] for (q, v) in tr.items() }
    sd_pop = { q : float(np.nanstd(tr[q])) for q in BANK_OBS }
    rows = []
    def score(lab, m):
        row = { 'model' : lab, 'parents' : p, 'draws' : int(next(iter(m.values())).shape[1]) }
        for q in BANK_OBS:
            cw = np.array([ wasserstein_distance(m[q][i], tr_a[q][i]) for i in range(p) ]) / sd_pop[q]
            rw = np.array([ wasserstein_distance(tr_b[q][i], tr_a[q][i]) for i in range(p) ]) / sd_pop[q]
            row[f'{q}_cond_w1'] = float(cw.mean())
            row[f'{q}_cond_w1_sd'] = boot_sd(lambda i, c = cw: c[i].mean(), p)
            row[f'{q}_ref_w1'] = float(rw.mean())
            lo68 = np.percentile(m[q], 16, axis = 1)[:, None]
            hi68 = np.percentile(m[q], 84, axis = 1)[:, None]
            lo90 = np.percentile(m[q], 5, axis = 1)[:, None]
            hi90 = np.percentile(m[q], 95, axis = 1)[:, None]
            row[f'{q}_cov68'] = float(np.mean((tr_b[q] >= lo68) & (tr_b[q] <= hi68)))
            row[f'{q}_cov90'] = float(np.mean((tr_b[q] >= lo90) & (tr_b[q] <= hi90)))
            row[f'{q}_cond_sd_model'] = float(np.mean(np.std(m[q], axis = 1)))
            row[f'{q}_cond_sd_truth'] = float(np.mean(np.std(tr_a[q], axis = 1)))
            row[f'{q}_cond_mean_bias'] = float(np.mean(m[q].mean(1) - tr_a[q].mean(1)))
            row[f'{q}_cond_mean_rms'] = float(np.sqrt(np.mean((m[q].mean(1) - tr_a[q].mean(1))**2)))
            row[f'{q}_corr_cond_means'] = float(np.corrcoef(m[q].mean(1), tr_a[q].mean(1))[0, 1])
        if m['cone'].shape[1] > 2:
            cc = [ np.corrcoef(m['cone'][i], m['ring'][i])[0, 1] for i in range(p) ]
            ct = [ np.corrcoef(tr_a['cone'][i], tr_a['ring'][i])[0, 1] for i in range(p) ]
            row['cone_ring_corr_model'] = float(np.nanmean(cc))
            row['cone_ring_corr_truth'] = float(np.nanmean(ct))
        return row
    dr_p = dr_of(a)
    row = score('truth (true-vs-true reference)', tr_b)
    # the best per-jet correlation any deterministic predictor can reach with
    # one true quenching: the parent's mean change against single draws'
    # changes (all 256 quenchings of each bank parent)
    inp_p = (d['jet'] + d['ue'])[parents] if ue_case else d['jet'][parents]
    o_in = per_draw(inp_p[:, None].astype(np.float32))
    for q in BANK_OBS:
        dlt = tr[q] - o_in[q]
        mu = np.repeat(np.nanmean(dlt, axis = 1, keepdims = True), k, axis = 1)
        ok = np.isfinite(dlt) & np.isfinite(mu)
        row[f'ceiling_corr_change_{q}'] = float(np.corrcoef(mu[ok], dlt[ok])[0, 1])
    if ue_case:
        # where the draws differ: the jet's neighbourhood or the far UE
        row.update(draw_spread(reps[:, half:], dr_p))
    rows.append(row)
    for (lab, dirn) in models:
        x = output(dirn, f'{bankname}_bank')
        if x is None:
            continue
        x = x.astype(np.float32)
        row = score(lab, per_draw(x))
        if ue_case and x.shape[1] > 1:
            row.update(draw_spread(x, dr_p))
        rows.append(row)
    write(pd.DataFrame(rows), out, f'ex{ex}_conditional.csv')

def swaps(models, cmdargs, ex, d, ax, out):
    """The condition swap at fixed noise (C models)."""
    # pylint: disable=too-many-locals,unused-argument
    rows = []
    n = 1000
    ue_case = ex == '4'
    inp = (d['jet'] + d['ue'])[:n] if ue_case else d['jet'][:n]
    a = ax[:n]
    drr = dr_of(a)
    for (lab, dirn) in models:
        path = os.path.join(jm.jamie_root(), 'outputs', dirn, 'swap.npz')
        if not os.path.exists(path):
            continue
        with np.load(path) as f:
            s = { k : f[k] for k in f.files }
        perm = s['perm']
        def readout(x, aa, dd):
            o = obs(x, aa, sub = False)
            if ue_case:
                o.update(ue_readout(x, dd))
                o['cone'] = o['cone_sub']
                o['ring'] = o['ring_sub']
            return o
        o_in = readout(inp, a, drr)
        o_own = readout(s['own'], a, drr)
        # the swapped output sits around its condition's axis
        o_sw = readout(s['swap'], a[perm], drr[perm])
        o_oth = readout(s['other'], a, drr)
        row = { 'model' : lab }
        for q in [ 'cone', 'ring', 'girth' ]:
            row[f'corr_{q}_swap_condition'] = float(np.corrcoef(o_sw[q], o_in[q][perm])[0, 1])
            row[f'corr_{q}_swap_noise_input'] = float(np.corrcoef(o_sw[q], o_in[q])[0, 1])
            f_sw = o_sw[q] - o_oth[q][perm]
            f_own = o_own[q] - o_oth[q]
            row[f'fluct_corr_{q}_same_noise'] = float(np.corrcoef(f_sw, f_own)[0, 1])
        pin = patches(inp, a)
        pairs = { 'cond' : (patches(s['swap'], a[perm]), pin[perm]),
                  'noise' : (patches(s['swap'], a[perm]), pin) }
        em = shape_emds(pairs, cmdargs.procs)
        row['swap_closer_to_condition'] = float(np.mean(em['cond'] < em['noise']))
        rows.append(row)
    if rows:
        write(pd.DataFrame(rows), out, f'ex{ex}_swap.csv')

def girth_rule(o_in, o_tr, o_mod, d, out):
    """Does the modification follow the input girth, or only its energy?"""
    rows = []
    g = d['g']
    for (lab, o) in [ ('truth (one draw)', o_tr) ] + list(o_mod.items()):
        loss = o_in['cone'] - o['cone']
        frac = loss / np.maximum(o_in['cone'], 1e-3)
        ok = np.isfinite(frac)
        x = np.stack([ np.ones(ok.sum()), (o_in['girth'][ok] - o_in['girth'][ok].mean()) / o_in['girth'][ok].std(),
                       (o_in['cone'][ok] - o_in['cone'][ok].mean()) / o_in['cone'][ok].std() ], 1)
        coef = np.linalg.lstsq(x, frac[ok], rcond = None)[0]
        rows.append({ 'model' : lab,
                      'corr_loss_frac_input_girth' : float(np.corrcoef(frac[ok], o_in['girth'][ok])[0, 1]),
                      'corr_loss_frac_particle_g' : float(np.corrcoef(frac[ok], g[ok])[0, 1]),
                      'corr_loss_frac_input_energy' : float(np.corrcoef(frac[ok], o_in['cone'][ok])[0, 1]),
                      'coef_girth' : float(coef[1]), 'coef_energy' : float(coef[2]),
                      'corr_loss_with_true_loss' : float(np.corrcoef(
                          frac[ok], ((o_in['cone'] - o_tr['cone']) / np.maximum(o_in['cone'], 1e-3))[ok])[0, 1]) })
    write(pd.DataFrame(rows), out, 'ex3g_girth_rule.csv')

def main():
    cmdargs = parse_cmdargs()
    torch.set_num_threads(16)
    models = specs(cmdargs.models)
    if cmdargs.exercise == '1':
        ex1(models, cmdargs)
    elif cmdargs.exercise == '2':
        ex2(models, cmdargs)
    elif cmdargs.exercise == '22':
        ex2(models, cmdargs, tag = 'prior_data')
    else:
        translation(models, cmdargs, cmdargs.exercise)

if __name__ == '__main__':
    main()
