#!/usr/bin/env python
"""Pools, held-out sets, checks and normalisation of the toy study
(FLOW_NOTES.md, "Jamie's toy exercises with OT flow matching"), from the
reimplemented toy calorimeter (toycalo.py).

    jamie_data.py --build [--procs 30]     every pool and set, the manifest
    jamie_data.py --checks                 generator checks against the deck

Parents. A parent is one generator draw: a vacuum jet (particles, axis,
energy) and/or an underlying event, with its own random stream (toycalo
rng_of(pool, index, role); role 0 jet, 1 UE, 2 + k quench replica k). Every
pool and set has its own pool id, so no parent is in two of them: training
pools are pairwise disjoint (the source and target pools of an unpaired
problem are different parents), and validation and test parents are in no
training pool, synthetic sums included. A mixture J + B is one parent with
both a jet and a UE stream. Repeated quenchings of a test parent (the banks)
reuse its role-0 particles, so all its replicas stay in its split.

Training pools (float16 images in OUTDIR/sphenix/flow/jamie/cache/NAME.npy,
metadata NAME_meta.npz: pool, parent, eta0, phi0, E, E_img, lost, f, g):
    e1_jet       vacuum jets alone, |eta| < 0.7, 20-80 GeV, E^-4 (the jet prior)
    e1_jet_flat  the same, flat 5-80 GeV (the training-support continuation)
    e1_ue        UE alone (the UE pool: OT targets and synthetic sums)
    e1_mix       vacuum jet + UE of other parents (Exercise 1's mixture data)
    e3_vac       vacuum jets, |eta| < 0.3 (Exercise 3's source; the 2 x 2's vacuum prior)
    e3_med       quenched jets of other parents, f ~ Beta(2, 6) (Exercise 3's target)
    e3_pair      40k further parents, the vacuum jet and 8 independent quenchings
                 (the paired positive control only)
    e3g_med      quenched jets of other parents, f from the particle girth
    e4_src       vacuum jet + UE (Exercise 4's source; the 2 x 2's vacuum data)
    e4_tgt       quenched jet + another UE (Exercise 4's target; the 2 x 2's quenched data)
    x_broad      the broad prior: half vacuum, half quenched with f ~ U(0, 0.5)

Held-out sets (float32, NAME.npz: components, axes, energies, f, g):
    v1_mix, t1_mix          Exercise 1: J (|eta| < 0.7, 20-80 GeV) and B (5k / 20k)
    t1_low, t1_high, t1_beyond   flat 5-20, 60-80, 80-100 GeV jets in UE (5k each)
    t1_ue                   UE alone (fake jets, 5k)
    v2_pair, t2_pair        J_vac, J_med (Beta) and one B shared by both (5k / 20k)
    v3_pair, t3_pair        J_vac and one J_med (5k / 20k); t3_ref an independent
                            quenched sample (20k); t3_bank 256 quenchings of 100 test
                            parents, 512 of 4 illustration parents
    v3g_pair, t3g_pair, t3g_ref, t3g_bank   the same with f from the girth
    v4_pair, t4_pair        J_vac + B_a and J_med + B_a (5k / 20k); t4_ref
                            quenched jet + another UE (20k); t4_bank as t3_bank
                            with each parent's B_a fixed
Bank parents: among the first 2000 test parents, the nearest to the energy
quantiles 0.005, 0.015, ..., 0.995 (100) and 0.1, 0.4, 0.7, 0.95 (4), fixed
before any model output.

Normalisation (norm.json): psi = log(E + 0.1), one mean and sd per state
kind fitted on its training pools together (20k images each): sub (e1_mix,
e1_ue; Exercises 1, 2 and the 2 x 2), clean (e3_vac, e3_med; Exercise 3 and
its girth-rule variant), ue (e4_src, e4_tgt; Exercise 4).
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

import fm_common as fc
import toycalo as tc

N_TRAIN, N_VAL, N_TEST = 40000, 5000, 20000
BANK_N, BANK_REPS, ILLU_REPS = 100, 256, 512

# name: (pool id, kind, n, options)
POOLS = {
    'e1_jet'      : (101, 'jet', N_TRAIN, dict(eta_max = 0.7)),
    'e1_jet_flat' : (102, 'jet', N_TRAIN, dict(eta_max = 0.7, emin = 5.0, emax = 80.0, power = 0.0)),
    'e1_ue'       : (103, 'ue', N_TRAIN, {}),
    'e1_mix'      : (104, 'mix', N_TRAIN, dict(eta_max = 0.7)),
    'e3_vac'      : (301, 'jet', N_TRAIN, dict(eta_max = 0.3)),
    'e3_med'      : (302, 'med', N_TRAIN, dict(eta_max = 0.3, mode = 'beta')),
    'e3_pair'     : (303, 'reps', N_TRAIN, dict(eta_max = 0.3, mode = 'beta', reps = 8)),
    'e3g_med'     : (312, 'med', N_TRAIN, dict(eta_max = 0.3, mode = 'girth')),
    'e4_src'      : (401, 'mix', N_TRAIN, dict(eta_max = 0.3)),
    'e4_tgt'      : (402, 'medmix', N_TRAIN, dict(eta_max = 0.3, mode = 'beta')),
    'x_broad'     : (501, 'broad', N_TRAIN, dict(eta_max = 0.3)),
}
SETS = {
    'v1_mix'    : (151, 'split', N_VAL, dict(eta_max = 0.7)),
    't1_mix'    : (152, 'split', N_TEST, dict(eta_max = 0.7)),
    't1_low'    : (153, 'split', 5000, dict(eta_max = 0.7, emin = 5.0, emax = 20.0, power = 0.0)),
    't1_high'   : (154, 'split', 5000, dict(eta_max = 0.7, emin = 60.0, emax = 80.0, power = 0.0)),
    't1_beyond' : (155, 'split', 5000, dict(eta_max = 0.7, emin = 80.0, emax = 100.0, power = 0.0)),
    't1_ue'     : (156, 'ue', 5000, {}),
    'v2_pair'   : (251, 'pairue', N_VAL, dict(eta_max = 0.3, mode = 'beta')),
    't2_pair'   : (252, 'pairue', N_TEST, dict(eta_max = 0.3, mode = 'beta')),
    'v3_pair'   : (351, 'pair', N_VAL, dict(eta_max = 0.3, mode = 'beta')),
    't3_pair'   : (352, 'pair', N_TEST, dict(eta_max = 0.3, mode = 'beta')),
    't3_ref'    : (353, 'med', N_TEST, dict(eta_max = 0.3, mode = 'beta')),
    'v3g_pair'  : (361, 'pair', N_VAL, dict(eta_max = 0.3, mode = 'girth')),
    't3g_pair'  : (362, 'pair', N_TEST, dict(eta_max = 0.3, mode = 'girth')),
    't3g_ref'   : (363, 'med', N_TEST, dict(eta_max = 0.3, mode = 'girth')),
    'v4_pair'   : (451, 'pairue', N_VAL, dict(eta_max = 0.3, mode = 'beta')),
    't4_pair'   : (452, 'pairue', N_TEST, dict(eta_max = 0.3, mode = 'beta')),
    't4_ref'    : (453, 'medmix', N_TEST, dict(eta_max = 0.3, mode = 'beta')),
}
# repeated quenchings of selected test parents: (bank, its test set)
BANKS = { 't3_bank' : 't3_pair', 't3g_bank' : 't3g_pair', 't4_bank' : 't4_pair' }
NORMS = { 'sub' : ('e1_mix', 'e1_ue'), 'clean' : ('e3_vac', 'e3_med'),
          'ue' : ('e4_src', 'e4_tgt') }

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Toy study: pools and sets')
    parser.add_argument('--build', action = 'store_true')
    parser.add_argument('--checks', action = 'store_true')
    parser.add_argument('--only', default = None, help = 'comma separated names')
    parser.add_argument('--procs', type = int, default = 30)
    parser.add_argument('--chunk', type = int, default = 500)
    return parser.parse_args()

def root():
    return os.path.join(fc.out_root(), 'jamie')

def cache_dir():
    return os.path.join(root(), 'cache')

# --- one parent

def vacuum(pool, i, opts):
    rng = tc.rng_of(pool, i, 0)
    return tc.VacuumJet(rng, opts['eta_max'], opts.get('emin', 20.0), opts.get('emax', 80.0),
                        opts.get('power', 4.0))

def ue(pool, i, info = None):
    return tc.background_image(tc.rng_of(pool, i, 1), info)

def parent(task):
    """Images and metadata of parent i of a pool or set."""
    # pylint: disable=too-many-locals,too-many-branches
    (pool, kind, opts, i) = task
    meta = { 'pool' : pool, 'parent' : i, 'eta0' : np.nan, 'phi0' : np.nan, 'E' : np.nan,
             'E_img' : np.nan, 'lost' : 0.0, 'f' : np.nan, 'g' : np.nan, 'E_med' : np.nan,
             'lost_med' : 0.0, 'm' : np.nan, 'v2' : np.nan, 'psi' : np.nan, 'minijets' : -1 }
    imgs = {}
    if kind != 'ue':
        jet = vacuum(pool, i, opts)
        (img, lost) = jet.image()
        meta.update(eta0 = jet.eta0, phi0 = jet.phi0, E = jet.E, E_img = img.sum(),
                    lost = lost, g = jet.g)
        if kind in ('jet', 'split', 'mix', 'pair', 'pairue', 'reps'):
            imgs['jet'] = img
        if kind == 'broad':
            # half the parents vacuum, half quenched with f ~ U(0, 0.5)
            if tc.rng_of(pool, i, 2).random() < 0.5:
                imgs['jet'] = img
                meta['f'] = 0.0
            else:
                (q, f, lq) = jet.quench(tc.rng_of(pool, i, 3), 'uniform')
                imgs['jet'] = q
                meta.update(f = f, E_med = q.sum(), lost_med = lq)
        if kind in ('med', 'medmix', 'pair', 'pairue'):
            (q, f, lq) = jet.quench(tc.rng_of(pool, i, 2), opts['mode'])
            imgs['med'] = q
            meta.update(f = f, E_med = q.sum(), lost_med = lq)
        if kind == 'reps':
            reps = [ jet.quench(tc.rng_of(pool, i, 2 + k), opts['mode'])
                     for k in range(opts['reps']) ]
            imgs['reps'] = np.stack([ r[0] for r in reps ])
            meta['f'] = reps[0][1]
    if kind in ('ue', 'split', 'mix', 'pairue', 'medmix'):
        info = {}
        imgs['ue'] = ue(pool, i, info)
        meta.update(m = info['m'], v2 = info['v2'], psi = info['psi'],
                    minijets = info['minijets'])
    return ({ k : v.astype(np.float32) for (k, v) in imgs.items() }, meta)

def bank_parent(task):
    """Repeated quenchings of one test parent (its role-0 particles)."""
    (pool, opts, i, reps) = task
    jet = vacuum(pool, i, opts)
    out = [ jet.quench(tc.rng_of(pool, i, 2 + k), opts['mode']) for k in range(1, reps + 1) ]
    return (np.stack([ o[0] for o in out ]).astype(np.float32),
            np.array([ o[1] for o in out ]), np.array([ o[2] for o in out ]))

_CPUS = os.sched_getaffinity(0)

def use_all_cpus():
    os.sched_setaffinity(0, _CPUS)

def generate(pool, kind, n, opts, procs, chunk):
    tasks = [ (pool, kind, opts, i) for i in range(n) ]
    with Pool(procs, initializer = use_all_cpus) as p:
        res = p.map(parent, tasks, chunksize = chunk)
    imgs = { k : np.stack([ r[0][k] for r in res ]) for k in res[0][0] }
    meta = { k : np.array([ r[1][k] for r in res ]) for k in res[0][1] }
    return (imgs, meta)

# --- building

def save_pool(name, imgs, meta, kind):
    d = cache_dir()
    if kind == 'jet' or kind == 'broad':
        x = imgs['jet']
    elif kind == 'ue':
        x = imgs['ue']
    elif kind == 'mix':
        x = imgs['jet'] + imgs['ue']
    elif kind == 'med':
        x = imgs['med']
    elif kind == 'medmix':
        x = imgs['med'] + imgs['ue']
    elif kind == 'reps':
        np.save(os.path.join(d, f'{name}_reps.npy'), imgs['reps'].astype(np.float16))
        x = imgs['jet']
    else:
        raise ValueError(kind)
    np.save(os.path.join(d, f'{name}.npy'), x.astype(np.float16))
    np.savez(os.path.join(d, f'{name}_meta.npz'), **meta)
    return x

def save_set(name, imgs, meta):
    np.savez(os.path.join(cache_dir(), f'{name}.npz'),
             **{ k : v.astype(np.float32) for (k, v) in imgs.items() }, **meta)

def bank_parents(meta):
    e = meta['E'][:2000]
    order = []
    for q in (np.arange(BANK_N) + 0.5) / BANK_N:
        dist = np.abs(e - np.quantile(e, q))
        dist[order] = np.inf
        order.append(int(np.argmin(dist)))
    illu = []
    for q in (0.1, 0.4, 0.7, 0.95):
        dist = np.abs(e - np.quantile(e, q))
        dist[order + illu] = np.inf
        illu.append(int(np.argmin(dist)))
    return (np.array(order), np.array(illu))

def build_bank(name, set_name, procs):
    (pool, _, _, opts) = SETS[set_name]
    with np.load(os.path.join(cache_dir(), f'{set_name}.npz')) as f:
        meta = { 'E' : f['E'] }
    (order, illu) = bank_parents(meta)
    tasks = [ (pool, opts, int(i), BANK_REPS) for i in order ] \
        + [ (pool, opts, int(i), ILLU_REPS) for i in illu ]
    with Pool(procs, initializer = use_all_cpus) as p:
        res = p.map(bank_parent, tasks, chunksize = 1)
    np.savez(os.path.join(cache_dir(), f'{name}.npz'),
             parents = order, reps = np.stack([ r[0] for r in res[:len(order)] ]),
             f = np.stack([ r[1] for r in res[:len(order)] ]),
             lost = np.stack([ r[2] for r in res[:len(order)] ]),
             illu_parents = illu, illu_reps = np.stack([ r[0] for r in res[len(order):] ]),
             illu_f = np.stack([ r[1] for r in res[len(order):] ]))
    print(f'{name}: {len(order)} parents x {BANK_REPS}, {len(illu)} x {ILLU_REPS}', flush = True)

def fit_norms():
    rng = np.random.default_rng(tc.SEED)
    stats = { '_bias' : fc.BIAS }
    for (kind, pools) in NORMS.items():
        vals = []
        for name in pools:
            x = np.load(os.path.join(cache_dir(), f'{name}.npy'), mmap_mode = 'r')
            idx = np.sort(rng.choice(len(x), 20000, replace = False))
            vals.append(np.log(np.asarray(x[idx], np.float64) + fc.BIAS))
        v = np.concatenate([ a.ravel() for a in vals ])
        stats[kind] = (float(v.mean()), float(v.std()))
    with open(os.path.join(root(), 'norm.json'), 'w', encoding = 'utf-8') as f:
        json.dump(stats, f, indent = 4)
    return stats

def build(cmdargs):
    os.makedirs(cache_dir(), exist_ok = True)
    names = cmdargs.only.split(',') if cmdargs.only else list(POOLS) + list(SETS) + list(BANKS)
    manifest_path = os.path.join(root(), 'manifest.json')
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding = 'utf-8') as f:
            manifest = json.load(f)
    for name in names:
        if name in BANKS:
            build_bank(name, BANKS[name], cmdargs.procs)
            manifest[name] = { 'from' : BANKS[name], 'reps' : BANK_REPS, 'parents' : BANK_N,
                               'illustration_reps' : ILLU_REPS }
            continue
        (pool, kind, n, opts) = POOLS.get(name) or SETS[name]
        (imgs, meta) = generate(pool, kind, n, opts, cmdargs.procs, cmdargs.chunk)
        if name in POOLS:
            save_pool(name, imgs, meta, kind)
        else:
            save_set(name, imgs, meta)
        manifest[name] = {
            'pool_id' : pool, 'kind' : kind, 'n' : n, 'options' : opts,
            'split' : 'train' if name in POOLS else ('val' if name[0] == 'v' else 'test'),
            'seed' : f'numpy SeedSequence([{tc.SEED}, {pool}, parent, role])',
            'mean_E' : float(np.nanmean(meta['E'])),
            'mean_E_img' : float(np.nanmean(meta['E_img'])),
            'lost_share_vac' : float(np.nansum(meta['lost']) / max(np.nansum(meta['E']), 1e-9)),
            'lost_share_med' : float(np.nansum(meta['lost_med'])
                                     / max(np.nansum(meta['E'][np.isfinite(meta['f'])]), 1e-9)),
            'mean_f' : float(np.nanmean(meta['f'])) if np.isfinite(meta['f']).any() else None,
        }
        print(f'{name}: {manifest[name]}', flush = True)
        with open(manifest_path, 'w', encoding = 'utf-8') as f:
            json.dump(manifest, f, indent = 4)
    if not cmdargs.only:
        manifest['norm'] = fit_norms()
        with open(manifest_path, 'w', encoding = 'utf-8') as f:
            json.dump(manifest, f, indent = 4)
        print('norm', manifest['norm'], flush = True)

# --- checks against the deck's quoted toy numbers

def load_set(name):
    with np.load(os.path.join(cache_dir(), f'{name}.npz')) as f:
        return { k : f[k] for k in f.files }

def checks():
    """The reimplementation against the toy numbers the deck quotes, and
    the geometry facts the exercises depend on (acceptance, retained energy,
    the protected region). docs/flow/jamie_otfm/generator_checks.json."""
    # pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
    import torch
    import jamie_obs as jo
    dev = torch.device('cpu')
    torch.set_num_threads(16)
    out = {}
    t3 = load_set('t3_pair')
    ax3 = np.stack([ t3['eta0'], t3['phi0'] ], 1)
    ov = jo.observables(t3['jet'], ax3, dev)
    oq = jo.observables(t3['med'], ax3, dev)
    out['clean_jets_t3'] = {
        'n' : len(ax3),
        'cone_vac_med' : [ float(ov['cone'].mean()), float(oq['cone'].mean()) ],
        'deck_cone' : '28.2 -> 21.9 GeV (slide 20), 28.2 -> 21.8 (slide 145)',
        'ring_vac_med' : [ float(ov['ring'].mean()), float(oq['ring'].mean()) ],
        'deck_ring' : '0.5 -> 6.4 GeV (slide 20), 0.4 -> 6.6 (slide 145)',
        'recoil_in_ring_mean_sd' : [ float((oq['ring'] - ov['ring']).mean()),
                                     float((oq['ring'] - ov['ring']).std()) ],
        'deck_recoil' : '6.0 +- 4.3 GeV (slide 22)',
        'mass_vac_med' : [ float(ov['mass'].mean()), float(oq['mass'].mean()) ],
        'deck_mass' : '2.97 -> 2.96 GeV (slide 74)',
        'girth_vac_med' : [ float(ov['girth'].mean()), float(oq['girth'].mean()) ],
        'deck_girth_shift' : '+0.027 (slide 73), +0.028 (slide 117)',
        'cone_shift' : float((oq['cone'] - ov['cone']).mean()),
        'deck_cone_shift' : '-6.3 GeV (slide 70)',
        'towers_above_0_in_cone_vac' : float((np.asarray(t3['jet']) > 0).reshape(len(ax3), -1).sum(1).mean()),
        'zlead_vac' : float(ov['zlead'].mean()),
        'deck_towers' : '7.9 towers with energy per cone, 48% in one tower (slide 55)',
        'far_med_minus_vac' : [ float((oq['far'] - ov['far']).mean()),
                                float(np.mean(np.abs(oq['far'] - ov['far']) > 1e-6)) ],
        'retained_energy_med_over_vac' : [
            float(np.mean(oq['total'] / ov['total'])), float(np.std(oq['total'] / ov['total'])),
            float(np.mean(np.abs(oq['total'] / ov['total'] - 1) > 0.01)) ],
        'lost_share_vac_med' : [ float(t3['lost'].sum() / t3['E'].sum()),
                                 float(t3['lost_med'].sum() / t3['E'].sum()) ],
        'corr_particle_girth_image_girth' : float(np.corrcoef(t3['g'], ov['girth'])[0, 1]),
        'corr_f_cone_loss' : float(np.corrcoef(t3['f'], ov['cone'] - oq['cone'])[0, 1]),
    }
    (_, _, dr) = jo.offsets(ax3[:2000], dev)
    out['clean_jets_t3']['ring_towers_mean'] = float(((dr >= 0.4) & (dr < 1.0)).sum((1, 2)).float().mean())
    out['clean_jets_t3']['cone_towers_mean'] = float((dr < 0.4).sum((1, 2)).float().mean())
    out['clean_jets_t3']['deck_towers_count'] = 'cone 52 (slide 9), ring 289 (slide 22)'
    tg = load_set('t3g_pair')
    out['girth_rule_t3g'] = {
        'f_mean_sd' : [ float(tg['f'].mean()), float(tg['f'].std()) ],
        'beta26_mean_sd' : [ 0.25, float(np.sqrt(2 * 6 / (8**2 * 9))) ],
        'deck' : 'same mean and similar spread as Beta(2,6) (slide 161)',
        'clipped_share' : float(np.mean((tg['f'] <= 0.02 + 1e-9) | (tg['f'] >= 0.6 - 1e-9))),
    }
    t1 = load_set('t1_mix')
    ax1 = np.stack([ t1['eta0'], t1['phi0'] ], 1)
    oj = jo.observables(t1['jet'], ax1, dev, substructure = False)
    ob = jo.observables(t1['ue'], ax1, dev, substructure = False)
    om = jo.observables(t1['jet'] + t1['ue'], ax1, dev, substructure = False)
    (_, _, dr1) = jo.offsets(ax1, dev)
    dr1 = dr1.numpy()
    mix = t1['jet'] + t1['ue']
    far = dr1 > 1.2
    rho = (mix * far).sum((1, 2)) / far.sum((1, 2))
    area = (dr1 < 0.4).sum((1, 2))
    e_rhoa = om['cone'] - rho * area
    out['subtraction_t1'] = {
        'n' : len(ax1), 'ue_per_tower' : float(t1['ue'].mean()),
        'ue_in_cone_mean_sd' : [ float(ob['cone'].mean()), float(ob['cone'].std()) ],
        'deck_ue_cone' : '29.2 GeV in the example, 31 on average (slide 7)',
        'ue_in_ring_mean' : float(ob['ring'].mean()), 'deck_ue_ring' : '170 GeV (slide 22)',
        'rho_area_offset_rms' : [ float((e_rhoa - oj['cone']).mean()),
                                  float((e_rhoa - oj['cone']).std()) ],
        'deck_rho_area' : 'offset +0.7-0.8 GeV, RMS 4.4 (slides 43, 51, 151)',
        'jet_cone_mean' : float(oj['cone'].mean()),
        'lost_share_vac' : float(t1['lost'].sum() / t1['E'].sum()),
        'rho_r_gt_1p2_mean' : float(rho.mean()),
        'ring_wobble_after_own_rho' : float((ob['ring'] - rho * ((dr1 >= 0.4) & (dr1 < 1.0)).sum((1, 2))).std()),
        'deck_ring_wobble' : '+-12 GeV after each event\'s own rho (slide 22)',
    }
    tu = load_set('t1_ue')['ue']
    cs = jo.cone_sums_everywhere(tu, dev).numpy()
    rho_u = tu.mean((1, 2))
    a_cone = float(fc.ev.cone_kernel(0.4).sum())
    sub = cs - rho_u[:, None, None] * a_cone
    inner = np.abs(tc.ETA_C) < 0.7
    out['fakes_t1_ue'] = {
        'n' : len(tu), 'cone_towers' : a_cone,
        'largest_cone_minus_rhoA_median_anywhere' : float(np.median(sub.max((1, 2)))),
        'share_above_10_anywhere' : float(np.mean(sub.max((1, 2)) > 10)),
        'share_above_10_eta_lt_0p7' : float(np.mean(sub[:, inner].max((1, 2)) > 10)),
        'deck' : 'flat rho x A: biggest cone typically 12 GeV, above 10 GeV in 88% (slide 44)',
    }
    with open(os.path.join('docs', 'flow', 'jamie_otfm', 'generator_checks.json'), 'w',
              encoding = 'utf-8') as f:
        json.dump(out, f, indent = 4)
    print(json.dumps(out, indent = 4))

def main():
    cmdargs = parse_cmdargs()
    if cmdargs.build:
        build(cmdargs)
    if cmdargs.checks:
        checks()

if __name__ == '__main__':
    main()
