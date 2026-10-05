#!/usr/bin/env python
"""CPU side of the PYTHIA -> JEWEL translation pilot's evaluation
(FLOW_NOTES.md, "PYTHIA -> JEWEL translation pilot"): every test metric,
table and figure, from the outputs saved by translation_eval.py --generate.

Samples (20k test jets each; every output clipped at 0, nothing else):
    JEWEL test     the target sample
    JEWEL ref      a second held-out JEWEL sample: the finite-sample floor
    identity       the PYTHIA test inputs, unchanged
    random JEWEL   20k jets of the JEWEL training pool, drawn without
                   replacement (seed 0), one per input, ignoring it
    MODEL          a model's outputs on the PYTHIA test inputs (--models
                   'LABEL=STEM,...', STEM of OUTDIR/.../translation/outputs)

Fixed-crop observables are those of the 53 cone towers around the canvas
centre (translation_eval.jet_observables); refound jets are the leading
anti-kT R = 0.4 jet that FastJet finds in the canvas towers (bench_jets.py's
constituents, substructure and soft drop; the canvas placed at eta = 0, phi =
pi, which leaves every distance unchanged). Both use the common final
selection (E or pT >= 10 GeV) on every sample, and report the share that
passes it.

Written to --out (docs/flow/translation):
    population.csv   W1 / sigma_JEWEL (bootstrap sd) of E, mass, girth, p_T^D,
                     z_lead, z_g, R_g, acceptance, and the pre-registered flags
    joint.csv        W1 of girth and z_lead in pT bins, their mean, and the
                     largest difference of the correlation matrices
    refound.csv      the leading refound jet's observables, likewise
    dependence.csv   input-output correlations (bootstrap sd), changes, and
                     the own-input preference rate (normalised-shape EMD)
    fidelity.csv     occupancy, soft energy, leading tower, core fraction;
                     the artefact flags
    migration.csv    output E bins and acceptance by input E bin
    variability.csv  alpha-DSBM: spread of 8 samples per input
    verdict.csv      the pre-registered reading, item by item
    figures tr_*.png
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
import translation_eval as te
from closure_data import OFFSET
from jet_fidelity import EMD
from substructure import Geometry

sys.path.insert(0, os.path.expanduser('~/pyext/jets'))
import fastjet._swig as fj            # pylint: disable=wrong-import-position,wrong-import-order
from bench_jets import substructure   # pylint: disable=wrong-import-position

ev = fc.ev
PT_BINS  = [ (10, 20), (20, 30), (30, 60) ]
E_EDGES  = [ 0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 1000 ]
IN_EDGES = [ 10, 15, 20, 25, 30, 40, 50, 70 ]
REFOUND  = [ 'pt', 'mass', 'girth', 'ptd', 'zlead', 'zg', 'rg' ]
LABELS   = { 'E' : 'E (cone), GeV', 'mass' : 'mass, GeV', 'girth' : 'girth',
             'ptd' : '$p_T^D$', 'zlead' : '$z_{lead}$', 'zg' : '$z_g$',
             'rg' : '$R_g$', 'core' : 'core fraction', 'lead' : 'leading tower, GeV',
             'pt' : 'refound $p_T$, GeV' }
REFS = [ 'JEWEL ref', 'random JEWEL', 'identity' ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'PYTHIA -> JEWEL report')
    parser.add_argument('--models', required = True,
        help = "comma separated LABEL=STEM (outputs/STEM.npy)")
    parser.add_argument('--multi', default = None,
        help = 'LABEL=STEM of the K-sample outputs (alpha-DSBM)')
    parser.add_argument('--curves', default = None,
        help = "comma separated LABEL=RUN[+RUN] (validation curves)")
    parser.add_argument('--boot', type = int, default = 200)
    parser.add_argument('--procs', type = int, default = 32)
    parser.add_argument('--out', default = 'docs/flow/translation')
    return parser.parse_args()

# --- samples

def samples(cmdargs):
    """{name: (canvas (N, 16, 16) GeV >= 0, rows)} and the PYTHIA inputs."""
    test  = te.load_set('test', 'pythia')
    jt    = te.load_set('test', 'jewel')
    jr    = te.load_set('ref', 'jewel')
    pool  = np.load(fc.cache_path('tr_jewel'), mmap_mode = 'r')
    meta  = te.train_meta('jewel')
    pick  = np.sort(np.random.default_rng(0).choice(len(pool), len(test['canvas']),
                                                   replace = False))
    order = np.random.default_rng(1).permutation(len(pick))
    rand  = np.asarray(pool[pick]).astype(np.float32)[order]
    out = { 'JEWEL test' : (jt['canvas'], jt['row']),
            'JEWEL ref' : (jr['canvas'], jr['row']),
            'random JEWEL' : (rand, meta['row'][pick][order]),
            'identity' : (test['canvas'], test['row']) }
    info = {}
    for spec in cmdargs.models.split(','):
        (label, stem) = spec.split('=', 1)
        raw = np.load(os.path.join(te.output_dir(), f'{stem}.npy'))
        with open(os.path.join(te.output_dir(), f'{stem}.json'), encoding = 'utf-8') as f:
            info[label] = json.load(f)
        info[label]['raw_min_gev'] = float(raw.min())
        info[label]['clipped_energy_share'] = float(
            -np.clip(raw, None, 0).sum() / np.clip(raw, 0, None).sum())
        out[label] = (te.clip(raw), test['row'])
    return (out, test, info)

# --- refound jets (FastJet), in parallel over chunks

_CANVASES = {}
_CPUS = os.sched_getaffinity(0)

def use_all_cpus():
    """Pool initializer: the parent's CPU mask (forked workers came up
    pinned to one core under SLURM, c.f. translation_data.py)."""
    os.sched_setaffinity(0, _CPUS)

def canvas_particles(img):
    parts = []
    for (r, c) in zip(*np.nonzero(img > 0)):
        pt  = float(img[r, c])
        eta = (r - 8) * ev.DETA
        phi = math.pi + (c - 8) * ev.DPHI
        parts.append(fj.PseudoJet(pt * math.cos(phi), pt * math.sin(phi),
                                  pt * math.sinh(eta), pt * math.cosh(eta)))
    return parts

def refind_chunk(task):
    (name, start, stop) = task
    use_all_cpus()
    imgs = _CANVASES[name]
    centre = fj.PseudoJet(math.cos(math.pi), math.sin(math.pi), 0.0, 1.0)
    rows = []
    for k in range(start, stop):
        parts = canvas_particles(np.asarray(imgs[k], dtype = np.float64))
        if not parts:
            rows.append([ np.nan ] * 9)
            continue
        cs   = fj.ClusterSequence(parts, fj.JetDefinition(fj.antikt_algorithm, 0.4))
        jets = fj.sorted_by_pt(cs.inclusive_jets(0.0))
        j    = jets[0]
        s    = substructure(j, 0.4)
        rows.append([ j.pt(), j.delta_R(centre), s['mass'], s['girth'], s['ptd'],
                      s['zlead'], s['zg'], s['rg'],
                      float(sum(1 for x in jets if x.pt() > 5.0)) ])
    return rows

def refound(canvases, procs):
    global _CANVASES       # pylint: disable=global-statement
    _CANVASES = { k : v[0] for (k, v) in canvases.items() }
    out = {}
    with Pool(procs, initializer = use_all_cpus) as pool:
        for (name, img) in _CANVASES.items():
            step  = max(1, len(img) // (procs * 4))
            tasks = [ (name, s, min(s + step, len(img))) for s in range(0, len(img), step) ]
            rows  = np.array(sum(pool.map(refind_chunk, tasks), []))
            out[name] = dict(zip([ 'pt', 'offset', 'mass', 'girth', 'ptd', 'zlead',
                                   'zg', 'rg', 'n_jets5' ], rows.T))
            print(f'refound jets: {name}', flush = True)
    return out

# --- shape EMD to the own input and to another input

_PAIRS = {}

def emd_chunk(task):
    (name, start, stop) = task
    use_all_cpus()
    torch.set_num_threads(1)
    (a, b, c) = _PAIRS[name]
    geo = Geometry(torch.device('cpu'))
    emd = EMD(geo)
    def norm(x):
        w = torch.as_tensor(x[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9]).float() \
            .clamp(min = 0) * geo.mask
        return w / w.sum((1, 2), keepdim = True).clamp(min = 1e-12)
    (na, nb, nc) = (norm(a[start:stop]), norm(b[start:stop]), norm(c[start:stop]))
    return np.stack([ emd(na, nb)[0], emd(na, nc)[0] ], axis = 1)

def preference(outputs, inputs, procs):
    """Normalised-shape EMD of each output to its own input and to another
    test input (a fixed derangement), per sample."""
    global _PAIRS          # pylint: disable=global-statement
    n = len(inputs)
    # another input for every jet: a cyclic shift of the (random) test order
    shift = 1 + int(np.random.default_rng(2).integers(n - 1))
    perm  = (np.arange(n) + shift) % n
    _PAIRS = { k : (v, inputs, inputs[perm]) for (k, v) in outputs.items() }
    out = {}
    with Pool(procs, initializer = use_all_cpus) as pool:
        for name in _PAIRS:
            step  = max(1, n // (procs * 4))
            tasks = [ (name, s, min(s + step, n)) for s in range(0, n, step) ]
            d = np.concatenate(pool.map(emd_chunk, tasks))
            out[name] = d
            print(f'shape EMD to inputs: {name}', flush = True)
    return out

# --- tables

def boot_sd(fn, arrays, boot, seed = 3):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(boot):
        i = rng.integers(0, n, n)
        vals.append(fn(*[ a[i] for a in arrays ]))
    return float(np.nanstd(vals))

def population_table(obs, boot, key = 'E', names = te.OBS, cut = te.MIN_JET):
    """W1 / sigma_JEWEL-test of each sample after the final selection."""
    ref = obs['JEWEL test']
    sel_r = ref[key] >= cut
    rows = []
    rng = np.random.default_rng(4)
    for (name, o) in obs.items():
        if name == 'JEWEL test':
            continue
        sel = o[key] >= cut
        row = { 'sample' : name, 'n' : int(len(o[key])),
                'acceptance' : float(sel.mean()) }
        for q in names:
            r = te.finite(ref[q][sel_r])
            (w, e) = te.w1_sigma(o[q][sel], r, r.std(), boot, rng)
            row[f'w1_{q}'] = w
            row[f'w1_{q}_sd'] = e
            row[f'mean_{q}'] = float(np.nanmean(o[q][sel]))
            row[f'nan_{q}'] = float(np.mean(~np.isfinite(o[q][sel])))
        rows.append(row)
    df = pd.DataFrame(rows)
    ident = df[df['sample'] == 'identity'].iloc[0]
    floor = df[df['sample'] == 'JEWEL ref'].iloc[0]
    for q in names:
        comb_id = np.sqrt(df[f'w1_{q}_sd']**2 + ident[f'w1_{q}_sd']**2)
        comb_fl = np.sqrt(df[f'w1_{q}_sd']**2 + floor[f'w1_{q}_sd']**2)
        df[f'moves_{q}'] = df[f'w1_{q}'] < ident[f'w1_{q}'] - 3 * comb_id
        df[f'consistent_{q}'] = df[f'w1_{q}'] < floor[f'w1_{q}'] + 3 * comb_fl
        gap = ident[f'w1_{q}'] - floor[f'w1_{q}']
        df[f'closed_{q}'] = (ident[f'w1_{q}'] - df[f'w1_{q}']) / gap if gap > 0 else np.nan
    df['n_moves'] = df[[ f'moves_{q}' for q in names ]].sum(axis = 1)
    df['n_consistent'] = df[[ f'consistent_{q}' for q in names ]].sum(axis = 1)
    return df

def conditional_w1(o, ref, q, boot, rng):
    """W1 / sigma of q in each pT bin (both samples E >= 10 GeV), and their
    mean with a bootstrap sd; a bin with fewer than 50 jets of a sample makes
    the mean undefined (then the joint comparison does not count as moved)."""
    vals, resampled = [], []
    for (lo, hi) in PT_BINS:
        a = te.finite(o[q][(o['E'] >= lo) & (o['E'] < hi)])
        b = te.finite(ref[q][(ref['E'] >= lo) & (ref['E'] < hi)])
        sd = b.std()
        vals.append(wasserstein_distance(a, b) / sd if len(a) > 50 else np.nan)
        resampled.append([ wasserstein_distance(rng.choice(a, len(a)),
                                                rng.choice(b, len(b))) / sd
                           if len(a) > 50 else np.nan for _ in range(boot) ])
    mean_boot = np.mean(np.array(resampled), axis = 0)
    return (vals, float(np.mean(vals)), float(np.std(mean_boot)))

CORR_KEYS = [ 'E', 'mass', 'girth', 'ptd', 'zlead' ]

def corr_matrix(o):
    sel = o['E'] >= te.MIN_JET
    m = np.stack([ np.log(o['E'][sel]) if q == 'E' else o[q][sel] for q in CORR_KEYS ])
    return np.corrcoef(m)

def joint_table(obs, boot):
    ref = obs['JEWEL test']
    rng = np.random.default_rng(5)
    rows = []
    c_ref = corr_matrix(ref)
    for (name, o) in obs.items():
        if name == 'JEWEL test':
            continue
        row = { 'sample' : name }
        for q in [ 'girth', 'zlead' ]:
            (vals, mean, sd) = conditional_w1(o, ref, q, boot, rng)
            for ((lo, hi), v) in zip(PT_BINS, vals):
                row[f'w1_{q}_pt{lo}_{hi}'] = v
            row[f'w1_{q}_given_pt'] = mean
            row[f'w1_{q}_given_pt_sd'] = sd
        row['corr_maxdiff'] = float(np.abs(corr_matrix(o) - c_ref).max())
        rows.append(row)
    df = pd.DataFrame(rows)
    ident = df[df['sample'] == 'identity'].iloc[0]
    for q in [ 'girth', 'zlead' ]:
        comb = np.sqrt(df[f'w1_{q}_given_pt_sd']**2 + ident[f'w1_{q}_given_pt_sd']**2)
        df[f'moves_{q}_given_pt'] = df[f'w1_{q}_given_pt'] < \
            ident[f'w1_{q}_given_pt'] - 3 * comb
    return df

def dependence_table(obs, emds, boot):
    """Input-output relations: correlations (bootstrap sd), changes, and the
    own-input preference rate."""
    o_in = obs['identity']
    rows = []
    for (name, o) in obs.items():
        if name in ('JEWEL test', 'JEWEL ref'):
            continue
        row = { 'sample' : name }
        for q in te.DEP:
            (a, b) = (np.asarray(o_in[q], float), np.asarray(o[q], float))
            ok = np.isfinite(a) & np.isfinite(b)
            (a, b) = (a[ok], b[ok])
            corr = (lambda x, y: np.corrcoef(x, y)[0, 1] if y.std() > 0 else np.nan)
            row[f'corr_{q}'] = float(corr(a, b))
            row[f'corr_{q}_sd'] = boot_sd(corr, [ a, b ], boot) if name != 'identity' else 0.0
            row[f'spearman_{q}'] = float(pd.Series(a).corr(pd.Series(b), method = 'spearman'))
            d = b - a
            row[f'delta_{q}_mean'] = float(d.mean())
            row[f'delta_{q}_sd'] = float(d.std())
            for (p, v) in zip([ 16, 50, 84 ], np.percentile(d, [ 16, 50, 84 ])):
                row[f'delta_{q}_p{p}'] = float(v)
        row['E_ratio_mean'] = float(np.mean(o['E'] / o_in['E']))
        row['E_ratio_sd'] = float(np.std(o['E'] / o_in['E']))
        if name in emds:
            (own, other) = emds[name].T
            ok = np.isfinite(own) & np.isfinite(other)
            p = float(np.mean(own[ok] < other[ok]))
            row.update({ 'shape_emd_to_input' : float(np.mean(own[ok])),
                         'shape_emd_to_other_input' : float(np.mean(other[ok])),
                         'preference_rate' : p,
                         'preference_se' : float(np.sqrt(p * (1 - p) / ok.sum())) })
        rows.append(row)
    df = pd.DataFrame(rows)
    df['dependence_beyond_random'] = (
        (df['corr_E'] > 3 * df['corr_E_sd']) & (df['corr_girth'] > 3 * df['corr_girth_sd'])
        & (df['preference_rate'] > 0.5 + 3 * df['preference_se']))
    return df

FIDELITY = [ 'n_pos', 'n_0p01', 'n_0p1', 'n1', 'soft_0p5', 'lead', 'core', 'zlead' ]

def fidelity_table(obs):
    rows = []
    for (name, o) in obs.items():
        row = { 'sample' : name }
        for q in FIDELITY:
            v = np.asarray(o[q], float)
            row[f'{q}_mean'] = float(v.mean())
            row[f'{q}_se'] = float(v.std() / np.sqrt(len(v)))
        rows.append(row)
    df = pd.DataFrame(rows)
    (py, jw) = (df[df['sample'] == 'identity'].iloc[0], df[df['sample'] == 'JEWEL test'].iloc[0])
    df['soft_floor'] = False
    for q in [ 'n_0p01', 'soft_0p5' ]:
        df['soft_floor'] |= (df[f'{q}_mean'] > 1.1 * py[f'{q}_mean']) \
            & (df[f'{q}_mean'] > 1.1 * jw[f'{q}_mean'])
    df['flattening'] = False
    for q in [ 'zlead', 'core' ]:
        lo = [ ref[f'{q}_mean'] - 3 * np.sqrt(df[f'{q}_se']**2 + ref[f'{q}_se']**2)
               for ref in (py, jw) ]
        df['flattening'] |= (df[f'{q}_mean'] < lo[0]) & (df[f'{q}_mean'] < lo[1])
    return df

def migration_table(obs):
    o_in = obs['identity']
    rows = []
    for (name, o) in obs.items():
        if name in ('JEWEL test', 'JEWEL ref', 'identity'):
            continue
        for (lo, hi) in zip(IN_EDGES[:-1], IN_EDGES[1:]):
            sel = (o_in['E'] >= lo) & (o_in['E'] < hi)
            h = np.histogram(o['E'][sel], E_EDGES)[0] / max(sel.sum(), 1)
            rows.append({ 'sample' : name, 'input_lo' : lo, 'input_hi' : hi,
                          'n' : int(sel.sum()),
                          'acceptance' : float(np.mean(o['E'][sel] >= te.MIN_JET)),
                          'E_out_median' : float(np.median(o['E'][sel])),
                          **{ f'out_{a}_{b}' : float(v) for (a, b, v)
                              in zip(E_EDGES[:-1], E_EDGES[1:], h) } })
    return pd.DataFrame(rows)

def variability_table(multi, test, obs_ref, device):
    """alpha-DSBM: 8 samples of each of the first n test inputs."""
    (label, stem) = multi.split('=', 1)
    outs = te.clip(np.load(os.path.join(te.output_dir(), f'{stem}.npy')))
    (k, n) = outs.shape[:2]
    rows_in = test['row'][:n]
    o_in = te.jet_observables(test['canvas'][:n], rows_in, device)
    o = [ te.jet_observables(x, rows_in, device) for x in outs ]
    row = { 'model' : label, 'samples' : k, 'inputs' : n }
    for q in te.DEP + [ 'ptd', 'zlead' ]:
        v = np.stack([ x[q] for x in o ])
        sd_ref = np.nanstd(obs_ref[q])
        row[f'spread_{q}_over_sd'] = float(np.nanmean(np.nanstd(v, axis = 0)) / sd_ref)
        m = np.nanmean(v, axis = 0)
        ok = np.isfinite(m) & np.isfinite(o_in[q])
        row[f'corr_mean_{q}'] = float(np.corrcoef(o_in[q][ok], m[ok])[0, 1])
        row[f'corr_single_{q}'] = float(np.corrcoef(o_in[q][ok], v[0][ok])[0, 1])
    geo = Geometry(device)
    emd = EMD(geo)
    def norm(x):
        w = torch.as_tensor(x[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9]).float() \
            .clamp(min = 0) * geo.mask
        return w / w.sum((1, 2), keepdim = True).clamp(min = 1e-12)
    m = min(n, 500)
    between = emd(norm(outs[0][:m]), norm(outs[1][:m]))[0]
    to_in   = emd(norm(outs[0][:m]), norm(test['canvas'][:m]))[0]
    row['shape_emd_between_samples'] = float(np.nanmean(between))
    row['shape_emd_sample_to_input'] = float(np.nanmean(to_in))
    return (pd.DataFrame([ row ]), outs)

# --- verdict

def verdict(pop, joint, dep, fid, models):
    rows = []
    for m in models:
        p = pop[pop['sample'] == m].iloc[0]
        j = joint[joint['sample'] == m].iloc[0]
        d = dep[dep['sample'] == m].iloc[0]
        f = fid[fid['sample'] == m].iloc[0]
        moved = [ q for q in te.OBS if p[f'moves_{q}'] ]
        cons  = [ q for q in te.OBS if p[f'consistent_{q}'] ]
        joint_ok = bool(j['moves_girth_given_pt'] and j['moves_zlead_given_pt'])
        rows.append({
            'model' : m, 'marginals_moved' : ', '.join(moved) or '-',
            'n_moved' : len(moved), 'consistent_with_jewel' : ', '.join(cons) or '-',
            'joint_girth_moved' : bool(j['moves_girth_given_pt']),
            'joint_zlead_moved' : bool(j['moves_zlead_given_pt']),
            'dependence_beyond_random' : bool(d['dependence_beyond_random']),
            'soft_floor' : bool(f['soft_floor']), 'flattening' : bool(f['flattening']),
            'useful' : bool(len(moved) >= 4 and joint_ok and d['dependence_beyond_random']
                            and not f['soft_floor'] and not f['flattening']) })
    return pd.DataFrame(rows)

# --- figures

def colour(name):
    """One colour per model in every figure: the pilot's (OT-CFM green,
    alpha-DSBM red), and for the CycleGAN comparison OT-FM blue and CycleGAN
    orange (a darker step for longer training): a pair that passes the
    colour-vision check, unlike green/red."""
    low = name.lower()
    for (key, c) in [ ('jewel test', 'k'), ('jewel ref', '#555555'),
                      ('random', '#bcbd22'), ('identity', '#7f7f7f'),
                      ('euler', '#98df8a'), ('ot-cfm', '#2ca02c'),
                      ('dsbm', '#d62728'), ('ot-fm', '#2a78d6'),
                      ('cyclegan 24', '#9c3d14'), ('cyclegan 8', '#c5521f'),
                      ('cyclegan', '#eb6834') ]:
        if key in low:
            return c
    return '#1f77b4'

def style(name):
    return { 'JEWEL test' : dict(lw = 1.8, ls = '-'), 'identity' : dict(lw = 1.2, ls = ':'),
             'random JEWEL' : dict(lw = 1.0, ls = '--') }.get(name, dict(lw = 1.2, ls = '-'))

def fig_marginals(obs, path, show, sel_key = 'E', names = te.OBS, title = ''):
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (fig, axes) = plt.subplots(1, len(names), figsize = (2.0 * len(names), 2.3))
    for (ax, q) in zip(axes, names):
        allv = te.finite(np.concatenate([ obs[s][q] for s in show ]))
        lo = 0 if q == sel_key else np.quantile(allv, 0.005)
        hi = np.quantile(allv, 0.995)
        bins = np.linspace(lo, hi, 36)
        for s in show:
            o = obs[s]
            sel = np.ones(len(o[q]), bool) if q == sel_key else o[sel_key] >= te.MIN_JET
            ax.hist(te.finite(o[q][sel]), bins, histtype = 'step', density = True,
                    color = colour(s), label = s, **style(s))
        if q == sel_key:
            ax.axvline(te.MIN_JET, color = 'k', lw = 0.6, ls = ':')
        ax.set_xlabel(LABELS.get(q, q), fontsize = 7)
        ax.tick_params(labelsize = 6)
        ax.set_yticks([])
    (h, l) = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc = 'lower center', ncol = len(l), fontsize = 6.5, frameon = False)
    if title:
        fig.suptitle(title, fontsize = 7.5)
    fig.tight_layout(rect = (0, 0.1, 1, 0.95 if title else 1))
    fig.savefig(path, dpi = 150)
    plt.close(fig)

def fig_joint(obs, path, models):
    # pylint: disable=import-outside-toplevel,too-many-locals
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = [ 'JEWEL test', 'identity' ] + models
    (fig, axes) = plt.subplots(2, len(show) + 1, figsize = (2.15 * (len(show) + 1), 4.3))
    edges = np.array([ 10, 13, 16, 20, 25, 30, 36, 45, 60 ])
    mids  = 0.5 * (edges[1:] + edges[:-1])
    for (r, q) in enumerate([ 'girth', 'zlead' ]):
        ref = obs['JEWEL test']
        rng = (np.quantile(te.finite(ref[q]), 0.002), np.quantile(te.finite(ref[q]), 0.998))
        for (c, s) in enumerate(show):
            o = obs[s]
            sel = (o['E'] >= 10) & (o['E'] < 60) & np.isfinite(o[q])
            ax = axes[r][c]
            h = np.histogram2d(o['E'][sel], o[q][sel], [ np.linspace(10, 60, 26),
                               np.linspace(*rng, 26) ])[0]
            ax.imshow(np.log10(h.T + 1), origin = 'lower', aspect = 'auto',
                      extent = (10, 60, *rng), cmap = 'viridis')
            ax.set_title(s, fontsize = 7)
            ax.tick_params(labelsize = 6)
            ax.set_xlabel('E (cone), GeV', fontsize = 6.5)
            if c == 0:
                ax.set_ylabel(LABELS[q], fontsize = 7)
        ax = axes[r][-1]
        for s in show + [ 'JEWEL ref' ]:
            if s == 'JEWEL ref':
                continue
            o = obs[s]
            means, errs = [], []
            for (lo, hi) in zip(edges[:-1], edges[1:]):
                v = te.finite(o[q][(o['E'] >= lo) & (o['E'] < hi)])
                means.append(v.mean() if len(v) > 20 else np.nan)
                errs.append(v.std() / np.sqrt(len(v)) if len(v) > 20 else np.nan)
            ax.errorbar(mids, means, errs, color = colour(s), label = s, marker = '.',
                        ms = 3, **style(s))
        ax.set_xlabel('E (cone), GeV', fontsize = 6.5)
        ax.set_title(f'mean {LABELS[q]}', fontsize = 7)
        ax.tick_params(labelsize = 6)
        if r == 0:
            ax.legend(fontsize = 5.5)
    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)

def fig_profiles(obs, path, models):
    """The slides' joint panels: mean girth and z_lead against the cone
    energy (E >= 10 GeV), with their standard errors."""
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    edges = np.array([ 10, 13, 16, 20, 25, 30, 36, 45, 60 ])
    mids  = 0.5 * (edges[1:] + edges[:-1])
    (fig, axes) = plt.subplots(1, 2, figsize = (4.6, 1.9))
    for (ax, q) in zip(axes, [ 'girth', 'zlead' ]):
        for s in [ 'JEWEL test', 'identity' ] + models:
            o = obs[s]
            means, errs = [], []
            for (lo, hi) in zip(edges[:-1], edges[1:]):
                v = te.finite(o[q][(o['E'] >= lo) & (o['E'] < hi)])
                means.append(v.mean() if len(v) > 20 else np.nan)
                errs.append(v.std() / np.sqrt(len(v)) if len(v) > 20 else np.nan)
            ax.errorbar(mids, means, errs, color = colour(s), label = s, marker = '.',
                        ms = 3, **style(s))
        ax.set_xlabel('E (cone), GeV', fontsize = 6.5)
        ax.set_title(f'mean {LABELS[q]}', fontsize = 7)
        ax.tick_params(labelsize = 6)
    axes[0].legend(fontsize = 5.5)
    fig.tight_layout()
    fig.savefig(path, dpi = 200)
    plt.close(fig)

def fig_changes_slide(obs, path, models):
    """The slides' change panels: E_out / E_in, the girth and the
    leading-tower change against the input energy (median, 16-84%)."""
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    o_in  = obs['identity']
    edges = np.array(IN_EDGES)
    mids  = 0.5 * (edges[1:] + edges[:-1])
    qs = [ ('E', '$E_{out} / E_{in}$'), ('girth', 'girth change'),
           ('lead', 'leading-tower change, GeV') ]
    (fig, axes) = plt.subplots(1, len(qs), figsize = (6.0, 1.9))
    for (ax, (q, name)) in zip(axes, qs):
        for s in models + [ 'random JEWEL' ]:
            o = obs[s]
            d = o['E'] / o_in['E'] if q == 'E' else o[q] - o_in[q]
            band = [ np.percentile(te.finite(d[(o_in['E'] >= a) & (o_in['E'] < b)]),
                                   [ 16, 50, 84 ]) for (a, b) in zip(edges[:-1], edges[1:]) ]
            (lo, med, hi) = np.array(band).T
            ax.plot(mids, med, color = colour(s), label = s, marker = '.', ms = 3, **style(s))
            ax.fill_between(mids, lo, hi, color = colour(s), alpha = 0.15, lw = 0)
        ax.axhline(1 if q == 'E' else 0, color = 'k', lw = 0.6)
        ax.set_xlabel('input E (cone), GeV', fontsize = 6.5)
        ax.set_title(name, fontsize = 7)
        ax.tick_params(labelsize = 6)
    axes[0].set_ylim(0, 2.2)
    axes[0].legend(fontsize = 5)
    fig.tight_layout()
    fig.savefig(path, dpi = 200)
    plt.close(fig)

def fig_changes(obs, path, models):
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    o_in = obs['identity']
    edges = np.array(IN_EDGES)
    mids  = 0.5 * (edges[1:] + edges[:-1])
    qs = [ ('E', 'E_out / E_in'), ('girth', 'girth change'), ('mass', 'mass change, GeV'),
           ('core', 'core fraction change'), ('lead', 'leading-tower change, GeV') ]
    (fig, grid) = plt.subplots(2, len(qs), figsize = (2.3 * len(qs), 4.6))
    for (ax, (q, name)) in zip(grid[1], qs):
        # the distributions of the changes, all inputs
        ds = { s : (obs[s]['E'] / o_in['E'] if q == 'E' else obs[s][q] - o_in[q])
               for s in models + [ 'random JEWEL' ] }
        allv = te.finite(np.concatenate(list(ds.values())))
        bins = np.linspace(*np.quantile(allv, [ 0.005, 0.995 ]), 41)
        for (s, d) in ds.items():
            ax.hist(te.finite(d), bins, histtype = 'step', density = True, color = colour(s),
                    label = s, **style(s))
        ax.axvline(1 if q == 'E' else 0, color = 'k', lw = 0.6)
        ax.set_xlabel(name, fontsize = 6.5)
        ax.set_yticks([])
        ax.tick_params(labelsize = 6)
    axes = grid[0]
    for (ax, (q, name)) in zip(axes, qs):
        for s in models + [ 'random JEWEL' ]:
            o = obs[s]
            d = o['E'] / o_in['E'] if q == 'E' else o[q] - o_in[q]
            med, lo, hi = [], [], []
            for (a, b) in zip(edges[:-1], edges[1:]):
                v = te.finite(d[(o_in['E'] >= a) & (o_in['E'] < b)])
                (x, y, z) = np.percentile(v, [ 16, 50, 84 ]) if len(v) > 20 else (np.nan,) * 3
                med.append(y)
                lo.append(x)
                hi.append(z)
            ax.plot(mids, med, color = colour(s), label = s, marker = '.', ms = 3, **style(s))
            ax.fill_between(mids, lo, hi, color = colour(s), alpha = 0.15, lw = 0)
        ax.axhline(1 if q == 'E' else 0, color = 'k', lw = 0.6)
        ax.set_xlabel('input E (cone), GeV', fontsize = 6.5)
        ax.set_title(name, fontsize = 7)
        ax.tick_params(labelsize = 6)
    axes[0].legend(fontsize = 5.5)
    fig.suptitle('Proposed changes, output minus its own input (top: median and 16-84% by input'
                 ' energy; bottom: all 20k inputs): the maps, not errors against a truth',
                 fontsize = 7)
    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)

def fig_migration(mig, path, show):
    """Output cone-energy bins by input bin, and the share passing the final
    selection."""
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_cols = [ c for c in mig.columns if c.startswith('out_') ]
    (fig, axes) = plt.subplots(1, len(show) + 1, figsize = (2.6 * (len(show) + 1), 2.6))
    for (ax, s) in zip(axes, show):
        d = mig[mig['sample'] == s]
        m = d[out_cols].to_numpy()
        ax.imshow(m.T, origin = 'lower', aspect = 'auto', cmap = 'viridis', vmin = 0, vmax = 1)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([ f'{a}-{b}' for (a, b) in zip(d.input_lo, d.input_hi) ],
                           rotation = 60, fontsize = 5)
        ax.set_yticks(range(len(out_cols)))
        ax.set_yticklabels([ c[4:].replace('_', '-').replace('-1000', '+') for c in out_cols ],
                           fontsize = 5)
        ax.axhline(1.5, color = 'w', lw = 0.6, ls = '--')
        ax.set_xlabel('input E (cone), GeV', fontsize = 6)
        ax.set_title(s, fontsize = 7)
    axes[0].set_ylabel('output E (cone), GeV', fontsize = 6)
    ax = axes[-1]
    for s in show:
        d = mig[mig['sample'] == s]
        x = 0.5 * (d.input_lo + d.input_hi)
        ax.plot(x, d.acceptance, marker = '.', color = colour(s), label = s, **style(s))
    ax.set_xlabel('input E (cone), GeV', fontsize = 6)
    ax.set_title('share with output E >= 10 GeV', fontsize = 7)
    ax.tick_params(labelsize = 6)
    ax.legend(fontsize = 5.5)
    fig.suptitle('Migration: output energy by input energy bin (columns sum to 1; white line: the'
                 ' 10 GeV final selection)', fontsize = 7)
    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)

def fig_towers(path, show):
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bins = np.linspace(-4.5, 1.8, 64)
    (fig, axes) = plt.subplots(1, 2, figsize = (7.2, 2.6))
    for s in show:
        (img, _) = _SAMPLES[s]
        w = img[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9][:, _MASK]
        pos = w[w > 0]
        h = np.histogram(np.log10(pos), bins)[0] / len(img)
        axes[0].step(bins[:-1], h, where = 'post', color = colour(s), label = s, **style(s))
        e = np.histogram(np.log10(pos), bins, weights = pos)[0] / len(img)
        axes[1].step(bins[:-1], e, where = 'post', color = colour(s), label = s, **style(s))
    axes[0].set_ylabel('towers per jet per bin', fontsize = 7)
    axes[1].set_ylabel('energy per jet per bin, GeV', fontsize = 7)
    for ax in axes:
        ax.set_xlabel('log10(tower E / GeV), cone towers > 0', fontsize = 7)
        ax.tick_params(labelsize = 6)
    axes[0].legend(fontsize = 5.5)
    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)

_SAMPLES = {}
_MASK = None

def display_indices(test, n = 1000, qs = (0.1, 0.3, 0.5, 0.7, 0.9)):
    """The fixed display inputs: among the first n test jets, those nearest
    the quantiles qs of their cone energy (fixed before any output)."""
    e = test['e_cone'][:n]
    return [ int(np.argmin(np.abs(e - np.quantile(e, q)))) for q in qs ]

def fig_displays(test, jewel, outputs, multi, path, qs = (0.1, 0.3, 0.5, 0.7, 0.9)):
    # pylint: disable=import-outside-toplevel,too-many-locals
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pick = display_indices(test, qs = qs)
    ej = jewel['e_cone']
    jpick = [ int(np.argmin(np.abs(ej - np.quantile(ej, q)))) for q in qs ]
    cols = [ ('PYTHIA input', test['canvas'][pick]) ]
    cols += [ (name, img[pick]) for (name, img) in outputs.items() ]
    if multi is not None:
        (label, outs) = multi
        cols += [ (f'{label}\nsample {k + 1} (of 8)', outs[k][pick]) for k in (0, 1) ]
    cols.append(('JEWEL test jet at the same\nenergy quantile (not a truth)',
                 jewel['canvas'][jpick]))
    (fig, axes) = plt.subplots(len(pick), len(cols), figsize = (1.55 * len(cols), 1.75 * len(pick)))
    for (c, (name, img)) in enumerate(cols):
        for r in range(len(pick)):
            ax = axes[r][c]
            w = img[r][OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * _MASK
            ax.imshow(np.log10(w + 0.1), cmap = 'viridis', vmin = -1, vmax = 1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{w.sum():.1f} GeV, lead {w.max():.1f}', fontsize = 5.5)
            if r == 0:
                ax.set_xlabel(name, fontsize = 6, labelpad = 10)
                ax.xaxis.set_label_position('top')
            if c == 0:
                ax.set_ylabel(f'q = {qs[r]}', fontsize = 6)
    fig.suptitle('Fixed held-out PYTHIA test jets (first 1000, nearest the cone-energy'
                 ' quantiles q), log10(E + 0.1)', fontsize = 7)
    fig.tight_layout(h_pad = 0.6)
    fig.savefig(path, dpi = 150)
    plt.close(fig)
    return pick

def fig_displays_slide(test, jewel, outputs, multi, path, qs = (0.1, 0.5, 0.9)):
    """The slides' displays: three of the fixed inputs (same rule), one
    output of each model, two more alpha-DSBM samples, a JEWEL jet at the
    same energy quantile."""
    # pylint: disable=import-outside-toplevel,too-many-locals
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pick  = display_indices(test, qs = qs)
    ej    = jewel['e_cone']
    jpick = [ int(np.argmin(np.abs(ej - np.quantile(ej, q)))) for q in qs ]
    cols  = [ ('PYTHIA input', test['canvas'][pick]) ]
    cols += [ (name.replace('alpha-', '$\\alpha$-'), img[pick]) for (name, img) in outputs.items() ]
    if multi is not None:
        cols += [ (f'sample {k + 1} of 8', multi[1][k][pick]) for k in (0, 1) ]
    cols.append(('JEWEL jet,\nsame E quantile', jewel['canvas'][jpick]))
    (fig, axes) = plt.subplots(len(pick), len(cols), figsize = (0.82 * len(cols), 1.2 * len(pick)))
    for (c, (name, img)) in enumerate(cols):
        for r in range(len(pick)):
            ax = axes[r][c]
            w = img[r][OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * _MASK
            ax.imshow(np.log10(w + 0.1), cmap = 'viridis', vmin = -1, vmax = 1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{w.sum():.1f} GeV', fontsize = 6, pad = 1.5)
            if r == 0:
                ax.set_xlabel(name, fontsize = 6, labelpad = 7)
                ax.xaxis.set_label_position('top')
            if c == 0:
                ax.set_ylabel(f'q = {qs[r]}', fontsize = 6)
    fig.tight_layout(h_pad = 1.0, w_pad = 0.2)
    fig.savefig(path, dpi = 220)
    plt.close(fig)

def fig_curves_slide(curves, val_refs, path):
    """The slides' curves: mean W1 / sigma of the 7 observables and the
    input correlations against training time (validation)."""
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (fig, axes) = plt.subplots(1, 2, figsize = (4.6, 1.75))
    for (label, df) in curves.items():
        for (setting, d) in df.groupby('setting'):
            d = d.sort_values('train_time')
            x = d['train_time'] / 3600
            name = f'{label} ({setting})'
            kw = dict(color = colour(name), marker = '.', ms = 3, label = name,
                      ls = '--' if 'euler' in setting else '-')
            axes[0].plot(x, d[[ f'w1_{o}' for o in te.OBS ]].mean(axis = 1), **kw)
            axes[1].plot(x, d['corr_E'], **kw)
            axes[1].plot(x, d['corr_girth'], **{ **kw, 'label' : None, 'marker' : 'x' })
    for (name, row) in val_refs.items():
        axes[0].axhline(np.mean([ row[f'w1_{o}'] for o in te.OBS ]), color = colour(name),
                        lw = 0.8, ls = ':')
    axes[0].set_yscale('log')
    axes[0].set_title('mean W1/$\\sigma$ to JEWEL val (7 obs.)', fontsize = 6.5)
    axes[1].set_title('input corr.: E (dot), girth (x)', fontsize = 6.5)
    for ax in axes:
        ax.set_xlabel('training, GPU h', fontsize = 6)
        ax.tick_params(labelsize = 5.5)
    axes[0].legend(fontsize = 4.5)
    fig.tight_layout()
    fig.savefig(path, dpi = 220)
    plt.close(fig)

def fig_samples(test, multi, path, pick):
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (label, outs) = multi
    k = len(outs)
    (fig, axes) = plt.subplots(len(pick), k + 1, figsize = (1.3 * (k + 1), 1.5 * len(pick)))
    for (r, i) in enumerate(pick):
        imgs = [ test['canvas'][i] ] + [ outs[s][i] for s in range(k) ]
        for (c, img) in enumerate(imgs):
            ax = axes[r][c]
            w = img[OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] * _MASK
            ax.imshow(np.log10(w + 0.1), cmap = 'viridis', vmin = -1, vmax = 1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{w.sum():.1f} GeV', fontsize = 5.5)
            if r == 0:
                ax.set_xlabel('input' if c == 0 else f'sample {c}', fontsize = 6, labelpad = 8)
                ax.xaxis.set_label_position('top')
    fig.suptitle(f'{label}: 8 samples of each fixed input (sampler seeds 1000-1007);'
                 ' algorithmic variability, not a physical uncertainty', fontsize = 7)
    fig.tight_layout(h_pad = 0.5)
    fig.savefig(path, dpi = 150)
    plt.close(fig)

def fig_curves(curves, val_refs, path):
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qs = [ 'mean', 'E', 'mass', 'girth', 'zlead' ]
    (fig, axes) = plt.subplots(1, len(qs) + 1, figsize = (2.2 * (len(qs) + 1), 2.5))
    for (label, df) in curves.items():
        for (setting, d) in df.groupby('setting'):
            d = d.sort_values('train_time')
            x = d['train_time'] / 3600
            name = f'{label} ({setting})'
            kw = dict(color = colour(name), marker = '.', ms = 3, label = name,
                      ls = '--' if 'euler' in setting else '-')
            for (ax, q) in zip(axes, qs):
                y = d[[ f'w1_{o}' for o in te.OBS ]].mean(axis = 1) if q == 'mean' \
                    else d[f'w1_{q}']
                ax.plot(x, y, **kw)
            axes[-1].plot(x, d['corr_E'], **kw)
            axes[-1].plot(x, d['corr_girth'], **{ **kw, 'label' : None, 'marker' : 'x' })
    for (ax, q) in zip(axes, qs):
        for (name, row) in val_refs.items():
            y = np.mean([ row[f'w1_{o}'] for o in te.OBS ]) if q == 'mean' else row[f'w1_{q}']
            ax.axhline(y, color = colour(name), lw = 0.8, ls = ':')
        ax.set_title('mean W1/$\\sigma$ (7 obs.)' if q == 'mean' else
                     f'W1/$\\sigma$ {LABELS[q]}', fontsize = 7)
        ax.set_xlabel('training, GPU h', fontsize = 6.5)
        ax.tick_params(labelsize = 6)
        ax.set_yscale('log')
    axes[-1].set_title('input corr.: E (dot), girth (x)', fontsize = 7)
    axes[-1].set_xlabel('training, GPU h', fontsize = 6.5)
    axes[-1].tick_params(labelsize = 6)
    axes[0].legend(fontsize = 5)
    fig.suptitle('Validation (PYTHIA val -> JEWEL val), every 10 min; dotted: identity'
                 ' (grey) and JEWEL-train draws (olive)', fontsize = 7)
    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)

# --- cost

def cost_table(info, curves_spec):
    rows = []
    for spec in (curves_spec or '').split(','):
        if not spec:
            continue
        (label, runs) = spec.split('=', 1)
        row = { 'model' : label, 'gpu_h' : 0.0, 'updates' : 0 }
        for run in runs.split('+'):
            path = te.resolve(run)
            if not os.path.exists(os.path.join(path, 'summary.json')):
                continue          # CycleGAN runs: costs from outputs.json
            with open(os.path.join(path, 'summary.json'), encoding = 'utf-8') as f:
                s = json.load(f)
            cfg = te.run_config(path)
            stage = cfg.get('stage', 'train')
            t = s['train_time']
            u = s.get('updates', s.get('step'))
            row['gpu_h'] += t / 3600
            row['updates'] += int(u)
            row[f'{stage}_min'] = t / 60
            row[f'{stage}_updates'] = int(u)
            row[f'{stage}_updates_per_s'] = u / t
            row[f'{stage}_peak_gb'] = s['peak_mem_gb']
            if 'rollout_share' in s:
                row[f'{stage}_rollout_share'] = s['rollout_share']
                row['rollout_nfe_per_update'] = s['rollout_nfe_per_update']
            if 'coupling_frac' in s:
                row['coupling_share'] = s['coupling_frac']
            row['n_params'] = s.get('n_params')
            row['end_to_end_min'] = row.get('end_to_end_min', 0.0) + \
                s.get('end_to_end_time', s.get('end_to_end', 0.0)) / 60
        for (k, v) in info.items():
            if k.startswith(label):
                row[f'latency_ms_{v["setting"]}'] = v['ms_per_jet']
                row[f'nfe_{v["setting"]}'] = v['nfe']
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    # pylint: disable=too-many-locals,too-many-statements
    global _SAMPLES, _MASK       # pylint: disable=global-statement
    cmdargs = parse_cmdargs()
    device  = torch.device('cpu')
    torch.set_num_threads(8)
    os.makedirs(cmdargs.out, exist_ok = True)
    fig = lambda name: os.path.join(cmdargs.out, name)      # pylint: disable=unnecessary-lambda-assignment

    (canv, test, info) = samples(cmdargs)
    _SAMPLES = canv
    _MASK = Geometry(device).mask.numpy()
    models = [ s.split('=', 1)[0] for s in cmdargs.models.split(',') ]
    obs = {}
    for (name, (img, rows)) in canv.items():
        obs[name] = te.jet_observables(img, rows, device)
        print(f'observables: {name}', flush = True)

    pop = population_table(obs, cmdargs.boot)
    pop.to_csv(fig('population.csv'), index = False)
    joint = joint_table(obs, cmdargs.boot)
    joint.to_csv(fig('joint.csv'), index = False)

    ref = refound({ k : v for (k, v) in canv.items() }, cmdargs.procs)
    rf = population_table(ref, cmdargs.boot, key = 'pt', names = REFOUND)
    for (name, r) in ref.items():
        rf.loc[rf['sample'] == name, 'offset_mean'] = float(np.nanmean(r['offset']))
        rf.loc[rf['sample'] == name, 'offset_gt_0p1'] = float(np.nanmean(r['offset'] > 0.1))
        rf.loc[rf['sample'] == name, 'n_jets5_mean'] = float(np.nanmean(r['n_jets5']))
    jt = ref['JEWEL test']
    rf = pd.concat([ rf, pd.DataFrame([{
        'sample' : 'JEWEL test', 'n' : len(jt['pt']),
        'acceptance' : float(np.mean(jt['pt'] >= te.MIN_JET)),
        'offset_mean' : float(np.nanmean(jt['offset'])),
        'offset_gt_0p1' : float(np.nanmean(jt['offset'] > 0.1)),
        'n_jets5_mean' : float(np.nanmean(jt['n_jets5'])) }]) ])
    rf.to_csv(fig('refound.csv'), index = False)

    emds = preference({ k : canv[k][0] for k in [ 'identity', 'random JEWEL' ] + models },
                      canv['identity'][0], cmdargs.procs)
    dep = dependence_table(obs, emds, cmdargs.boot)
    dep.to_csv(fig('dependence.csv'), index = False)
    fid = fidelity_table(obs)
    fid.to_csv(fig('fidelity.csv'), index = False)
    mig = migration_table(obs)
    mig.to_csv(fig('migration.csv'), index = False)
    ver = verdict(pop, joint, dep, fid, models)
    ver.to_csv(fig('verdict.csv'), index = False)

    multi = None
    if cmdargs.multi:
        (var, outs) = variability_table(cmdargs.multi, test, obs['JEWEL test'], device)
        var.to_csv(fig('variability.csv'), index = False)
        multi = (cmdargs.multi.split('=', 1)[0], outs)

    primary = [ m for m in models if 'euler' not in m.lower() ]
    show = [ 'JEWEL test', 'identity', 'random JEWEL' ] + models
    fig_marginals(obs, fig('tr_marginals.png'), show,
                  title = 'Fixed-crop observables, 20k test jets each; shapes after'
                          ' E >= 10 GeV on every sample (E: all outputs)')
    fig_marginals(ref, fig('tr_refound.png'), show, sel_key = 'pt', names = REFOUND,
                  title = 'Leading anti-kT R = 0.4 jet refound in each canvas;'
                          ' shapes after pT >= 10 GeV')
    fig_joint(obs, fig('tr_joint.png'), primary)
    fig_profiles(obs, fig('tr_profiles.png'), primary)
    fig_changes_slide(obs, fig('tr_changes_slide.png'), primary)
    fig_changes(obs, fig('tr_changes.png'), models)
    fig_migration(mig, fig('tr_migration.png'), primary + [ 'random JEWEL' ])
    fig_towers(fig('tr_towers.png'), show)
    jewel = te.load_set('test', 'jewel')
    pick = fig_displays(test, jewel, { m : canv[m][0] for m in primary }, multi,
                        fig('tr_displays.png'))
    if multi is not None:
        fig_samples(test, multi, fig('tr_samples.png'), pick)
    fig_displays_slide(test, jewel, { m : canv[m][0] for m in primary }, multi,
                       fig('tr_displays_slide.png'))

    curves = {}
    if cmdargs.curves:
        for spec in cmdargs.curves.split(','):
            (label, runs) = spec.split('=', 1)
            paths = [ os.path.join(te.resolve(r), 'evals', 'translation_val.csv')
                      for r in runs.split('+') ]
            for p in paths:
                if not os.path.exists(p):
                    print(f'no validation curve {p}', flush = True)
            found = [ pd.read_csv(p) for p in paths if os.path.exists(p) ]
            if found:
                curves[label] = pd.concat(found)
        vp = te.load_set('val', 'pythia')
        vj = te.load_set('val', 'jewel')
        o_vj = te.jet_observables(vj['canvas'], vj['row'], device)
        pool = np.load(fc.cache_path('tr_jewel'), mmap_mode = 'r')
        meta = te.train_meta('jewel')
        n = len(vp['canvas'])
        val_refs = {
            'identity' : te.population(te.jet_observables(vp['canvas'], vp['row'], device), o_vj),
            'random JEWEL' : te.population(te.jet_observables(
                np.asarray(pool[:n]).astype(np.float32), meta['row'][:n], device), o_vj) }
        fig_curves(curves, val_refs, fig('tr_curves.png'))
        fig_curves_slide(curves, val_refs, fig('tr_curves_slide.png'))
        pd.concat([ d.assign(model = k) for (k, d) in curves.items() ]).to_csv(
            fig('curves.csv'), index = False)
        pd.DataFrame([ { 'sample' : k, **v } for (k, v) in val_refs.items() ]).to_csv(
            fig('curves_refs.csv'), index = False)
    cost_table(info, cmdargs.curves).to_csv(fig('cost.csv'), index = False)
    # every sample's per-jet observables, fixed crop and refound, for the
    # deck figures (translation_deck_figs.py), so they show these numbers
    store = os.path.join(fc.translation_root(), 'report')
    os.makedirs(store, exist_ok = True)
    np.savez_compressed(
        os.path.join(store, f'{os.path.basename(os.path.normpath(cmdargs.out))}.npz'),
        **{ f'crop|{name}|{q}' : np.asarray(v) for (name, o) in obs.items()
            for (q, v) in o.items() },
        **{ f'refound|{name}|{q}' : np.asarray(v) for (name, o) in ref.items()
            for (q, v) in o.items() })
    with open(fig('outputs.json'), 'w', encoding = 'utf-8') as f:
        json.dump(info, f, indent = 4)

    with pd.option_context('display.width', 250, 'display.max_columns', 80):
        print(pop[[ 'sample', 'acceptance' ] + [ f'w1_{q}' for q in te.OBS ]
                  + [ 'n_moves', 'n_consistent' ]].round(3).to_string(index = False))
        print(joint.round(3).to_string(index = False))
        print(dep[[ 'sample' ] + [ f'corr_{q}' for q in te.DEP ]
                  + [ 'preference_rate', 'dependence_beyond_random' ]].round(3)
              .to_string(index = False))
        print(fid.round(3).to_string(index = False))
        print(ver.to_string(index = False))
    print(f'wrote {cmdargs.out}')

if __name__ == '__main__':
    main()
