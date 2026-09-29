#!/usr/bin/env python
"""Clean PYTHIA and JEWEL jets for the unpaired PYTHIA -> JEWEL translation
pilot (FLOW_NOTES.md, "PYTHIA -> JEWEL translation pilot"). A separate data
set: nothing of the closure test (closure_data.py) or of the subtraction
study is read for training or overwritten.

Parents. A parent is one generated event, identified by its generator file
and event number, key = file x 10^6 + event:
- PYTHIA: every event of train/signal.h5 (its index file names the file and
  event of each sample; read from the flat cache, which holds the same
  samples in the same order);
- JEWEL: every histogram of the 870 ROOT files of Zenodo record 17594612
  (DATA/sphenix/jewel_jet30/jewel_jet30/jewel_jet30_file<F>.root,
  h_eta_phi_cent0_file<F>_evt<E>), the clean JEWEL images that
  eval_val_truth.py reads as subtraction truth. They are rounded to float16
  on reading, the precision in which the PYTHIA h5 stores its images, and
  copied once into a flat cache (jewel_events.npy, jewel_keys.npy).
Keys are checked unique, and images are deduplicated by content (a hash of
the float16 image; the first parent of a duplicate group is kept).

Jets. One jet per parent, the leading one, with the closure test's
selection and crop (closure_data.select / windows, the same code for both
domains): the axis is the tower whose R = 0.4 cone holds the most energy
(phi periodic, eta not); kept if the 9 x 9 window around it lies inside the
acceptance (axis rows 4-19, |eta| < 0.73) and the cone holds >= --min-jet
GeV. The jet image is the 53 towers of the R = 0.4 cone, in GeV of tower
E_T as the images hold them (towers clipped at 0; neither sample has
negative towers), on a 16 x 16 canvas (window at rows and columns 4-12, axis
at 8, 8), the rest zero. No per-jet energy normalisation, no reweighting of
either spectrum. What the crop leaves out is recorded per jet: the event's
energy outside the cone (e_event - e_cone) and the part of it in the ring
0.4 < dR <= 0.8 around the axis (inside the acceptance only).

Splits, by parent (one jet per parent, so a parent and all it contributes
sit in one split), from one fixed permutation per domain of the selected,
deduplicated parents:
- PYTHIA: train 200k, val 10k, test 20k;
- JEWEL: train 200k, val 10k, test 20k, and ref 20k, a second held-out
  sample used only as the JEWEL-vs-JEWEL finite-sample reference.
The 20k JEWEL parents of the subtraction study's JEWEL evaluation
(OUTDIR/sphenix/val_truth/pairs_jewel_n20000_seed0.npz) are kept out of
every split, so that evaluation stays untouched by JEWEL training.

Normalisation: psi = log(E + 0.1), standardised with one mean and deviation
over the training canvases of both pools together (as the closure's),
written to the translation area, not to norm_closure.json.

Written to OUTDIR/sphenix/flow/translation/:

    cache/train_pythia.npy, cache/train_jewel.npy   (N, 16, 16) float16, GeV
    cache/{val,test}_{pythia,jewel}.npz, cache/ref_jewel.npz
            canvas (float32), key, parent_file, parent_event, row, col,
            e_cone, e_event, e_ring, source_index
    cache/meta_train.npz    the same per-jet columns of the training pools
    cache/jewel_events.npy, cache/jewel_keys.npy   the JEWEL images, float16
    norm.json               {'jet': (mean, sd), '_bias': 0.1}
    manifest.json           counts, selection, crop, checks, provenance

and the parent split lists to docs/flow/translation/splits/{pythia,jewel}.csv.gz
(file, event, split, row, col, e_cone).

    translation_data.py [--n-train 200000] [--n-val 10000] [--n-test 20000]
                        [--n-ref 20000] [--min-jet 10] [--seed 0] [--procs 32]
"""

import argparse
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import time

import numpy as np
import pandas as pd
import torch

import fm_common as fc
from closure_data import CANVAS, HALF, OFFSET, cone_mask, select, windows

ev = fc.ev

JEWEL_DIR   = 'sphenix/jewel_jet30/jewel_jet30'
OLD_JEWEL   = 'pairs_jewel_n20000_seed0.npz'
RING        = (0.4, 0.8)
SPLITS      = { 'pythia' : ('train', 'val', 'test'),
                'jewel'  : ('train', 'val', 'test', 'ref') }

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'PYTHIA -> JEWEL jets')
    parser.add_argument('--n-train', type = int, default = 200000)
    parser.add_argument('--n-val', type = int, default = 10000)
    parser.add_argument('--n-test', type = int, default = 20000)
    parser.add_argument('--n-ref', type = int, default = 20000)
    parser.add_argument('--min-jet', type = float, default = 10.0)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--chunk', type = int, default = 20000)
    parser.add_argument('--procs', type = int, default = 32)
    parser.add_argument('--docs', default = 'docs/flow/translation')
    return parser.parse_args()

def root():
    return fc.translation_root()

def cache_dir():
    path = os.path.join(root(), 'cache')
    os.makedirs(path, exist_ok = True)
    return path

# --- the JEWEL images, copied once into a flat float16 cache

_CPUS = None

def use_all_cpus():
    """Pool initializer: the parent's CPU mask. Under SLURM the forked
    workers came up pinned to one core (all 30 of them), which made the read
    CPU-starved."""
    if _CPUS:
        os.sched_setaffinity(0, _CPUS)

def read_jewel_file(path):
    # pylint: disable=import-outside-toplevel
    import uproot
    use_all_cpus()
    fileno = int(re.search(r'_file(\d+)\.root$', path).group(1))
    events, images = [], []
    # one sequential read, then parsed from memory: on NFS the default
    # handler re-reads ~0.7 GB per 1.5 MB file, and a memory map faults page
    # by page over the network
    with open(path, 'rb') as raw:
        buffer = io.BytesIO(raw.read())
    with uproot.open(buffer) as f:
        for name in f.keys():
            m = re.match(r'h_eta_phi_cent0_file(\d+)_evt(\d+);\d+$', name)
            assert m and int(m.group(1)) == fileno, (path, name)
            events.append(int(m.group(2)))
            images.append(f[name].values())
    images = np.stack(images)
    assert images.shape[1:] == fc.SHAPE, images.shape
    order = np.argsort(events)
    x = images[order]
    rel = np.abs(x.astype(np.float16).astype(np.float64) - x) \
        / np.maximum(np.abs(x), 1e-30)
    return (fileno, np.asarray(events)[order], x.astype(np.float16),
            float(rel[x != 0].max()) if (x != 0).any() else 0.0,
            int((x < 0).sum()))

def jewel_cache(procs):
    """(keys, images) of every JEWEL parent: jewel_keys.npy, jewel_events.npy."""
    (kpath, ipath) = (os.path.join(cache_dir(), 'jewel_keys.npy'),
                      os.path.join(cache_dir(), 'jewel_events.npy'))
    info_path = os.path.join(cache_dir(), 'jewel_read.json')
    if os.path.exists(kpath) and os.path.exists(ipath):
        with open(info_path, encoding = 'utf-8') as f:
            info = json.load(f)
        return (np.load(kpath), np.load(ipath, mmap_mode = 'r'), info)

    files = sorted(glob.glob(os.path.join(
        os.environ.get('UVCGAN_S_DATA', 'data'), JEWEL_DIR,
        'jewel_jet30_file*.root')),
        key = lambda p: int(re.search(r'_file(\d+)\.root$', p).group(1)))
    t0 = time.time()
    global _CPUS           # pylint: disable=global-statement
    _CPUS = os.sched_getaffinity(0)
    with mp.Pool(procs, initializer = use_all_cpus) as pool:
        parts = pool.map(read_jewel_file, files, chunksize = 4)
    keys = np.concatenate([ fileno * 1_000_000 + evs
                            for (fileno, evs, _, _, _) in parts ])
    imgs = np.concatenate([ x for (_, _, x, _, _) in parts ])
    info = { 'files' : len(files), 'histograms' : int(len(keys)),
             'read_seconds' : time.time() - t0,
             'float16_max_rel_rounding' : max(p[3] for p in parts),
             'negative_towers' : int(sum(p[4] for p in parts)) }
    np.save(f'{ipath}.tmp.npy', imgs)
    os.replace(f'{ipath}.tmp.npy', ipath)
    np.save(kpath, keys)
    with open(info_path, 'w', encoding = 'utf-8') as f:
        json.dump(info, f, indent = 4)
    print(f"JEWEL: {info['histograms']} images from {info['files']} files in"
          f" {info['read_seconds']:.0f} s", flush = True)
    return (keys, np.load(ipath, mmap_mode = 'r'), info)

def pythia_keys():
    return ev.read_keys(os.path.join(
        fc.data_root(), 'train', ev.PYTHIA_IDX))

# --- one pass over every parent: content hash, selection, what the crop drops

def ring_kernel():
    ni = int(RING[1] / ev.DETA)
    nj = int(RING[1] / ev.DPHI)
    deta = np.arange(-ni, ni + 1)[:, None] * ev.DETA
    dphi = np.arange(-nj, nj + 1)[None, :] * ev.DPHI
    dr2  = deta**2 + dphi**2
    return torch.tensor((dr2 <= RING[1]**2 + 1e-9) & (dr2 > RING[0]**2 + 1e-9),
                        dtype = torch.float32)

def scan(images, chunk, device, min_jet):
    """Per parent: hash, the selection (closure_data.select), the axis, and
    the energies of the cone, the event and the ring around the axis."""
    # pylint: disable=too-many-locals
    kernel = ev.cone_kernel(ev.R_JET).to(device)
    ring   = ring_kernel().to(device)
    mask   = torch.from_numpy(cone_mask()).to(device)
    n      = len(images)
    out = { 'hash' : np.empty(n, dtype = 'S16'), 'kept' : np.zeros(n, bool),
            'row' : np.full(n, -1, np.int16), 'col' : np.full(n, -1, np.int16),
            'axis_row' : np.zeros(n, np.int16), 'e_lead' : np.zeros(n),
            'e_cone' : np.zeros(n), 'e_event' : np.zeros(n),
            'e_ring' : np.zeros(n), 'empty' : np.zeros(n, bool) }
    for start in range(0, n, chunk):
        x16 = np.ascontiguousarray(images[start:start + chunk])
        for (k, img) in enumerate(x16):
            out['hash'][start + k] = hashlib.blake2b(
                img.tobytes(), digest_size = 16).digest()
        img = torch.from_numpy(x16.astype(np.float32)).to(device)
        # the leading cone of every parent (as select() finds it)
        sums = ev.cone_sums(img, kernel)
        flat = sums.flatten(1).argmax(dim = 1)
        (row, col) = (flat // sums.shape[2], flat % sums.shape[2])
        m    = torch.arange(len(img), device = device)
        sl   = slice(start, start + len(img))
        out['axis_row'][sl] = row.cpu().numpy()
        out['e_lead'][sl]   = sums[m, row, col].double().cpu().numpy()
        out['e_event'][sl]  = img.double().sum((1, 2)).cpu().numpy()
        out['e_ring'][sl]   = ev.cone_sums(img, ring)[m, row, col] \
            .double().cpu().numpy()
        out['empty'][sl]    = (img.flatten(1).amax(1) <= 0).cpu().numpy()
        (idx, rows, cols, canvas) = select(img, kernel, mask, min_jet)
        idx = idx.cpu().numpy() + start
        out['kept'][idx] = True
        out['row'][idx]  = rows.cpu().numpy()
        out['col'][idx]  = cols.cpu().numpy()
        out['e_cone'][idx] = canvas.double().sum((1, 2)).cpu().numpy()
        print(f'  scanned {min(start + chunk, n)} / {n}', flush = True)
    # select() and this scan must agree on the axis of every kept jet
    kept = out['kept']
    assert (out['row'][kept] == out['axis_row'][kept]).all()
    assert np.allclose(out['e_cone'][kept], out['e_lead'][kept], rtol = 1e-4,
                       atol = 1e-3)
    return out

def first_of_duplicates(hashes):
    """Mask of parents that are the first of their content group."""
    (_, first) = np.unique(hashes, return_index = True)
    keep = np.zeros(len(hashes), bool)
    keep[first] = True
    return keep

# --- canvases of the chosen parents

def canvases(images, index, rows, cols, device, chunk):
    """windows() of parents `index` (any order), float32, on the CPU; rows,
    cols: the axes of every parent."""
    mask  = torch.from_numpy(cone_mask()).to(device)
    order = np.argsort(index)
    out   = np.zeros((len(index), CANVAS, CANVAS), np.float32)
    for start in range(0, len(order), chunk):
        sel = order[start:start + chunk]
        img = torch.from_numpy(np.asarray(images[index[sel]]).astype(np.float32)) \
            .to(device)
        par = index[sel]
        can = windows(img, torch.as_tensor(rows[par].astype(np.int64), device = device),
                      torch.as_tensor(cols[par].astype(np.int64), device = device),
                      mask)
        out[sel] = can.cpu().numpy()
    return out

def summary(name, keys, scanned, eligible, chosen, info):
    """The counts and distributions of one domain, for the manifest."""
    (kept, dedup) = (scanned['kept'], scanned['dedup'])
    inside = (scanned['axis_row'] >= HALF) \
        & (scanned['axis_row'] < fc.SHAPE[0] - HALF)
    e = scanned['e_cone'][kept]
    frac = scanned['e_cone'][kept] / scanned['e_event'][kept]
    return {
        'domain' : name,
        'parents' : int(len(keys)),
        'unique_keys' : int(len(np.unique(keys))),
        'files' : int(len(np.unique(keys // 1_000_000))),
        'empty_images' : int(scanned['empty'].sum()),
        'content_duplicates_dropped' : int((~dedup).sum()),
        'leading_axis_outside_acceptance' : int((~inside).sum()),
        'leading_cone_below_min_jet' : int((inside & ~kept).sum()),
        'selected' : int(kept.sum()),
        'selected_fraction' : float(kept.mean()),
        'selected_and_unique' : int((kept & dedup).sum()),
        'excluded_old_evaluation' : int((kept & dedup).sum() - eligible.sum()),
        'eligible' : int(eligible.sum()),
        'used' : { k : int(len(v)) for (k, v) in chosen.items() },
        'unused_eligible' : int(eligible.sum() - sum(len(v) for v in chosen.values())),
        'e_cone_gev_quantiles_5_25_50_75_95' : [
            float(v) for v in np.quantile(e, [ 0.05, 0.25, 0.5, 0.75, 0.95 ]) ],
        'e_event_gev_median' : float(np.median(scanned['e_event'][kept])),
        'cone_share_of_event_energy_median' : float(np.median(frac)),
        'cone_share_of_event_energy_mean' : float(np.mean(frac)),
        'ring_0p4_0p8_gev_median' : float(np.median(scanned['e_ring'][kept])),
        'ring_0p4_0p8_gev_mean' : float(np.mean(scanned['e_ring'][kept])),
        'ring_share_of_cone_mean' : float(np.mean(scanned['e_ring'][kept] / e)),
        **({ 'read' : info } if info else {}),
    }

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    rng     = np.random.default_rng(cmdargs.seed)
    t_start = time.time()

    sources = {}
    print('PYTHIA: keys and the flat signal cache', flush = True)
    sources['pythia'] = (pythia_keys(), np.load(fc.cache_path('signal'),
                                                mmap_mode = 'r'), None)
    assert len(sources['pythia'][0]) == len(sources['pythia'][1])
    print('JEWEL: ROOT images', flush = True)
    sources['jewel'] = jewel_cache(cmdargs.procs)

    old = np.load(os.path.join(os.environ.get('UVCGAN_S_OUTDIR', 'outdir'),
                               'sphenix', 'val_truth', OLD_JEWEL))['keys']

    sizes = { 'train' : cmdargs.n_train, 'val' : cmdargs.n_val,
              'test' : cmdargs.n_test, 'ref' : cmdargs.n_ref }
    manifest = { 'domains' : {}, 'splits' : {}, 'checks' : {} }
    columns  = {}
    hashes   = {}
    for (name, (keys, images, info)) in sources.items():
        assert len(np.unique(keys)) == len(keys), f'{name}: duplicate keys'
        print(f'{name}: scanning {len(keys)} parents', flush = True)
        sc = scan(images, cmdargs.chunk, device, cmdargs.min_jet)
        sc['dedup'] = first_of_duplicates(sc['hash'])
        hashes[name] = sc['hash']
        eligible = sc['kept'] & sc['dedup']
        if name == 'jewel':
            eligible &= ~np.isin(keys, old)
        pool  = np.nonzero(eligible)[0]
        order = rng.permutation(len(pool))
        need  = sum(sizes[s] for s in SPLITS[name])
        assert len(pool) >= need, (name, len(pool), need)
        chosen, at = {}, 0
        for s in SPLITS[name]:
            chosen[s] = pool[order[at:at + sizes[s]]]
            at += sizes[s]
        manifest['domains'][name] = summary(name, keys, sc, eligible, chosen,
                                            info)
        columns[name] = (keys, images, sc, chosen)

    # disjointness: within a domain by construction, checked; across domains
    for (name, (keys, _, _, chosen)) in columns.items():
        allk = np.concatenate([ keys[v] for v in chosen.values() ])
        assert len(np.unique(allk)) == len(allk), name
    common = np.intersect1d(np.unique(hashes['pythia']), np.unique(hashes['jewel']))
    manifest['checks']['identical_images_across_domains'] = int(len(common))
    manifest['checks']['jewel_old_evaluation_parents_in_splits'] = int(sum(
        np.isin(columns['jewel'][0][v], old).sum()
        for v in columns['jewel'][3].values()))

    # canvases, caches, split lists
    os.makedirs(os.path.join(cmdargs.docs, 'splits'), exist_ok = True)
    train_canvas = {}
    for (name, (keys, images, sc, chosen)) in columns.items():
        rows = []
        for (s, idx) in chosen.items():
            can = canvases(images, idx, sc['row'], sc['col'], device, cmdargs.chunk)
            e_can = can.sum((1, 2))
            assert np.allclose(e_can, sc['e_cone'][idx], rtol = 1e-4, atol = 1e-3)
            # (np.savez reserves the keyword 'file')
            cols = { 'key' : keys[idx], 'parent_file' : keys[idx] // 1_000_000,
                     'parent_event' : keys[idx] % 1_000_000, 'row' : sc['row'][idx],
                     'col' : sc['col'][idx], 'e_cone' : sc['e_cone'][idx],
                     'e_event' : sc['e_event'][idx], 'e_ring' : sc['e_ring'][idx],
                     'source_index' : idx }
            if s == 'train':
                train_canvas[name] = can
                np.save(fc.cache_path(f'tr_{name}'), can.astype(np.float16))
                np.savez(os.path.join(cache_dir(), f'meta_train_{name}.npz'), **cols)
            else:
                np.savez(os.path.join(cache_dir(), f'{s}_{name}.npz'), canvas = can,
                         **cols)
            rows.append(pd.DataFrame({ 'file' : cols['parent_file'],
                                       'event' : cols['parent_event'],
                                       'split' : s, 'row' : cols['row'],
                                       'col' : cols['col'],
                                       'e_cone' : np.round(cols['e_cone'], 4) }))
            manifest['splits'][f'{name}_{s}'] = {
                'n' : int(len(idx)),
                'e_cone_median' : float(np.median(cols['e_cone'])),
                'files' : int(len(np.unique(cols['parent_file']))) }
            print(f'{name} {s}: {len(idx)} jets', flush = True)
        pd.concat(rows).to_csv(os.path.join(cmdargs.docs, 'splits', f'{name}.csv.gz'),
                               index = False)

    # the float16 training caches reproduce the canvases
    for name in columns:
        c16 = np.load(fc.cache_path(f'tr_{name}')).astype(np.float32)
        r = (c16.sum((1, 2)) / train_canvas[name].sum((1, 2)))
        manifest['checks'][f'train_{name}_float16_energy_ratio'] = [
            float(r.min()), float(r.max()) ]

    # one normalisation over both training pools (as closure_data.py)
    both = np.concatenate([ np.load(fc.cache_path(f'tr_{n}')) for n in columns ]) \
        .astype(np.float64)
    psi  = np.log(both + fc.BIAS)
    norm = { 'jet' : (float(psi.mean()), float(psi.std())), '_bias' : fc.BIAS }
    with open(fc.translation_norm_path(), 'w', encoding = 'utf-8') as f:
        json.dump(norm, f, indent = 4)
    manifest['norm'] = { **norm, 'fitted_on' : 'train_pythia + train_jewel canvases,'
                         ' every pixel of the 16 x 16 canvas (the closure procedure)' }
    for name in columns:
        z = (np.log(np.load(fc.cache_path(f'tr_{name}'))[:20000].astype(np.float64)
                    + fc.BIAS) - norm['jet'][0]) / norm['jet'][1]
        manifest['checks'][f'train_{name}_z_mean_sd_second_moment'] = [
            float(z.mean()), float(z.std()), float((z**2).mean()) ]

    manifest.update({
        'seed' : cmdargs.seed, 'min_jet_gev' : cmdargs.min_jet,
        'selection' : f'leading R = 0.4 cone (axis = tower of largest cone sum,'
                      f' phi periodic) >= {cmdargs.min_jet} GeV, axis rows'
                      f' {HALF}-{fc.SHAPE[0] - HALF - 1} (9 x 9 window inside'
                      ' the acceptance); one jet per parent',
        'canvas' : f'{CANVAS} x {CANVAS}, 9 x 9 window at {OFFSET}..{OFFSET + 8},'
                   ' R = 0.4 cone towers only (53), tower E_T in GeV, clipped at 0',
        'crop_excludes' : [
            'all energy outside the R = 0.4 cone around the leading axis:'
            ' wide-angle radiation, out-of-cone medium-induced radiation and'
            ' recoil energy, the recoiling jet, the underlying event',
            'towers of the 9 x 9 window outside the cone (dR > 0.4)',
            'everything beyond |eta| < 1.1 (the image), and particles the'
            ' images do not hold' ],
        'ring' : f'{RING[0]} < dR <= {RING[1]} around the axis, inside |eta| < 1.1',
        'splits_by' : 'parent key = file x 10^6 + event, one jet per parent;'
                      ' one fixed permutation per domain',
        'excluded' : f'the {len(old)} JEWEL parents of {OLD_JEWEL} (the'
                     ' subtraction study\'s JEWEL evaluation)',
        'provenance' : {
            'pythia' : 'train/signal.h5 = type11_run19_jet30_pythia_noNoise_'
                       'allCentrality (sPHENIX-style production naming: sample'
                       ' type 11, "jet30", run 19, no noise); generator version,'
                       ' tune, the definition of the jet30 requirement and the'
                       ' detector simulation are not recorded with the data',
            'jewel' : 'Zenodo record 17594612, jewel_jet30.tar.gz: one ROOT TH2D'
                      ' per event (24 x 64, eta -1.1..1.1, phi 0..2 pi), no'
                      ' generator record, seeds, version or medium parameters;'
                      ' the mixed test sample is named'
                      ' type4_hijing_plus_jewel_jet30_40_50 (meaning of 40_50 not'
                      ' documented); recoil treatment unknown',
            'common' : 'both samples share the smallest tower value (1.1e-4 GeV)'
                       ' and the low quantiles of the tower spectrum, which'
                       ' suggests the same tower-level pipeline; not documented',
            'differences_can_include' : [ 'medium effects (JEWEL)',
                'generator differences (JEWEL runs PYTHIA 6 for the hard'
                ' process; the PYTHIA sample is presumably PYTHIA 8; tunes,'
                ' underlying event)',
                'generator-level jet requirements of each sample',
                'the common >= 10 GeV leading-cone selection acting on different'
                ' spectra' ] },
        'build_seconds' : time.time() - t_start,
    })
    with open(os.path.join(root(), 'manifest.json'), 'w', encoding = 'utf-8') as f:
        json.dump(manifest, f, indent = 4)
    with open(os.path.join(cmdargs.docs, 'manifest.json'), 'w', encoding = 'utf-8') as f:
        json.dump(manifest, f, indent = 4)
    print(json.dumps(manifest, indent = 2))

if __name__ == '__main__':
    main()
