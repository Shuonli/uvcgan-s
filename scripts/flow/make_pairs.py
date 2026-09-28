#!/usr/bin/env python
"""True components of the training mixtures, for the paired arms of the
subtraction benchmark (FLOW_NOTES.md, "Consolidated benchmark").

Every training mixture M (train/embed.h5; the same flat cache the unpaired
arm trains on, `train_embed.npy`) is a PYTHIA event S added tower by tower
to a HIJING event. Its PYTHIA parent is the training signal event with the
same (file, event) key (`train/signal.h5`, as eval_val_truth.py finds the
signal of a val mixture), and its background is B = M - S. Written to
OUTDIR/sphenix/flow/cache/train_embed_pairs.npy, (N, 2, 24, 64) float16:
[M, S], row i for mixture i of train_embed.npy. The flows form
B = max(M - S, 0).

Checks:
- every mixture has exactly one signal parent;
- M >= S up to rounding;
- B from the float16 caches against B from the float32 h5 files, for a
  random sample of events.

The keys of the training mixtures are disjoint from those of the val
mixtures (checked here too).

    make_pairs.py [--check 2000]
"""

import argparse
import json
import os

import h5py
import numpy as np

import fm_common as fc

ev = fc.ev

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Training mixture pairs')
    parser.add_argument('--check', type = int, default = 2000)
    parser.add_argument('--chunk', type = int, default = 20000)
    parser.add_argument('--seed', type = int, default = 0)
    return parser.parse_args()

def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    root    = fc.data_root()
    embed_keys  = ev.read_keys(os.path.join(root, 'train', ev.EMBED_IDX))
    val_keys    = ev.read_keys(os.path.join(root, 'val', ev.EMBED_IDX))
    pythia_keys = ev.read_keys(os.path.join(root, 'train', ev.PYTHIA_IDX))

    order = np.argsort(pythia_keys)
    pos   = np.searchsorted(pythia_keys, embed_keys, sorter = order)
    match = order[np.minimum(pos, len(order) - 1)]
    assert (pythia_keys[match] == embed_keys).all(), 'mixture without parent'
    assert len(np.unique(match)) == len(match), 'shared signal parents'
    overlap = len(np.intersect1d(embed_keys, val_keys))

    embed  = np.load(fc.cache_path('embed'), mmap_mode = 'r')
    signal = np.load(fc.cache_path('signal'), mmap_mode = 'r')
    assert len(embed) == len(embed_keys)

    out   = np.lib.format.open_memmap(
        os.path.join(os.path.dirname(fc.cache_path('embed')),
                     'train_embed_pairs.npy'),
        mode = 'w+', dtype = np.float16, shape = (len(embed), 2, *fc.SHAPE))
    worst = 0.0
    for start in range(0, len(embed), cmdargs.chunk):
        sl  = slice(start, start + cmdargs.chunk)
        idx = match[sl]
        srt = np.argsort(idx)
        s   = np.empty((len(idx), *fc.SHAPE), np.float16)
        s[srt] = signal[idx[srt]]
        m   = embed[sl]
        worst = max(worst, float((s.astype(np.float32)
                                  - m.astype(np.float32)).max()))
        out[sl, 0] = m
        out[sl, 1] = s
        print(f'{start + len(idx)} of {len(embed)}', flush = True)
    out.flush()

    # B from the float16 caches against B from the float32 h5 files
    rng  = np.random.default_rng(cmdargs.seed)
    pick = np.sort(rng.choice(len(embed), cmdargs.check, replace = False))
    with h5py.File(os.path.join(root, 'train', 'embed.h5'), 'r') as f:
        m32 = f['data'][pick].astype(np.float32)
    sig_idx = match[pick]
    srt = np.argsort(sig_idx)
    s32 = np.empty_like(m32)
    with h5py.File(os.path.join(root, 'train', 'signal.h5'), 'r') as f:
        s32[srt] = f['data'][sig_idx[srt]]
    b32 = m32 - s32
    p16 = np.asarray(out[pick]).astype(np.float32)
    b16 = np.maximum(p16[:, 0] - p16[:, 1], 0)

    report = {
        'mixtures' : int(len(embed)),
        'train_val_key_overlap' : overlap,
        'max_signal_minus_mixture_float16' : worst,
        'check_events' : int(cmdargs.check),
        'float32_min_B' : float(b32.min()),
        'B_float16_vs_float32_max_abs' : float(np.abs(b16 - b32).max()),
        'B_float16_vs_float32_mean_abs' : float(np.abs(b16 - b32).mean()),
        'B_float16_vs_float32_max_abs_where_S_gt_5GeV' : float(
            np.abs(b16 - b32)[s32 > 5].max()),
    }
    with open(os.path.join(os.path.dirname(fc.cache_path('embed')),
                           'train_embed_pairs.json'), 'w',
              encoding = 'utf-8') as f:
        json.dump(report, f, indent = 4)
    print(json.dumps(report, indent = 2))

if __name__ == '__main__':
    main()
