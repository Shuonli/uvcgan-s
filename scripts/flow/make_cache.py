#!/usr/bin/env python
"""Copy the training domains into flat float16 .npy files.

The h5 files hold one lzf-compressed chunk per event, which caps reads at
~5k events/s even for contiguous slices (50-150 events/s at random) on this
file system -- far too slow for a flow model drawing thousands of events per
second. The copy is lossless (the h5 data are float16) and made once, by
several processes reading contiguous ranges:

    make_cache.py [--domains background,signal,embed] [--procs 16]

Output: OUTDIR/sphenix/flow/cache/train_{domain}.npy, (N, 24, 64) float16.
"""

import argparse
import multiprocessing as mp
import os
import time

import h5py
import numpy as np

import fm_common as fc

CHUNK = 20000

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Flatten the training h5')
    parser.add_argument('--domains', default = 'background,signal,embed')
    parser.add_argument('--procs', type = int, default = 16)
    return parser.parse_args()

def copy_range(job):
    (src, dst, start, stop) = job

    with h5py.File(src, 'r') as f:
        x = f['data'][start:stop]

    out = np.load(dst, mmap_mode = 'r+')
    out[start:stop] = x
    out.flush()

    return stop - start

def main():
    cmdargs = parse_cmdargs()

    for domain in cmdargs.domains.split(','):
        src = os.path.join(fc.data_root(), 'train', f'{domain}.h5')
        dst = fc.cache_path(domain)

        if os.path.exists(dst):
            print(f'{dst} exists')
            continue

        with h5py.File(src, 'r') as f:
            (shape, dtype) = (f['data'].shape, f['data'].dtype)

        tmp = f'{dst}.tmp.npy'
        os.makedirs(os.path.dirname(dst), exist_ok = True)
        np.lib.format.open_memmap(tmp, mode = 'w+', dtype = dtype,
                                  shape = shape).flush()

        jobs = [
            (src, tmp, a, min(a + CHUNK, shape[0]))
                for a in range(0, shape[0], CHUNK)
        ]

        t0 = time.time()
        with mp.Pool(cmdargs.procs) as pool:
            done = sum(pool.imap_unordered(copy_range, jobs))
        dt = time.time() - t0

        # check a few events against the source
        out = np.load(tmp, mmap_mode = 'r')
        with h5py.File(src, 'r') as f:
            for i in np.random.default_rng(0).integers(0, shape[0], 20):
                assert np.array_equal(out[i], f['data'][i]), (domain, i)

        os.replace(tmp, dst)
        print(f'{domain}: {done} events {shape} {dtype} in {dt:.0f} s'
              f' ({done / dt:.0f} events/s) -> {dst}', flush = True)

if __name__ == '__main__':
    main()
