#!/usr/bin/env python
"""Frozen few-pair teacher for the guided coupling (FLOW_NOTES.md,
"Teacher-guided coupling").

The teacher is a paired-only jet -> jet flow trained on the first 2000
source jets and their T(J) (`closure_pk1_s0`), at its validation-selected
checkpoint, EMA network. It is frozen. Its endpoints F_teacher(x) come from
the accurate solve (32 midpoint evaluations), in GeV and clipped at 0, as
every closure output is. They are cached for all training source jets A
(`teacher_<run>_train.npy`, float16, the order of train_closure_src.npy)
and for the validation pairs (`teacher_<run>_val.npy`, float32), with the
cost in `teacher_<run>.json`.

    closure_teacher.py [--run closure_pk1_s0] [--nfe 32]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import fm_common as fc
from closure_eval import load_pairs, selected_step, checkpoints, ckpt_step

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Cache a frozen teacher')
    parser.add_argument('--run', default = 'closure_pk1_s0')
    parser.add_argument('--nfe', type = int, default = 32)
    parser.add_argument('--solver', default = 'midpoint')
    parser.add_argument('--batch', type = int, default = 4000)
    return parser.parse_args()

def teacher(run, nfe, solver, device):
    path = os.path.join(fc.out_root(), run)
    step = selected_step(path, f'_{nfe}{solver}')
    ckpt = [ c for c in checkpoints(path) if ckpt_step(c) == step ][0]
    (method, net, _, _) = fc.load_run(path, ckpt, device, 'ema')
    for p in net.parameters():
        p.requires_grad_(False)
    return (fc.JetMapper(method, net, nfe, solver), step)

@torch.no_grad()
def predict(mapper, src, batch, device):
    out = np.empty(src.shape, dtype = np.float32)
    for start in range(0, len(src), batch):
        j = torch.as_tensor(np.asarray(src[start:start + batch]),
                            device = device).float()
        out[start:start + batch] = mapper(j).cpu().numpy()
    return out

def cache_paths(run):
    base = os.path.join(os.path.dirname(fc.cache_path('signal')),
                        f'teacher_{run}')
    return { 'train' : f'{base}_train.npy', 'val' : f'{base}_val.npy',
             'meta' : f'{base}.json' }

def main():
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    (mapper, step) = teacher(cmdargs.run, cmdargs.nfe, cmdargs.solver, device)
    paths   = cache_paths(cmdargs.run)

    src = np.load(fc.cache_path('closure_src'), mmap_mode = 'r')
    torch.cuda.synchronize()
    t0  = time.perf_counter()
    out = predict(mapper, src, cmdargs.batch, device)
    torch.cuda.synchronize()
    dt  = time.perf_counter() - t0
    np.save(paths['train'], out.astype(np.float16))

    val = load_pairs('val')
    np.save(paths['val'], predict(mapper, val['src'], cmdargs.batch, device))

    meta = {
        'run' : cmdargs.run, 'step' : step, 'network' : 'ema',
        'solver' : f'{cmdargs.solver} {cmdargs.nfe}', 'clip' : 'at 0',
        'train_jets' : int(len(out)), 'seconds' : dt,
        'ms_per_jet' : 1000 * dt / len(out),
        'gpu' : torch.cuda.get_device_name(),
    }
    with open(paths['meta'], 'w', encoding = 'utf-8') as f:
        json.dump(meta, f, indent = 4)
    print(json.dumps(meta, indent = 2))

if __name__ == '__main__':
    main()
