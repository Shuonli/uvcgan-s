#!/usr/bin/env python
"""Images of the consolidated subtraction benchmark (FLOW_NOTES.md,
"Consolidated benchmark"): the inputs of every analysis and figure.

The events are those of every earlier score: 20k PYTHIA-in-HIJING val
mixtures (development; val events 10k-20k also serve as the calibration
subset) and 20k JEWEL-in-HIJING test mixtures (frozen, for the final
domain-shift check), as eval_val_truth.load_pairs picks them. Written to
OUTDIR/sphenix/flow/bench/images/<set>/, float32, (N, 24, 64), GeV:

    mixture.npy, truth.npy            M and the detector-level signal S
    uvcgan__sig.npy / __bkg.npy       the published UVCGAN-S (EMA
                                      generator), its two channels
    <run>__<setting>__bkg.npy         a flow run at its val-selected
                                      checkpoint (EMA): the background B_hat
    <run>__<setting>__sig.npy         its signal: M - B_hat (one-panel
                                      arms, and the joint arm's residual
                                      readout) ...
    <run>__<setting>__sigdirect.npy   ... and the joint arm's directly
                                      generated S_hat

Settings: `euler4` (the selection readout) and `midpoint32` (the accurately
resolved ODE reference). Nothing is clipped here; the analyses apply their
documented rules. The checkpoint and settings are recorded in
`<set>/manifest.json`.

    bench_images.py --uvcgan --runs RUN [RUN ...] [--sets val,jewel]
"""

import argparse
import fcntl
import json
import os
import time

import numpy as np
import torch

import fm_common as fc
from fm_eval import best_step, ckpt_step
from readout_test import PUBLISHED

ev = fc.ev
SETTINGS = { 'euler4' : (4, 'euler'), 'midpoint32' : (32, 'midpoint') }

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Benchmark images')
    parser.add_argument('--runs', nargs = '*', default = [])
    parser.add_argument('--uvcgan', action = 'store_true')
    parser.add_argument('--sets', default = 'val,jewel')
    parser.add_argument('--settings', default = 'euler4,midpoint32')
    parser.add_argument('--n-events', type = int, default = 20000)
    parser.add_argument('--batch', type = int, default = 1000)
    parser.add_argument('--force', action = 'store_true')
    return parser.parse_args()

def out_dir(name):
    path = os.path.join(fc.out_root(), 'bench', 'images', name)
    os.makedirs(path, exist_ok = True)
    return path

def manifest(path, key, value):
    f = os.path.join(path, 'manifest.json')
    # several runs' jobs write at once
    with open(f + '.lock', 'w', encoding = 'utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.load(open(f, encoding = 'utf-8')) if os.path.exists(f) else {}
        data[key] = value
        with open(f, 'w', encoding = 'utf-8') as out:
            json.dump(data, out, indent = 4)

@torch.no_grad()
def run_batches(fn, embed, batch, device):
    outs = None
    for start in range(0, len(embed), batch):
        m = torch.from_numpy(embed[start:start + batch]).to(device).float()
        res = fn(m)
        if outs is None:
            outs = [ [] for _ in res ]
        for (o, r) in zip(outs, res):
            o.append(r.cpu().numpy().astype(np.float32))
    return [ np.concatenate(o) for o in outs ]

def uvcgan(device):
    # pylint: disable=import-outside-toplevel
    from uvcgan_s.config import Args
    from uvcgan_s.cgan   import construct_model
    args  = Args.load(os.path.join(
        os.environ.get('UVCGAN_S_OUTDIR', 'outdir'), PUBLISHED))
    model = construct_model(args.savedir, args.config, is_train = False,
                            device = device)
    model.load(None)
    (gen, norm) = (model.models.ema_gen_ba.eval(), model.data_norm)

    def fn(m):
        y = norm.denormalize(gen(norm.normalize(m.unsqueeze(1))))
        return (y[:, 1], y[:, 0])
    return fn

def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    data    = os.environ.get('UVCGAN_S_DATA', 'data')

    for name in cmdargs.sets.split(','):
        (embed, signal) = ev.load_pairs(data, cmdargs.n_events, 0, truth = name)
        path = out_dir(name)
        for (f, x) in [ ('mixture', embed), ('truth', signal) ]:
            if cmdargs.force or not os.path.exists(os.path.join(path, f'{f}.npy')):
                np.save(os.path.join(path, f'{f}.npy'), x.astype(np.float32))
            else:
                # every stored image refers to these events
                assert np.array_equal(np.load(os.path.join(path, f'{f}.npy'),
                                              mmap_mode = 'r'), x.astype(np.float32))

        if cmdargs.uvcgan:
            (sig, bkg) = run_batches(uvcgan(device), embed, cmdargs.batch, device)
            np.save(os.path.join(path, 'uvcgan__sig.npy'), sig)
            np.save(os.path.join(path, 'uvcgan__bkg.npy'), bkg)
            manifest(path, 'uvcgan', { 'checkpoint' : PUBLISHED,
                                       'network' : 'ema_gen_ba' })
            print(f'{name}: uvcgan done', flush = True)

        for run in cmdargs.runs:
            run_dir = os.path.join(fc.out_root(), run)
            step = best_step(run_dir)
            ckpt = [ c for c in fc.list_checkpoints(run_dir)
                     if ckpt_step(c) == step ][0]
            (method, net, state, config) = fc.load_run(run_dir, ckpt, device, 'ema')
            for setting in cmdargs.settings.split(','):
                (nfe, solver) = SETTINGS[setting]
                dec = fc.Decomposer(method, net, nfe = nfe, solver = solver,
                                    decode = 'direct')
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                (y,) = run_batches(lambda m: (dec(m.unsqueeze(1)),), embed,
                                   cmdargs.batch, device)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                bkg = y[:, 0]
                np.save(os.path.join(path, f'{run}__{setting}__bkg.npy'), bkg)
                np.save(os.path.join(path, f'{run}__{setting}__sig.npy'),
                        embed.astype(np.float32) - bkg)
                if method.name in ('joint_paired', 'otcfm', 'otcfm_pieces'):
                    np.save(os.path.join(path, f'{run}__{setting}__sigdirect.npy'),
                            y[:, 1])
                manifest(path, f'{run}__{setting}', {
                    'run' : run, 'method' : method.name,
                    'backbone' : config.get('backbone', 'unet'),
                    'seed' : config.get('seed'), 'step' : step,
                    'train_time_min' : state['stats']['train_time'] / 60,
                    'updates' : state['stats']['step'], 'network' : 'ema',
                    'nfe' : nfe, 'solver' : solver,
                    'seconds' : dt, 'gpu' : torch.cuda.get_device_name(),
                })
                print(f'{name}: {run} {setting} done ({dt:.0f} s)', flush = True)

if __name__ == '__main__':
    main()
