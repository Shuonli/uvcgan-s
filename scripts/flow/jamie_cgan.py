#!/usr/bin/env python
"""CycleGAN on the toy's clean translation (FLOW_NOTES.md, "CycleGAN on the
girth-rule toy"): the repository's UVCGAN2 (`uvcgan-v2`), as in the PYTHIA ->
JEWEL baseline (translation_cgan.py), trained on the same unpaired pools as
the flow models E3g-D and E3g-C (or E3-D and E3-C with --target e3_med).

Held equal to the flow runs: the pools (40k vacuum jets, 40k quenched jets
of other parents), the standardised log(E + 0.1) coordinates of the toy's
`clean` kind (jamie/cache/norm.json), the ViT-ModNet generator (the flows'
velocity backbone without the time input), seed 0, one A6000. Outputs are
decoded as the flows' are, raw: E = exp(sd z + mu) - 0.1, not clipped.

CycleGAN settings as in translation_cgan.py (the repository's sPHENIX ones
where they apply): the sPHENIX resnet discriminator for 24 x 64 images with
spectral norm, hinge loss, gradient penalty 0.01, Adam 5e-5 (0.5, 0.99),
batch 32, EMA 0.9999 of the generators (used for inference), cycle weights
10, identity weight 0.5, learning-rate warm-up over the first 2000 updates.

    jamie_cgan.py --prepare [--target e3g_med]      # h5 copies of the pools
    jamie_cgan.py --label E3g-CG_s0 --hours 8 --milestones 1.5,4
    jamie_cgan.py --generate --label E3g-CG_s0      # test outputs per milestone
    jamie_cgan.py --generate --raw --label E3g-CG_s0    # the same, generator without EMA

--crop N trains on an N x N window around each jet's axis instead of the
whole 24 x 64 image (16: the PYTHIA -> JEWEL jet image and its
discriminator), standardised again on the windows; outputs are pasted back
into empty 24 x 64 images. jamie_cgan_score.py scores those against the
truth inside the same window.

Outputs go to jamie/outputs/LABEL_<h>h/ in jamie_eval.py's layout for a
deterministic model (one output per input), so jamie_report.py and
jamie_figs.py score them like D.
"""

import argparse
import json
import os
import time

import h5py
import numpy as np

import fm_common as fc
import jamie_methods as jm
import toycalo as tc
import translation_cgan as tcg

DISC_BLOCKS = [
    # (1, 24, 64): scripts/train/sphenix/train_uvcgan-s.py
    ('stem', { 'kernel_size' : 3, 'padding' : 1, 'stride' : 1, 'features' : 64 }),
    ('resnet', 3),
    ('stem', { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 128 }),
    # (128, 12, 32)
    ('resnet', 3),
    ('stem', { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 256 }),
    # (256, 6, 16)
    ('resnet', 3),
    ('stem', { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 512 }),
    # (512, 3, 8)
    ('resnet', 3),
]
TESTS = { 'e3g_med' : ('t3g_pair', 't3g_bank'), 'e3_med' : ('t3_pair', 't3_bank') }

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'CycleGAN on the toy translation')
    parser.add_argument('--prepare', action = 'store_true')
    parser.add_argument('--generate', action = 'store_true')
    parser.add_argument('--raw', action = 'store_true',
                        help = 'with --generate: the trained generator, not its EMA copy '
                               '(outputs LABEL_raw_<h>h)')
    parser.add_argument('--label', default = 'E3g-CG_s0')
    parser.add_argument('--crop', type = int, default = 0,
                        help = 'train on an N x N window around the jet axis (0: the whole image)')
    parser.add_argument('--source', default = 'e3_vac')
    parser.add_argument('--target', default = 'e3g_med', choices = list(TESTS))
    parser.add_argument('--hours', type = float, default = 8.0)
    parser.add_argument('--milestones', default = '1.5,4')
    parser.add_argument('--checkpoint', type = int, default = 10)
    parser.add_argument('--steps', type = int, default = 1000)
    parser.add_argument('--batch', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 5e-5)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--workers', type = int, default = 2)
    parser.add_argument('--outdir', default = None)
    return parser.parse_args()

def data_dir(cmdargs):
    crop = f'_c{cmdargs.crop}' if cmdargs.crop else ''
    return os.path.join(jm.jamie_root(), 'cgan_data', f'{cmdargs.source}_{cmdargs.target}{crop}')

def runs_dir():
    return os.path.join(jm.jamie_root(), 'runs')

def norm_path(cmdargs):
    if cmdargs.crop:
        return os.path.join(data_dir(cmdargs), 'norm.json')
    return os.path.join(jm.jamie_root(), 'norm.json')

def load_norm(cmdargs):
    with open(norm_path(cmdargs), encoding = 'utf-8') as f:
        return fc.Norm(json.load(f))

def window(eta0, phi0, n):
    """Tower indices of the n x n window whose tower n // 2 holds the axis:
    rows (M, n) in eta (outside 0..23 beyond the acceptance), columns (M, n)
    in phi (periodic)."""
    ie = np.floor((np.asarray(eta0) - tc.ETA_MIN) / tc.DETA).astype(int)
    ip = np.floor(np.asarray(phi0) / tc.DPHI).astype(int) % tc.NPHI
    off = np.arange(n) - n // 2
    return (ie[:, None] + off, (ip[:, None] + off) % tc.NPHI)

def window_mask(eta0, phi0, n):
    """(M, 24, 64) bool: the towers inside each jet's window."""
    return paste(np.ones((len(np.asarray(eta0)), n, n), bool), eta0, phi0, n)

def crop(images, eta0, phi0, n):
    """(M, 24, 64) -> (M, n, n) around the axes; towers beyond the
    acceptance are empty."""
    (rows, cols) = window(eta0, phi0, n)
    ok = (rows >= 0) & (rows < tc.NETA)
    k = np.arange(len(rows))[:, None, None]
    out = np.asarray(images)[k, np.clip(rows, 0, tc.NETA - 1)[:, :, None], cols[:, None, :]]
    return out * ok[:, :, None]

def paste(windows, eta0, phi0, n):
    """The inverse of crop(): (M, n, n) into empty (M, 24, 64) images."""
    (rows, cols) = window(eta0, phi0, n)
    ok = (rows >= 0) & (rows < tc.NETA)
    out = np.zeros((len(rows), tc.NETA, tc.NPHI), np.asarray(windows).dtype)
    k = np.broadcast_to(np.arange(len(rows))[:, None, None], windows.shape)
    r = np.broadcast_to(rows[:, :, None], windows.shape)
    c = np.broadcast_to(cols[:, None, :], windows.shape)
    sel = np.broadcast_to(ok[:, :, None], windows.shape)
    out[k[sel], r[sel], c[sel]] = np.asarray(windows)[sel]
    return out

def prepare(cmdargs):
    """The two pools as h5 arrays (N, 24, 64) of z, float32 (domains `a`:
    the source, `b`: the target)."""
    out = os.path.join(data_dir(cmdargs), 'train')
    os.makedirs(out, exist_ok = True)
    pools = {}
    for (domain, pool) in [ ('a', cmdargs.source), ('b', cmdargs.target) ]:
        e = np.load(os.path.join(jm.jamie_root(), 'cache', f'{pool}.npy')).astype(np.float64)
        if cmdargs.crop:
            with np.load(os.path.join(jm.jamie_root(), 'cache', f'{pool}_meta.npz')) as f:
                full = e.sum((1, 2))
                e = crop(e, f['eta0'], f['phi0'], cmdargs.crop)
            print(f'{pool}: {cmdargs.crop} x {cmdargs.crop} window keeps '
                  f'{(e.sum((1, 2)) / full).mean():.4f} of the energy, '
                  f'{(e > 1e-3).mean() * 100:.1f}% of its towers above 1 MeV', flush = True)
        pools[domain] = (pool, e)
    if cmdargs.crop:
        # the windows' own standardisation, fitted on both pools as norm.json's
        v = np.concatenate([ np.log(e + fc.BIAS).ravel() for (_, e) in pools.values() ])
        with open(norm_path(cmdargs), 'w', encoding = 'utf-8') as f:
            json.dump({ '_bias' : fc.BIAS, 'clean' : (float(v.mean()), float(v.std())) }, f, indent = 4)
    norm = load_norm(cmdargs)
    (mean, sd) = norm.stats['clean']
    for (domain, (pool, e)) in pools.items():
        z = ((np.log(e + norm.bias) - mean) / sd).astype(np.float32)
        path = os.path.join(out, f'{domain}.h5')
        with h5py.File(f'{path}.tmp', 'w') as f:
            f.create_dataset('data', data = z)
        os.replace(f'{path}.tmp', path)
        print(f'{path}: {pool} {z.shape}, z mean {z.mean():.4f}, sd {z.std():.4f}', flush = True)

def args_dict(cmdargs):
    tr = argparse.Namespace(**vars(cmdargs))
    args = tcg.args_dict(tr)
    dataset = lambda domain: {           # pylint: disable=unnecessary-lambda-assignment
        'dataset' : { 'name' : 'h5array-domain-hierarchy', 'path' : data_dir(cmdargs),
                      'domain' : domain },
        'shape' : (1, cmdargs.crop, cmdargs.crop) if cmdargs.crop else (1, 24, 64),
        'transform_train' : None, 'transform_test' : None }
    args['data']['datasets'] = [ dataset('a'), dataset('b') ]
    if cmdargs.crop != 16:
        # 16 x 16 keeps the PYTHIA -> JEWEL discriminator (tcg.DISC_BLOCKS)
        args['discriminator']['model_args']['block_specs'] = DISC_BLOCKS
    args['outdir'] = cmdargs.outdir or runs_dir()
    return args

def run_dir(cmdargs):
    root = cmdargs.outdir or runs_dir()
    names = [ n for n in os.listdir(root) if n.endswith(f'_{cmdargs.label}') ]
    assert len(names) == 1, names
    return os.path.join(root, names[0])

def generate(cmdargs):
    """Each milestone's EMA source -> target generator on the test inputs
    and the bank's parents (one output each)."""
    # pylint: disable=import-outside-toplevel,too-many-locals
    import torch
    from uvcgan_s.config import Args
    from uvcgan_s.cgan import construct_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = run_dir(cmdargs)
    with open(os.path.join(run, 'budget.json'), encoding = 'utf-8') as f:
        marks = json.load(f)['milestones']
    norm = load_norm(cmdargs)
    (setname, bankname) = TESTS[cmdargs.target]
    cache = lambda n: dict(np.load(os.path.join(jm.jamie_root(), 'cache', f'{n}.npz')))  # pylint: disable=unnecessary-lambda-assignment
    (d, b) = (cache(setname), cache(bankname))
    args = Args.load(run)
    for m in marks:
        model = construct_model(args.savedir, args.config, is_train = False, device = device)
        model.load(m['epoch'])
        model.eval()
        gen = model.models['avg_gen_ab']
        if cmdargs.raw:
            # the EMA copy has the generator's architecture; load the trained
            # weights into it (only the EMA copies are built at inference)
            gen.load_state_dict(torch.load(os.path.join(
                run, 'checkpoints', f"{m['epoch']:04d}_net_gen_ab.pth"), map_location = device))
            gen.eval()
        @torch.no_grad()
        def translate(x, sel = slice(None)):
            (eta0, phi0) = (d['eta0'][sel], d['phi0'][sel])
            if cmdargs.crop:
                x = crop(x, eta0, phi0, cmdargs.crop)
            out = []
            for s in range(0, len(x), 500):
                j = torch.as_tensor(np.asarray(x[s:s + 500], np.float32), device = device)
                y = gen(norm.z(j, 'clean').unsqueeze(1))
                out.append(norm.energy(y[:, 0], 'clean').cpu())
            out = torch.cat(out).numpy()
            return paste(out, eta0, phi0, cmdargs.crop) if cmdargs.crop else out
        label = f"{cmdargs.label}{'_raw' if cmdargs.raw else ''}_{m['milestone_h']:g}h"
        od = os.path.join(jm.jamie_root(), 'outputs', label)
        os.makedirs(od, exist_ok = True)
        t0 = time.perf_counter()
        out = translate(d['jet'])
        dt = time.perf_counter() - t0
        np.save(os.path.join(od, f'{setname}_vac.npy'), out.astype(np.float32))
        for (key, parents) in [ ('bank', b['parents']), ('illu', b['illu_parents']) ]:
            np.save(os.path.join(od, f'{bankname}_{key}.npy'),
                    translate(d['jet'][parents], parents)[:, None].astype(np.float16))
        meta = { 'run' : os.path.basename(run), 'model' : 'uvcgan-v2', **m, 'cond' : False,
                 'generator' : 'gen_ab' if cmdargs.raw else 'avg_gen_ab', 'crop' : cmdargs.crop,
                 'kind' : 'clean', 'nfe' : 1, 'solver' : 'generator',
                 'sets' : { f'{setname}_vac' : { 'n' : len(out), 'seconds' : dt,
                                                 'ms_per_image' : 1000 * dt / len(out) } } }
        with open(os.path.join(od, 'meta.json'), 'w', encoding = 'utf-8') as f:
            json.dump(meta, f, indent = 4)
        cone = float(out.sum((1, 2)).mean())
        print(f"{label}: epoch {m['epoch']}, {out.shape}, mean total {cone:.2f} GeV "
              f"(input {float(d['jet'].sum((1, 2)).mean()):.2f}, truth "
              f"{float(d['med'].sum((1, 2)).mean()):.2f})", flush = True)

def main():
    # pylint: disable=import-outside-toplevel
    cmdargs = parse_cmdargs()
    if cmdargs.prepare:
        prepare(cmdargs)
        return
    if cmdargs.generate:
        generate(cmdargs)
        return
    import torch
    from uvcgan_s import train
    torch.backends.cudnn.benchmark = True
    budget = tcg.Budget(cmdargs.hours, [ float(x) for x in cmdargs.milestones.split(',') if x ],
                        cmdargs.steps, norm_path = norm_path(cmdargs))
    try:
        train(args_dict(cmdargs), epoch_callback = budget)
    except tcg.BudgetReached:
        print(f'budget reached: {budget.marks}', flush = True)

if __name__ == '__main__':
    main()
