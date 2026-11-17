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
    return os.path.join(jm.jamie_root(), 'cgan_data', f'{cmdargs.source}_{cmdargs.target}')

def runs_dir():
    return os.path.join(jm.jamie_root(), 'runs')

def norm_path():
    return os.path.join(jm.jamie_root(), 'norm.json')

def prepare(cmdargs):
    """The two pools as h5 arrays (N, 24, 64) of z, float32 (domains `a`:
    the source, `b`: the target)."""
    norm = jm.load_norm()
    (mean, sd) = norm.stats['clean']
    out = os.path.join(data_dir(cmdargs), 'train')
    os.makedirs(out, exist_ok = True)
    for (domain, pool) in [ ('a', cmdargs.source), ('b', cmdargs.target) ]:
        e = np.load(os.path.join(jm.jamie_root(), 'cache', f'{pool}.npy')).astype(np.float64)
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
        'shape' : (1, 24, 64), 'transform_train' : None, 'transform_test' : None }
    args['data']['datasets'] = [ dataset('a'), dataset('b') ]
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
    norm = jm.load_norm()
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
        def translate(x):
            out = []
            for s in range(0, len(x), 500):
                j = torch.as_tensor(np.asarray(x[s:s + 500], np.float32), device = device)
                y = gen(norm.z(j, 'clean').unsqueeze(1))
                out.append(norm.energy(y[:, 0], 'clean').cpu())
            return torch.cat(out).numpy()
        label = f"{cmdargs.label}{'_raw' if cmdargs.raw else ''}_{m['milestone_h']:g}h"
        od = os.path.join(jm.jamie_root(), 'outputs', label)
        os.makedirs(od, exist_ok = True)
        t0 = time.perf_counter()
        out = translate(d['jet'])
        dt = time.perf_counter() - t0
        np.save(os.path.join(od, f'{setname}_vac.npy'), out.astype(np.float32))
        for (key, parents) in [ ('bank', b['parents']), ('illu', b['illu_parents']) ]:
            np.save(os.path.join(od, f'{bankname}_{key}.npy'),
                    translate(d['jet'][parents])[:, None].astype(np.float16))
        meta = { 'run' : os.path.basename(run), 'model' : 'uvcgan-v2', **m, 'cond' : False,
                 'generator' : 'gen_ab' if cmdargs.raw else 'avg_gen_ab',
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
                        cmdargs.steps, norm_path = norm_path())
    try:
        train(args_dict(cmdargs), epoch_callback = budget)
    except tcg.BudgetReached:
        print(f'budget reached: {budget.marks}', flush = True)

if __name__ == '__main__':
    main()
