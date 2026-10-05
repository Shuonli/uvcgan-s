#!/usr/bin/env python
"""CycleGAN baseline of the PYTHIA -> JEWEL translation pilot (FLOW_NOTES.md,
"CycleGAN baseline"): the repository's own UVCGAN2 (`uvcgan-v2`; Torbunov et
al., arXiv:2203.02557 and 2303.16280, after CycleGAN, Zhu et al.,
arXiv:1703.10593), trained by the repository's trainer (uvcgan_s.train) on
the translation pilot's pools.

Held equal to the OT-FM (OT-CFM) run:
- the data: the same 200k PYTHIA and 200k JEWEL training canvases
  (translation_data.py), in the same standardised log(E + 0.1) coordinates
  (translation/norm.json); outputs are decoded the same way (E = exp(sd z +
  mu) - 0.1, clipped at 0);
- the generator architecture: the UVCGAN-S ViT-ModNet of the published
  sPHENIX model, which is also the OT-FM velocity backbone (without the
  time input), for both directions;
- the seed and one A6000.

CycleGAN settings are the repository's sPHENIX ones where they apply: the
resnet discriminator with spectral norm, hinge loss, gradient penalty 0.01,
Adam 5e-5 (0.5, 0.99), batch 32 (the repository's validated batch-32
setting), EMA 0.9999 of the generators (used for inference); UVCGAN2's cycle
weights 10 and identity weight 0.5; a linear learning-rate warm-up over the
first 2000 updates.

Budget: training time (the trainer's epoch_time, without checkpointing),
stopped at --hours; a checkpoint is saved at each of --milestones hours (the
first at the OT-FM's 2 GPU hours) and every --checkpoint epochs of 1000
updates (validation curves).

    translation_cgan.py --prepare                  # h5 copies of the pools
    translation_cgan.py --label tr_cgan_s0 --hours 8 --milestones 2
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
import pandas as pd

import fm_common as fc

DISC_BLOCKS = [
    # (1, 16, 16)
    ('stem', { 'kernel_size' : 3, 'padding' : 1, 'stride' : 1, 'features' : 64 }),
    ('resnet', 3),
    ('stem', { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 128 }),
    # (128, 8, 8)
    ('resnet', 3),
    ('stem', { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 256 }),
    # (256, 4, 4)
    ('resnet', 3),
    ('stem', { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 512 }),
    # (512, 2, 2)
    ('resnet', 3),
]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'CycleGAN baseline')
    parser.add_argument('--prepare', action = 'store_true')
    parser.add_argument('--label', default = 'tr_cgan_s0')
    parser.add_argument('--hours', type = float, default = 8.0)
    parser.add_argument('--milestones', default = '2')
    parser.add_argument('--checkpoint', type = int, default = 10,
        help = 'epochs (of --steps updates) between checkpoints')
    parser.add_argument('--steps', type = int, default = 1000)
    parser.add_argument('--batch', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 5e-5)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--workers', type = int, default = 2)
    parser.add_argument('--outdir', default = None,
        help = 'where the run directory goes (default: translation/runs)')
    return parser.parse_args()

def data_dir():
    return os.path.join(fc.translation_root(), 'cgan_data')

def runs_dir():
    return os.path.join(fc.translation_root(), 'runs')

def prepare():
    """The training pools as h5 arrays (N, 16, 16) of z, float32, for the
    repository's h5array-domain-hierarchy dataset (its default transform,
    ToTensor, adds the channel axis, as for the sPHENIX h5 files)."""
    with open(fc.translation_norm_path(), encoding = 'utf-8') as f:
        (mean, sd) = json.load(f)['jet']
    out = os.path.join(data_dir(), 'train')
    os.makedirs(out, exist_ok = True)
    for domain in [ 'pythia', 'jewel' ]:
        e = np.load(fc.cache_path(f'tr_{domain}')).astype(np.float64)
        z = ((np.log(e + fc.BIAS) - mean) / sd).astype(np.float32)
        path = os.path.join(out, f'{domain}.h5')
        with h5py.File(f'{path}.tmp', 'w') as f:
            f.create_dataset('data', data = z)
        os.replace(f'{path}.tmp', path)
        print(f'{path}: {z.shape}, z mean {z.mean():.4f}, sd {z.std():.4f}', flush = True)

def args_dict(cmdargs):
    gen_args = { **fc.UVCGAN_GENERATOR }
    dataset = lambda domain: {           # pylint: disable=unnecessary-lambda-assignment
        'dataset' : { 'name' : 'h5array-domain-hierarchy', 'path' : data_dir(),
                      'domain' : domain },
        'shape' : (1, *fc.JET_SHAPE), 'transform_train' : None, 'transform_test' : None }
    return {
        'batch_size' : cmdargs.batch,
        'data' : { 'datasets' : [ dataset('pythia'), dataset('jewel') ],
                   'merge_type' : 'unpaired', 'workers' : cmdargs.workers },
        'epochs' : 100000,
        'discriminator' : {
            'model' : 'resnet',
            'model_args' : { 'block_specs' : DISC_BLOCKS, 'norm' : 'batch',
                             'activ' : 'leakyrelu', 'rezero' : True,
                             'reduce_output_channels' : True },
            'optimizer' : { 'name' : 'Adam', 'lr' : cmdargs.lr, 'betas' : (0.5, 0.99) },
            'weight_init' : { 'name' : 'normal', 'init_gain' : 0.02 },
            'spectr_norm' : True,
        },
        'generator' : {
            'model' : 'vit-modnet', 'model_args' : gen_args,
            'optimizer' : { 'name' : 'Adam', 'lr' : cmdargs.lr, 'betas' : (0.5, 0.99) },
            'weight_init' : { 'name' : 'kaiming' },
        },
        'model' : 'uvcgan-v2',
        'model_args' : { 'lambda_a' : 10, 'lambda_b' : 10, 'lambda_idt' : 0.5,
                         'avg_momentum' : 0.9999, 'gp_cache_period' : 0,
                         'head_queue_size' : 0, 'head_config' : 'idt' },
        'seed' : cmdargs.seed,
        'scheduler' : { 'name' : 'linear-v2', 'start_factor' : 0.01, 'end_factor' : 1.0,
                        'total_iters' : 2 },
        'loss' : 'hinge',
        'steps_per_epoch' : cmdargs.steps,
        'transfer' : None,
        'gradient_penalty' : { 'center' : 0, 'lambda_gp' : 0.01, 'mix_type' : 'real-fake',
                               'reduction' : 'mean' },
        'label' : cmdargs.label,
        'outdir' : cmdargs.outdir or runs_dir(),
        'log_level' : 'INFO',
        'checkpoint' : cmdargs.checkpoint,
    }

class BudgetReached(Exception):
    pass

class Budget:
    """Epoch callback: saves a checkpoint when the training time first
    reaches each milestone, and stops at the budget. Training time is the
    sum of the trainer's epoch_time (checkpointing and this callback not
    included), as fm_train.py and dsbm.py count theirs."""

    def __init__(self, hours, milestones, steps):
        self.budget = 3600 * hours
        self.milestones = sorted(3600 * m for m in milestones)
        self.steps = steps
        self.marks = []
        self.t0 = time.perf_counter()

    def __call__(self, model, epoch):
        hist = pd.read_csv(os.path.join(model.savedir, 'history.csv'))
        t = float(hist['epoch_time'].sum())
        while self.milestones and t >= self.milestones[0]:
            model.save(epoch)
            self.marks.append({ 'milestone_h' : self.milestones.pop(0) / 3600,
                                'epoch' : epoch, 'updates' : epoch * self.steps,
                                'train_time' : t })
            self.write(model)
        if t >= self.budget:
            model.save(epoch)
            self.marks.append({ 'milestone_h' : self.budget / 3600, 'epoch' : epoch,
                                'updates' : epoch * self.steps, 'train_time' : t,
                                'final' : True })
            self.write(model, hist)
            raise BudgetReached()

    def write(self, model, hist = None):
        info = { 'milestones' : self.marks, 'norm_path' : fc.translation_norm_path(),
                 'wall_since_start' : time.perf_counter() - self.t0 }
        if hist is not None:
            t = float(hist['epoch_time'].sum())
            info.update({ 'epochs' : int(hist['epoch'].max()),
                          'updates' : int(hist['epoch'].max()) * self.steps,
                          'train_time' : t,
                          'updates_per_s' : int(hist['epoch'].max()) * self.steps / t })
        with open(os.path.join(model.savedir, 'budget.json'), 'w', encoding = 'utf-8') as f:
            json.dump(info, f, indent = 4)

def main():
    # pylint: disable=import-outside-toplevel
    cmdargs = parse_cmdargs()
    if cmdargs.prepare:
        prepare()
        return
    import torch
    from uvcgan_s import train
    torch.backends.cudnn.benchmark = True
    budget = Budget(cmdargs.hours, [ float(x) for x in cmdargs.milestones.split(',') if x ],
                    cmdargs.steps)
    try:
        train(args_dict(cmdargs), epoch_callback = budget)
    except BudgetReached:
        print(f'budget reached: {budget.marks}', flush = True)

if __name__ == '__main__':
    main()
