#!/usr/bin/env python
"""Train a flow-matching decomposition of the sPHENIX mixed events.

    fm_train.py --method METHOD --label NAME [--augment none|jets]
                [--minutes 25] [--ckpt-minutes 2.5] [--batch 256] ...

Output: OUTDIR/sphenix/flow/NAME/
    config.json      the arguments, library versions, GPU
    history.csv      one row per log interval: steps, examples, training
                     time (GPU-synchronised, without checkpointing and
                     evaluation), losses, throughput, coupling time, memory
    checkpoints/step_########.pt   raw and EMA weights, every --ckpt-minutes
                     of training time
    resume.pt        the full state of the last checkpoint (optimizer, RNGs)
    inline_eval.csv  quick val scores at each checkpoint (--inline-events),
                     read as fm_common.SELECTION says

Rerunning the same command resumes from resume.pt until the time budget
(--minutes of training time in total) is spent. c.f. FLOW_NOTES.md.
"""

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch

import fm_common as fc

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Train a flow decomposition')
    parser.add_argument('--method', required = True, choices = fc.METHODS)
    parser.add_argument('--label', required = True)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--batch', type = int, default = 256)
    parser.add_argument('--lr', type = float, default = 2e-4)
    parser.add_argument('--warmup', type = int, default = 1000,
        help = 'steps of linear learning rate warm-up')
    parser.add_argument('--cosine-steps', type = int, default = None,
        help = 'decay the rate to 0 along a cosine ending at this step'
               ' (default: constant after the warm-up)')
    parser.add_argument('--ema', type = float, default = 0.9999,
        help = 'EMA momentum, warmed up as min(ema, (1 + t) / (10 + t))')
    parser.add_argument('--grad-clip', type = float, default = 1.0)
    parser.add_argument('--log-bias', type = float, default = fc.BIAS,
        help = 'psi(E) = log(E + bias); 0.1 is the baseline data norm')
    parser.add_argument('--augment', default = 'none',
        choices = [ 'none', 'jets' ],
        help = 'jets: randomised signal shapes (fm_common.JetShapes)')
    parser.add_argument('--sigma', type = float, default = None,
        help = 'path noise: 0 for otcfm and condcfm, 1 for sbcfm')
    parser.add_argument('--backbone', default = 'unet',
        choices = [ 'unet', 'uvcgan' ],
        help = 'velocity network: the ADM U-Net, or the UVCGAN-S generator'
               ' (fm_common.UVCGANVelocity)')
    parser.add_argument('--channels', type = int, default = 96)
    parser.add_argument('--res-blocks', type = int, default = 2)
    parser.add_argument('--attn', default = '4',
        help = 'comma separated downsampling rates with attention')
    parser.add_argument('--minutes', type = float, default = 25,
        help = 'training time budget of the run, in total')
    parser.add_argument('--max-steps', type = int, default = None)
    parser.add_argument('--ckpt-minutes', type = float, default = 2.5)
    parser.add_argument('--log-steps', type = int, default = 100)
    parser.add_argument('--inline-events', type = int, default = 2000,
        help = 'val events scored at each checkpoint, 0 for none')
    parser.add_argument('--inline-nfe', type = int, default = 16)
    parser.add_argument('--outdir', default = None)
    return parser.parse_args()

def versions():
    # pylint: disable=import-outside-toplevel
    import ot
    import torchcfm

    try:
        commit = subprocess.run(
            [ 'git', 'rev-parse', '--short', 'HEAD' ], capture_output = True,
            text = True, check = False,
            cwd = os.path.dirname(os.path.abspath(__file__))
        ).stdout.strip()
    except OSError:
        commit = None

    return {
        'torch'    : torch.__version__,
        'torchcfm' : torchcfm.__version__,
        'pot'      : ot.__version__,
        'python'   : platform.python_version(),
        'gpu'      : torch.cuda.get_device_name(),
        'node'     : platform.node(),
        'commit'   : commit,
    }

def write_config(run_dir, cmdargs, n_params):
    path   = os.path.join(run_dir, 'config.json')
    config = {
        **vars(cmdargs),
        'attn'     : [ int(x) for x in cmdargs.attn.split(',') ],
        'n_params' : n_params,
    }

    if os.path.exists(path):
        with open(path, 'r', encoding = 'utf-8') as f:
            old = json.load(f)

        old.setdefault('augment', 'none')
        old.setdefault('backbone', 'unet')
        for key in [ 'method', 'batch', 'lr', 'sigma', 'channels',
                     'res_blocks', 'attn', 'seed', 'ema', 'warmup',
                     'cosine_steps', 'log_bias', 'augment', 'backbone' ]:
            if old.get(key) != config.get(key):
                raise RuntimeError(
                    f"resuming '{run_dir}' with {key} = {config.get(key)},"
                    f" it was trained with {old.get(key)}"
                )

        # the budget may be extended on resume
        old['minutes']   = cmdargs.minutes
        old['max_steps'] = cmdargs.max_steps
        config = old

    config.setdefault('runs', []).append(versions())

    with open(path, 'w', encoding = 'utf-8') as f:
        json.dump(config, f, indent = 4)

    return config

@torch.no_grad()
def update_ema(ema, net, momentum):
    torch._foreach_lerp_(                  # pylint: disable=protected-access
        list(ema.parameters()), list(net.parameters()), 1 - momentum
    )

    for (be, b) in zip(ema.buffers(), net.buffers()):
        be.copy_(b)

class InlineScorer:
    """Quick val scores of a network in training, not part of the timing."""

    def __init__(self, n_events, nfe):
        (embed, signal) = fc.ev.load_pairs(
            os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0
        )
        self.embed  = embed [:n_events]
        self.signal = signal[:n_events]
        self.nfe    = nfe
        self.truth  = None

    def __call__(self, method, nets, device):
        if self.truth is None:
            self.truth = fc.ev.Truth(
                self.embed, self.signal, fc.ev.cone_kernel(fc.ev.R_JET),
                10.0, device
            )

        rows = {}
        for (name, net) in nets.items():
            net.eval()
            dec = fc.Decomposer(
                method, net, nfe = self.nfe,
                decode = fc.SELECTION[method.name][2]
            )   # one sample: a monitor, not the selection setting
            (scores, _) = fc.ev.score_generator(
                dec, None, self.truth, 500, device
            )
            rows[name] = scores
            net.train()

        return rows

def main():
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')

    torch.backends.cudnn.benchmark = True

    run_dir = cmdargs.outdir or os.path.join(fc.out_root(), cmdargs.label)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok = True)

    t_start = time.perf_counter()

    norm   = fc.Norm.load_or_fit(
        fc.Norm.path(cmdargs.log_bias), bias = cmdargs.log_bias
    )
    method = fc.Method(cmdargs.method, norm, cmdargs.sigma, cmdargs.augment)
    cmdargs.sigma = method.sigma

    torch.manual_seed(cmdargs.seed)
    net = fc.construct_net(
        cmdargs.method, cmdargs.channels, cmdargs.res_blocks,
        [ int(x) for x in cmdargs.attn.split(',') ], cmdargs.backbone
    ).to(device)
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(net.parameters(), lr = cmdargs.lr)

    config = write_config(run_dir, cmdargs, fc.count_params(net))

    # resume
    resume = os.path.join(run_dir, 'resume.pt')
    stats  = {
        'step' : 0, 'train_time' : 0.0, 'coupling_time' : 0.0,
        'data_time' : 0.0, 'examples' : 0, 'eval_time' : 0.0,
        'ckpt_time' : 0.0,
    }

    if os.path.exists(resume):
        state = torch.load(resume, map_location = device, weights_only = False)
        net.load_state_dict(state['raw'])
        ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['opt'])
        stats.update(state['stats'])
        # map_location moved the RNG states to the GPU with the weights
        torch.set_rng_state(state['rng_cpu'].cpu())
        torch.cuda.set_rng_state(state['rng_cuda'].cpu())
        np.random.set_state(state['rng_np'])
        print(f"resumed at step {stats['step']},"
              f" {stats['train_time'] / 60:.1f} min of training")
    else:
        np.random.seed(cmdargs.seed)

    budget = 60 * cmdargs.minutes

    t0   = time.perf_counter()
    data = fc.GPUData(method.domains, device, cmdargs.seed, stats['step'])
    torch.cuda.synchronize()
    stats['load_time'] = stats.get('load_time', 0.0) + time.perf_counter() - t0
    print(f"data in GPU memory: {data.sizes()},"
          f" {time.perf_counter() - t0:.0f} s to load", flush = True)

    scorer = None
    if cmdargs.inline_events > 0:
        scorer = InlineScorer(cmdargs.inline_events, cmdargs.inline_nfe)

    history_path = os.path.join(run_dir, 'history.csv')

    def checkpoint():
        t0 = time.perf_counter()
        step = stats['step']

        common = {
            'raw' : net.state_dict(), 'ema' : ema.state_dict(),
            'norm' : norm.to_dict(), 'stats' : dict(stats), 'config' : config,
        }
        fc.save_checkpoint(
            os.path.join(run_dir, 'checkpoints', f'step_{step:08d}.pt'),
            **common
        )
        fc.save_checkpoint(
            resume, **common, opt = opt.state_dict(),
            rng_cpu = torch.get_rng_state(),
            rng_cuda = torch.cuda.get_rng_state(),
            rng_np = np.random.get_state(),
        )
        stats['ckpt_time'] += time.perf_counter() - t0

        if scorer is not None:
            t0   = time.perf_counter()
            rows = scorer(method, { 'ema' : ema, 'raw' : net }, device)
            dt   = time.perf_counter() - t0
            stats['eval_time'] += dt

            path = os.path.join(run_dir, 'inline_eval.csv')
            pd.DataFrame([
                { 'step' : step, 'train_time' : stats['train_time'],
                  'net' : name, **r, 'eval_time' : dt }
                    for (name, r) in rows.items()
            ]).to_csv(
                path, mode = 'a', header = not os.path.exists(path),
                index = False
            )

            print(f"step {step:7d} {stats['train_time'] / 60:6.1f} min  "
                  + '  '.join(
                      f"{n} jer_cal {r['jer_cal']:.2f} l1_sig {r['l1_sig']:.4f}"
                      f" jes {r['jes']:.2f}" for (n, r) in rows.items()
                  ) + f'  ({dt:.1f} s)', flush = True)

    loss_sum  = torch.zeros((), device = device)
    gnorm_sum = torch.zeros((), device = device)
    n_sum     = 0
    next_ckpt = (
        math.floor(stats['train_time'] / (60 * cmdargs.ckpt_minutes)) + 1
    ) * 60 * cmdargs.ckpt_minutes
    first     = True

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_seg = time.perf_counter()

    while True:
        if first:
            # start-up (normalisation, data loading, model) is not training
            stats['startup_time'] = (
                stats.get('startup_time', 0.0) + time.perf_counter() - t_start
            )
            torch.cuda.synchronize()
            t_seg = time.perf_counter()
            first = False

        t0 = time.perf_counter()
        batch = data.batch(cmdargs.batch)
        stats['data_time'] += time.perf_counter() - t0

        (x0, x1, cond) = method.endpoints(batch)

        if method.coupled:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            (x0, x1) = method.couple(x0, x1)
            torch.cuda.synchronize()
            stats['coupling_time'] += time.perf_counter() - t0

        lr = cmdargs.lr * min(1.0, (stats['step'] + 1) / cmdargs.warmup)
        if cmdargs.cosine_steps is not None:
            frac = min(stats['step'], cmdargs.cosine_steps) / cmdargs.cosine_steps
            lr  *= 0.5 * (1 + math.cos(math.pi * frac))
        for group in opt.param_groups:
            group['lr'] = lr

        loss = method.loss(net, x0, x1, cond)

        opt.zero_grad(set_to_none = True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            net.parameters(), cmdargs.grad_clip
        )
        opt.step()

        stats['step'] += 1
        momentum = min(cmdargs.ema, (1 + stats['step']) / (10 + stats['step']))
        update_ema(ema, net, momentum)

        stats['examples'] += cmdargs.batch
        loss_sum  += loss.detach()
        gnorm_sum += gnorm.detach()
        n_sum     += 1

        at_log = (stats['step'] % cmdargs.log_steps == 0)
        if not at_log and (stats['step'] % 10 != 0):
            continue

        torch.cuda.synchronize()
        now = time.perf_counter()
        stats['train_time'] += now - t_seg
        t_seg = now

        done = (stats['train_time'] >= budget) or (
            (cmdargs.max_steps is not None)
            and (stats['step'] >= cmdargs.max_steps)
        )
        at_ckpt = (stats['train_time'] >= next_ckpt) or done

        if at_log or at_ckpt:
            row = {
                'step'          : stats['step'],
                'train_time'    : stats['train_time'],
                'examples'      : stats['examples'],
                'loss'          : float(loss_sum) / n_sum,
                'gnorm'         : float(gnorm_sum) / n_sum,
                'lr'            : lr,
                'coupling_time' : stats['coupling_time'],
                'data_time'     : stats['data_time'],
                'peak_mem_gb'   : torch.cuda.max_memory_allocated() / 2**30,
            }
            recovery = method.pop_recovery()
            if recovery is not None:
                row['plan_recovery'] = recovery
            pd.DataFrame([ row ]).to_csv(
                history_path, mode = 'a',
                header = not os.path.exists(history_path), index = False
            )
            loss_sum.zero_()
            gnorm_sum.zero_()
            n_sum = 0

            if not math.isfinite(row['loss']):
                raise RuntimeError(f"loss is {row['loss']} at step {row['step']}")

        if at_ckpt:
            checkpoint()
            next_ckpt += 60 * cmdargs.ckpt_minutes
            # checkpointing and scoring are not training
            torch.cuda.synchronize()
            t_seg = time.perf_counter()

        if done:
            break

    stats['end_to_end_time'] = time.perf_counter() - t_start
    with open(os.path.join(run_dir, 'summary.json'), 'w',
              encoding = 'utf-8') as f:
        json.dump({
            **stats,
            'steps_per_s'   : stats['step'] / stats['train_time'],
            'samples_per_s' : stats['examples'] / stats['train_time'],
            'coupling_frac' : stats['coupling_time'] / stats['train_time'],
            'peak_mem_gb'   : torch.cuda.max_memory_allocated() / 2**30,
            'n_params'      : fc.count_params(net),
        }, f, indent = 4)

    print(f"done: {stats['step']} steps, {stats['train_time'] / 60:.1f} min"
          f" of training, {stats['step'] / stats['train_time']:.2f} steps/s,"
          f" coupling {stats['coupling_time'] / stats['train_time']:.1%}"
          f" of it", flush = True)

if __name__ == '__main__':
    main()
