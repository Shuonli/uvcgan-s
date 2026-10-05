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
    parser.add_argument('--cost', default = 'l2',
        choices = [ 'l2', 'shape_energy' ],
        help = 'jetflow matching cost: squared L2 of the states, or'
               ' fm_common.ShapeEnergyCost')
    parser.add_argument('--cost-lambda', type = float, default = 1.0,
        help = 'weight of the energy term of the shape_energy cost')
    parser.add_argument('--target-domain', default = 'closure_tgt',
        choices = [ 'closure_tgt', 'closure_null_tgt', 'tr_jewel' ],
        help = 'jetflow targets: T(B), B unmodified (the null test), or the'
               ' translation pilot\'s JEWEL jets (translation_data.py)')
    parser.add_argument('--source-domain', default = 'closure_src',
        choices = [ 'closure_src', 'tr_pythia' ],
        help = 'jetflow sources: the closure pool A, or the translation'
               ' pilot\'s PYTHIA jets')
    parser.add_argument('--norm-path', default = None,
        help = 'jetflow: the normalisation JSON (default: the closure\'s,'
               ' norm_closure.json; the translation pilot:'
               ' OUTDIR/sphenix/flow/translation/norm.json)')
    parser.add_argument('--coupling', default = 'exact', choices = fc.COUPLINGS,
        help = 'condjet: the minibatch plan of the (source, target) pairs,'
               ' exact OT or entropic OT (--ot-reg); one target per source row')
    parser.add_argument('--ot-reg', type = float, default = None,
        help = 'condjet --coupling entropic: the regularisation, in units of'
               ' the squared L2 cost of the standardised states')
    parser.add_argument('--sinkhorn-tol', type = float, default = 1e-4,
        help = 'condjet --coupling entropic: L1 error of the row marginals')
    parser.add_argument('--toy-kind', default = None, choices = [ 'sub', 'clean', 'ue' ],
        help = 'toyflow / toycond: the state transform (jamie norm.json)')
    parser.add_argument('--toy-pairing', default = None,
        choices = [ 'unpaired', 'synthetic', 'paired', 'hybrid' ],
        help = 'toy runs: pure unpaired, synthetic paired, true-paired control,'
               ' or the hybrid (jamie_methods.py)')
    parser.add_argument('--toy-source', default = None)
    parser.add_argument('--toy-target', default = None)
    parser.add_argument('--toy-prior', default = None)
    parser.add_argument('--toy-ue', default = None)
    parser.add_argument('--toy-data', default = None)
    parser.add_argument('--toy-cost', default = 'full', choices = [ 'full', 'nearfar' ],
        help = 'toy unpaired plans: full-image squared L2, or the pre-specified'
               ' near-jet / far-region normalised cost (jamie_methods.py)')
    parser.add_argument('--toy-lambda-u', type = float, default = 1.0,
        help = 'hybrid: weight of the unpaired term')
    parser.add_argument('--toy-extra', default = None,
        help = 'comma separated extra terms (jamie_methods.py)')
    parser.add_argument('--toy-lambda', default = None,
        help = 'comma separated weights of --toy-extra')
    parser.add_argument('--roll-n', type = int, default = 16,
        help = 'toy extra terms on solved endpoints: images per update')
    parser.add_argument('--roll-nfe', type = int, default = 16,
        help = 'toy extra terms on solved endpoints: midpoint evaluations')
    parser.add_argument('--init', default = None,
        help = 'continuation: start from this run\'s resume.pt (weights, EMA,'
               ' optimizer), no warm-up, a fresh step count')
    parser.add_argument('--ot-pool', type = int, default = None,
        help = 'matching pool per domain (default: the batch); the plan is'
               ' solved on the pool and --batch complete pairs drawn from it')
    parser.add_argument('--pairing', default = 'unpaired',
        choices = [ 'unpaired', 'paired', 'semi' ],
        help = 'jetflow: paired = each source jet with its own T(J), the'
               ' positive control of the closure test; semi = true pairs'
               ' next to OT-coupled unpaired pairs in every batch')
    parser.add_argument('--paired-n', type = int, default = None,
        help = 'jetflow paired / semi: only the first N source jets have a'
               ' known pair (default: all)')
    parser.add_argument('--paired-share', type = float, default = 0.5,
        help = 'jetflow semi: share of each batch that is true pairs')
    parser.add_argument('--sigma', type = float, default = None,
        help = 'path noise: 0 for otcfm and condcfm, 1 for sbcfm')
    parser.add_argument('--path', default = 'straight', choices = fc.PATHS,
        help = 'training path of the flow methods (sigma 0): straight, or'
               ' sine = straight + eta sin(pi t) eps with the target'
               ' b - a + eta pi cos(pi t) eps (fm_common.Method.sine_path)')
    parser.add_argument('--eta', type = float, default = 0.0,
        help = '--path sine: the largest noise sd, at t = 1/2, in the'
               ' standardised log-energy coordinates')
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
    parser.add_argument('--ckpt-steps', type = int, default = None,
        help = 'checkpoint every N updates instead (a multiple of 10), so'
               ' that runs are compared at the same updates')
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
        old.setdefault('cost', 'l2')
        old.setdefault('cost_lambda', 1.0)
        old.setdefault('pairing', 'unpaired')
        old.setdefault('target_domain', 'closure_tgt')
        old.setdefault('ot_pool', None)
        old.setdefault('paired_n', None)
        old.setdefault('paired_share', 0.5)
        old.setdefault('path', 'straight')
        old.setdefault('eta', 0.0)
        old.setdefault('source_domain', 'closure_src')
        old.setdefault('norm_path', None)
        old.setdefault('coupling', 'exact')
        old.setdefault('ot_reg', None)
        old.setdefault('sinkhorn_tol', 1e-4)
        for k in [ 'toy_kind', 'toy_pairing', 'toy_source', 'toy_target', 'toy_prior',
                   'toy_ue', 'toy_data', 'toy_extra', 'toy_lambda', 'init' ]:
            old.setdefault(k, None)
        old.setdefault('toy_lambda_u', 1.0)
        old.setdefault('toy_cost', 'full')
        old.setdefault('roll_n', 16)
        old.setdefault('roll_nfe', 16)
        for key in [ 'method', 'batch', 'lr', 'sigma', 'channels',
                     'res_blocks', 'attn', 'seed', 'ema', 'warmup',
                     'cosine_steps', 'log_bias', 'augment', 'backbone',
                     'cost', 'cost_lambda', 'pairing', 'target_domain',
                     'ot_pool', 'paired_n', 'paired_share', 'path', 'eta',
                     'source_domain', 'norm_path', 'coupling', 'ot_reg',
                     'sinkhorn_tol', 'toy_kind', 'toy_pairing', 'toy_source',
                     'toy_target', 'toy_prior', 'toy_ue', 'toy_data', 'toy_extra',
                     'toy_lambda', 'toy_lambda_u', 'roll_n', 'roll_nfe', 'init',
                     'toy_cost' ]:
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

    toy = cmdargs.method in fc.TOY_METHODS
    if toy:
        # pylint: disable=import-outside-toplevel
        import jamie_methods as jm
        norm = jm.load_norm()
        cmdargs.inline_events = 0
    elif cmdargs.method in fc.JET_METHODS:
        # fitted on the training jets by closure_data.py (or, for the
        # translation pilot, by translation_data.py: --norm-path)
        with open(cmdargs.norm_path or fc.Norm.path(method = 'jetflow'), 'r',
                  encoding = 'utf-8') as f:
            norm = fc.Norm(json.load(f))
        assert (cmdargs.source_domain == 'tr_pythia') \
            == (cmdargs.target_domain == 'tr_jewel') \
            == (cmdargs.norm_path is not None), \
            'the translation pools go with their own normalisation'
        if cmdargs.inline_events > 0:
            print('jetflow: no inline scoring (closure_eval.py, or'
                  ' translation_eval.py for the translation pilot, scores the'
                  ' checkpoints)')
            cmdargs.inline_events = 0
    else:
        norm = fc.Norm.load_or_fit(
            fc.Norm.path(cmdargs.log_bias), bias = cmdargs.log_bias
        )
    method = jm.ToyMethod(cmdargs, norm, device) if toy else \
        fc.Method(cmdargs.method, norm, cmdargs.sigma, cmdargs.augment,
                       cmdargs.cost, cmdargs.cost_lambda, cmdargs.pairing,
                       cmdargs.target_domain, cmdargs.paired_n, cmdargs.path,
                       cmdargs.eta, cmdargs.source_domain, cmdargs.coupling,
                       cmdargs.ot_reg, cmdargs.sinkhorn_tol)
    assert (cmdargs.method in ('condjet',) + fc.TOY_METHODS) or (cmdargs.coupling == 'exact'), \
        '--coupling: condjet and the toy methods only (the other OT methods have their own plans)'
    if cmdargs.pairing == 'semi':
        method.n_paired = int(round(cmdargs.paired_share * cmdargs.batch))
    cmdargs.sigma = method.sigma
    if cmdargs.path == 'sine':
        assert method.matcher is not None and method.sigma == 0, \
            '--path sine: a flow method with sigma 0'
    assert cmdargs.ckpt_steps is None or cmdargs.ckpt_steps % 10 == 0, \
        '--ckpt-steps: a multiple of 10 (the loop checks every 10 steps)'

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

    if (not os.path.exists(resume)) and cmdargs.init:
        # a continuation: the base run's weights, EMA and optimizer state; the
        # rate at its full value (no second warm-up), a fresh step count
        state = torch.load(os.path.join(cmdargs.init, 'resume.pt'), map_location = device,
                           weights_only = False)
        net.load_state_dict(state['raw'])
        ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['opt'])
        cmdargs.warmup = 1
        print(f"continuation of {cmdargs.init} (its step {state['stats']['step']})")
        np.random.seed(cmdargs.seed)
    elif os.path.exists(resume):
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
    # eps of the sine path: its own stream, like the data's, so that the
    # global RNG draws (times, the matcher's) do not depend on the path
    method.path_gen = torch.Generator(device = device)
    method.path_gen.manual_seed(2_000_029 * cmdargs.seed + 7 + stats['step'])
    if toy:
        method.set_streams(cmdargs.seed, stats['step'])
    if cmdargs.method == 'condjet':
        # times, noise and the pairs' row draws: one stream each, so that
        # the exact and the entropic coupling train on the same batches,
        # times and noise, step by step
        for (k, name) in enumerate([ 'time_gen', 'noise_gen', 'pair_gen' ]):
            gen = torch.Generator(device = device)
            gen.manual_seed((3_000_017 + k) * cmdargs.seed + 11 + k + stats['step'])
            setattr(method, name, gen)
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
    parts_sum = {}
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
        batch = method.draw(data, cmdargs.batch, cmdargs.ot_pool)
        stats['data_time'] += time.perf_counter() - t0

        (x0, x1, cond) = method.endpoints(batch)

        if method.coupled:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if method.name in ('condjet', 'toycond'):
                (cond, x1) = method.pair(cond, x1)
            else:
                (x0, x1) = method.couple(x0, x1, cmdargs.batch - method.n_paired)
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
        for (k, v) in getattr(method, 'last_parts', {}).items():
            parts_sum[k] = parts_sum.get(k, 0) + v
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
        if cmdargs.ckpt_steps:
            at_ckpt = (stats['step'] % cmdargs.ckpt_steps == 0) or done
        else:
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
            for (k, v) in parts_sum.items():
                row[k] = float(v) / n_sum
            parts_sum = {}
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
