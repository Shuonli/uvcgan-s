#!/usr/bin/env python
"""Continue the frozen few-pair teacher, with or without teacher-guided
unpaired data (FLOW_NOTES.md, "Teacher-guided coupling").

Both modes start from the teacher's validation-selected checkpoint (EMA
weights as the starting weights, and as the starting EMA). They then take
the same number of optimizer updates, each with the same paired term:
- 256 true pairs (J, T(J)) drawn from the same 2000 pairs (A[:2000]);
- the same pair-index and time sequences (their own generators);
- a paired-loss coefficient of 1.

    paired  loss = L_paired
    guided  loss = L_paired + weight * L_guided

L_guided is the flow-matching loss of 256 unpaired pairs (x, y):
- the x are 256 random training sources (all of A) and the y 256 random real
  targets from T(B);
- they are paired by the exact OT plan (the training solver) of the guided
  cost C(i, j) = d(F_teacher(x_i), y_j), with d the shape + energy cost
  (lambda 1, frozen scales), and 256 complete pairs are drawn from the plan
  as TorchCFM draws them;
- F_teacher comes from the cache of closure_teacher.py.

The student still flows from x to the real y: the teacher only chooses
which y. Each term is its own mean, so adding the unpaired term does not
dilute the paired gradient. Everything else is as in the closure runs: ADM
U-Net, Adam 2e-4 with 1000 warm-up steps, gradient clip 1, EMA 0.9999 (its
warm-up continuing from the teacher's step count), straight path sigma 0.

Checkpoints are saved every --ckpt-updates updates, in fm_train's format,
so that closure_eval.py can select and score them. Each checkpoint is also
monitored on 1000 validation pairs (32 midpoint evaluations). The run stops
if the monitored shape EMD exceeds --stop-factor times the teacher's at
three checkpoints in a row. History: losses and the time spent on the
paired term, the teacher lookup, the matching, and the unpaired term.

    closure_continue.py --label NAME --mode paired|guided [--updates 40000]
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import ot as pot
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.optimal_transport import OTPlanSampler

import fm_common as fc
from closure_data import Modification
from closure_eval import (load_pairs, Jets, apply, selected_step, checkpoints,
                          ckpt_step)
from closure_teacher import cache_paths

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Continue the teacher')
    parser.add_argument('--label', required = True)
    parser.add_argument('--mode', required = True, choices = [ 'paired', 'guided' ])
    parser.add_argument('--teacher', default = 'closure_pk1_s0')
    parser.add_argument('--paired-n', type = int, default = 2000)
    parser.add_argument('--batch', type = int, default = 256)
    parser.add_argument('--weight', type = float, default = 1.0,
        help = 'coefficient of the guided unpaired term')
    parser.add_argument('--updates', type = int, default = 40000)
    parser.add_argument('--ckpt-updates', type = int, default = 4000)
    parser.add_argument('--lr', type = float, default = 2e-4)
    parser.add_argument('--warmup', type = int, default = 1000)
    parser.add_argument('--ema', type = float, default = 0.9999)
    parser.add_argument('--grad-clip', type = float, default = 1.0)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--monitor-n', type = int, default = 1000)
    parser.add_argument('--stop-factor', type = float, default = 3.0)
    return parser.parse_args()

def cfm_loss(cfm, net, x0, x1, t):
    (t, xt, ut) = cfm.sample_location_and_conditional_flow(x0, x1, t)
    return torch.mean((net(t, xt) - ut)**2)

def main():
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    run_dir = os.path.join(fc.out_root(), cmdargs.label)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok = True)

    # the frozen teacher's selected checkpoint: the starting point
    t_dir  = os.path.join(fc.out_root(), cmdargs.teacher)
    t_step = selected_step(t_dir, '_32midpoint')
    t_ckpt = [ c for c in checkpoints(t_dir) if ckpt_step(c) == t_step ][0]
    (method, teacher_net, t_state, t_config) = fc.load_run(t_dir, t_ckpt,
                                                           device, 'ema')
    norm = method.norm
    net  = copy.deepcopy(teacher_net).train()
    for p in net.parameters():
        p.requires_grad_(True)
    ema  = copy.deepcopy(teacher_net).eval()
    del teacher_net
    opt  = torch.optim.Adam(net.parameters(), lr = cmdargs.lr)

    config = {
        **{ k : t_config[k] for k in [ 'method', 'channels', 'res_blocks',
                                       'attn', 'backbone', 'sigma' ]
            if k in t_config },
        'label' : cmdargs.label, 'mode' : cmdargs.mode,
        'teacher' : cmdargs.teacher, 'teacher_step' : t_step,
        'paired_n' : cmdargs.paired_n, 'batch' : cmdargs.batch,
        'weight' : cmdargs.weight if cmdargs.mode == 'guided' else 0.0,
        'updates' : cmdargs.updates, 'lr' : cmdargs.lr,
        'warmup' : cmdargs.warmup, 'ema' : cmdargs.ema,
        'grad_clip' : cmdargs.grad_clip, 'seed' : cmdargs.seed,
        'cost' : 'shape_energy (guided)' if cmdargs.mode == 'guided' else None,
        'pairing' : cmdargs.mode, 'gpu' : torch.cuda.get_device_name(),
    }
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding = 'utf-8') as f:
        json.dump(config, f, indent = 4)

    # data: training sources A (and the teacher's endpoints), targets T(B)
    src = torch.from_numpy(np.load(fc.cache_path('closure_src'))).to(device)
    modify = Modification(device)
    cfm = ConditionalFlowMatcher(sigma = 0.0)
    g_pair = torch.Generator(device = device).manual_seed(1_000_003 * cmdargs.seed)
    g_tp   = torch.Generator(device = device).manual_seed(7 + cmdargs.seed)
    if cmdargs.mode == 'guided':
        tgt   = torch.from_numpy(np.load(fc.cache_path('closure_tgt'))).to(device)
        teach = torch.from_numpy(np.load(
            cache_paths(cmdargs.teacher)['train'])).to(device)
        cost  = fc.ShapeEnergyCost.load(1.0)
        ot    = OTPlanSampler(method = 'exact')
        g_unp = torch.Generator(device = device).manual_seed(11 + cmdargs.seed)
        g_tu  = torch.Generator(device = device).manual_seed(13 + cmdargs.seed)
        np.random.seed(cmdargs.seed)     # sample_map draws with numpy

    # monitor: validation pairs, the teacher's value first
    val  = load_pairs('val', cmdargs.monitor_n)
    jets = Jets(val['row'], device)

    def monitor(model):
        model.eval()
        pred = apply(fc.JetMapper(method, model, 32, 'midpoint'), val['src'],
                     2000, device)
        (_, shape) = jets.emds(pred, val['tgt'])
        return float(np.nanmean(shape))

    teacher_monitor = monitor(ema)
    print(f'teacher (step {t_step}): monitored shape EMD {teacher_monitor:.5f}',
          flush = True)

    step0  = int(t_state['stats']['step'])
    timing = { 'paired' : 0.0, 'teacher_lookup' : 0.0, 'matching' : 0.0,
               'unpaired' : 0.0 }
    hist   = []
    bad    = 0
    t_start = time.perf_counter()
    train_time = 0.0

    for update in range(1, cmdargs.updates + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        lr = cmdargs.lr * min(1.0, update / cmdargs.warmup)
        for group in opt.param_groups:
            group['lr'] = lr

        idx = torch.randint(cmdargs.paired_n, (cmdargs.batch,), device = device,
                            generator = g_pair)
        j   = src[idx].float()
        x0p = method.source(j)
        x1p = method.source(modify(j))
        tp  = torch.rand(cmdargs.batch, device = device, generator = g_tp)
        loss_p = cfm_loss(cfm, net, x0p, x1p, tp)
        loss   = loss_p
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timing['paired'] += t1 - t0

        loss_u = torch.zeros((), device = device)
        if cmdargs.mode == 'guided':
            iu = torch.randint(len(src), (cmdargs.batch,), device = device,
                               generator = g_unp)
            iy = torch.randint(len(tgt), (cmdargs.batch,), device = device,
                               generator = g_unp)
            (xs, ys) = (src[iu].float(), tgt[iy].float())
            fx = teach[iu].float()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            timing['teacher_lookup'] += t2 - t1
            c    = cost(fx, ys).double().cpu().numpy()
            plan = ot.ot_fn(pot.unif(cmdargs.batch), pot.unif(cmdargs.batch), c)
            (a, b) = ot.sample_map(plan, cmdargs.batch)
            (a, b) = (torch.as_tensor(a, device = device),
                      torch.as_tensor(b, device = device))
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            timing['matching'] += t3 - t2
            tu = torch.rand(cmdargs.batch, device = device, generator = g_tu)
            loss_u = cfm_loss(cfm, net, method.source(xs[a]),
                              method.source(ys[b]), tu)
            loss = loss_p + cmdargs.weight * loss_u

        opt.zero_grad(set_to_none = True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cmdargs.grad_clip)
        opt.step()
        t_now = step0 + update
        with torch.no_grad():
            m = min(cmdargs.ema, (1 + t_now) / (10 + t_now))
            torch._foreach_lerp_(       # pylint: disable=protected-access
                list(ema.parameters()), list(net.parameters()), 1 - m)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        if cmdargs.mode == 'guided':
            timing['unpaired'] += t4 - t3
        else:
            timing['paired'] += t4 - t1
        train_time += t4 - t0

        if update % 100 == 0:
            hist.append({ 'update' : update, 'train_time' : train_time,
                          'loss_paired' : float(loss_p),
                          'loss_guided' : float(loss_u), 'lr' : lr,
                          **{ f'time_{k}' : v for (k, v) in timing.items() } })

        if update % cmdargs.ckpt_updates == 0:
            stats = { 'step' : update, 'train_time' : train_time,
                      'teacher_step' : t_step, **timing }
            fc.save_checkpoint(
                os.path.join(run_dir, 'checkpoints', f'step_{update:08d}.pt'),
                raw = net.state_dict(), ema = ema.state_dict(),
                norm = norm.to_dict(), stats = stats, config = config)
            value = monitor(ema)
            net.train()
            bad = bad + 1 if value > cmdargs.stop_factor * teacher_monitor else 0
            print(f'update {update:6d} {train_time / 60:6.1f} min  loss'
                  f' {float(loss_p):.2e} / {float(loss_u):.2e}  monitored shape'
                  f' EMD {value:.5f} (teacher {teacher_monitor:.5f})',
                  flush = True)
            pd.DataFrame(hist).to_csv(os.path.join(run_dir, 'history.csv'),
                                      index = False)
            if bad >= 3:
                print('stopped: three checkpoints in a row above'
                      f' {cmdargs.stop_factor}x the teacher', flush = True)
                break

    summary = { 'updates' : update, 'train_time' : train_time,
                'end_to_end' : time.perf_counter() - t_start,
                'updates_per_s' : update / train_time, **timing,
                'teacher_monitor' : teacher_monitor }
    with open(os.path.join(run_dir, 'summary.json'), 'w', encoding = 'utf-8') as f:
        json.dump(summary, f, indent = 4)
    print(json.dumps(summary, indent = 2))

if __name__ == '__main__':
    main()
