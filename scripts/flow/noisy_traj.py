#!/usr/bin/env python
"""Solver check and trajectory diagnostic of the paired noisy-interpolant
pilot (FLOW_NOTES.md, "Paired noisy-interpolant pilot"); outside training and
model selection. Every run at the same checkpoint (--step, default each
run's last), EMA network, deterministic ODE from a = z_bkg(M); B_hat =
energy(x_1), S_hat = M - B_hat; b = z_bkg(B) with B = M - S (as in training).
Hard towers: true signal S > --hard GeV (5).

1. Solver check (the first --n-solver val events): B_hat with 4 Euler steps,
   midpoint 32 and 64 network evaluations; per-tower RMS difference to the
   64 solve, and the leakage share sum(B_hat - B) / sum(S) per true-signal
   bin (bench_diag.py's bins) at each solve.
2. ODE trajectories (the first --n-traj val events): midpoint 32 (16
   steps), the state after every step read as a background, B(x_t) =
   energy(x_t); per hard tower whether B(x_t) ever falls below the true B,
   and whether it ends above it; the signed error sum(B(x_t) - B) / sum(S)
   of the hard towers against t, next to that of the clean interpolant
   (1 - t) a + t b (always >= 0 there, reaching 0 at t = 1).
3. Perturbation response (the same events): from the clean interpolant at
   t0 = 0.25, 0.5, 0.75, unperturbed and with -d and +d along two fixed
   directions (the same for every run): 'hard', +1 in every hard tower (a
   state too close to the mixture there); 'random', a fixed standard
   Gaussian tensor. Integrated to t = 1 with midpoint steps of 1/64. Per
   start: the endpoint's hard-tower leakage share and RMS error (GeV)
   against the true B, and the endpoint response (x_1(+d) - x_1(-d)) / 2d
   projected on the direction (hard towers; 1 = the displacement
   survives, 0 = it is removed). A small response alone is not success:
   the error against the true B decides.

    noisy_traj.py RUN [RUN ...] --labels A,B [--out docs/flow/bench/noisy/traj]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import fm_common as fc
from fm_eval import ckpt_step

ev = fc.ev
S_BINS = [ ('S = 0', 0, 0), ('0-0.5', 0, 0.5), ('0.5-2', 0.5, 2),
           ('2-5', 2, 5), ('5-10', 5, 10), ('>10', 10, 1e9) ]
T0 = (0.25, 0.5, 0.75)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Noisy-interpolant diagnostics')
    parser.add_argument('runs', nargs = '+')
    parser.add_argument('--labels', required = True)
    parser.add_argument('--step', default = 'last', help = "'last' or a step")
    parser.add_argument('--n-solver', type = int, default = 1000)
    parser.add_argument('--n-traj', type = int, default = 256)
    parser.add_argument('--hard', type = float, default = 5.0)
    parser.add_argument('--delta', type = float, default = 0.05,
        help = 'perturbation size, standardised log-energy units')
    parser.add_argument('--batch', type = int, default = 250)
    parser.add_argument('--out', default = 'docs/flow/bench/noisy/traj')
    return parser.parse_args()

def load(run, step, device):
    run_dir = os.path.join(fc.out_root(), run)
    ckpts = fc.list_checkpoints(run_dir)
    ckpt = ckpts[-1] if step == 'last' else \
        [ c for c in ckpts if ckpt_step(c) == int(step) ][0]
    (method, net, state, config) = fc.load_run(run_dir, ckpt, device, 'ema')
    return (method, net, ckpt_step(ckpt), config)

def velocity(net, t, x):
    return net(torch.full((x.shape[0],), t, device = x.device), x)

@torch.no_grad()
def solve(net, x, t0, nfe_or_h, solver, record = False):
    """Midpoint (steps of h from t0, or nfe/2 equal steps from 0) or Euler;
    with record, the states after every step."""
    if solver == 'euler':
        h = 1.0 / nfe_or_h
        for k in range(nfe_or_h):
            x = x + h * velocity(net, k * h, x)
        return x
    if t0 == 0 and nfe_or_h >= 1:
        steps = nfe_or_h // 2
        h = 1.0 / steps
    else:
        h = nfe_or_h
        steps = int(round((1 - t0) / h))
    states = [ x ] if record else None
    for k in range(steps):
        t = t0 + k * h
        xm = x + 0.5 * h * velocity(net, t, x)
        x  = x + h * velocity(net, t + 0.5 * h, xm)
        if record:
            states.append(x)
    return states if record else x

def shares(bhat, b, s):
    """Leakage share sum(B_hat - B) / sum(S) per true-signal bin; S = 0: the
    mean B_hat - B per tower (GeV)."""
    out = {}
    err = bhat - b
    for (name, lo, hi) in S_BINS:
        sel = (s == 0) if name == 'S = 0' else ((s > lo) & (s <= hi))
        if name == 'S = 0':
            out[f'bkgerr_mean[{name}]'] = float(err[sel].mean())
        else:
            out[f'share[{name}]'] = float(err[sel].sum() / s[sel].sum())
    return out

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    labels  = cmdargs.labels.split(',')
    n = max(cmdargs.n_solver, cmdargs.n_traj)
    (embed, signal) = ev.load_pairs(os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0)
    m_all = torch.from_numpy(embed[:n]).float().to(device)
    s_all = torch.from_numpy(signal[:n]).float().to(device)
    b_all = (m_all - s_all).clamp(min = 0)
    gen = torch.Generator(device = device).manual_seed(2026)
    xi  = torch.randn((cmdargs.n_traj, 1, *m_all.shape[1:]), device = device,
                      generator = gen)
    solver_rows, traj_rows, pert_rows, meta = [], [], [], {}

    for (run, label) in zip(cmdargs.runs, labels):
        (method, net, step, config) = load(run, cmdargs.step, device)
        norm = method.norm
        meta[label] = { 'run' : run, 'step' : step, 'eta' : config.get('eta', 0.0),
                        'path' : config.get('path', 'straight') }
        dec = lambda x: norm.energy(x[:, 0], 'bkg')

        # 1. solver check
        outs = {}
        for (name, nfe, solver) in [ ('euler4', 4, 'euler'), ('mid32', 32, 'midpoint'),
                                     ('mid64', 64, 'midpoint') ]:
            res = []
            for i in range(0, cmdargs.n_solver, cmdargs.batch):
                a = method.source(m_all[i:i + cmdargs.batch])
                res.append(dec(solve(net, a, 0, nfe, solver)))
            outs[name] = torch.cat(res)
        (bb, ss) = (b_all[:cmdargs.n_solver], s_all[:cmdargs.n_solver])
        for (name, bhat) in outs.items():
            row = { 'model' : label, 'solve' : name, 'n_events' : cmdargs.n_solver,
                    'rms_diff_to_mid64_gev' : float((bhat - outs['mid64']).pow(2).mean().sqrt()),
                    'event_sum_diff_to_mid64_gev' : float((bhat - outs['mid64']).sum((1, 2)).mean()),
                    'mae_gev' : float((bhat - bb).abs().mean()),
                    **shares(bhat, bb, ss) }
            solver_rows.append(row)
            print(row, flush = True)

        # 2. trajectories
        (m, s, b) = (m_all[:cmdargs.n_traj], s_all[:cmdargs.n_traj], b_all[:cmdargs.n_traj])
        hard = s > cmdargs.hard
        a  = method.source(m)
        zb = norm.z(b, 'bkg').unsqueeze(1)
        states = solve(net, a, 0, 32, 'midpoint', record = True)
        ts = np.linspace(0, 1, len(states))
        errs = torch.stack([ dec(x) - b for x in states ])            # (T, N, H, W)
        below = (errs < 0) & hard                                    # per t
        ever_below = below[1:].any(0) & hard
        end_above = (errs[-1] > 0) & hard
        n_hard = int(hard.sum())
        for (k, t) in enumerate(ts):
            x_ref = (1 - t) * a + t * zb
            traj_rows.append({
                'model' : label, 't' : float(t),
                'hard_share_err' : float(errs[k][hard].sum() / s[hard].sum()),
                'hard_share_below' : float(below[k].sum() / n_hard),
                'ref_share_err' : float((dec(x_ref) - b)[hard].sum() / s[hard].sum()),
            })
        meta[label].update({
            'hard_towers' : n_hard,
            'share_ever_below_true_b' : float(ever_below.sum() / n_hard),
            'share_end_above_true_b' : float(end_above.sum() / n_hard),
            'share_ever_below_and_end_above' : float((ever_below & end_above).sum() / n_hard),
            'endpoint_hard_share_err' : float(errs[-1][hard].sum() / s[hard].sum()),
        })

        # 3. perturbation response from the clean interpolant
        dirs = { 'hard' : hard.unsqueeze(1).float(), 'random' : xi }
        for t0 in T0:
            x_ref = (1 - t0) * a + t0 * zb
            for (dname, v) in dirs.items():
                ends = {}
                for sign in (-1, 0, 1):
                    x = x_ref + sign * cmdargs.delta * v
                    ends[sign] = solve(net, x, t0, 1 / 64, 'midpoint')
                resp = (ends[1] - ends[-1]) / (2 * cmdargs.delta)
                hv = hard.unsqueeze(1)
                # projection of the response on the direction, hard towers
                proj = float((resp * v)[hv].sum() / (v * v)[hv].sum())
                for sign in (-1, 0, 1):
                    bhat = dec(ends[sign])
                    pert_rows.append({
                        'model' : label, 't0' : t0, 'direction' : dname,
                        'delta' : sign * cmdargs.delta,
                        'hard_share_err' : float((bhat - b)[hard].sum() / s[hard].sum()),
                        'hard_rms_err_gev' : float((bhat - b)[hard].pow(2).mean().sqrt()),
                        'all_rms_err_gev' : float((bhat - b).pow(2).mean().sqrt()),
                        'response_hard' : proj,
                        'response_rms_all' : float(resp.pow(2).mean().sqrt()
                                                   / v.pow(2).mean().sqrt()),
                    })
        print(label, meta[label], flush = True)

    os.makedirs(os.path.dirname(cmdargs.out), exist_ok = True)
    pd.DataFrame(solver_rows).to_csv(f'{cmdargs.out}_solver.csv', index = False)
    tr = pd.DataFrame(traj_rows)
    tr.to_csv(f'{cmdargs.out}_paths.csv', index = False)
    pe = pd.DataFrame(pert_rows)
    pe.to_csv(f'{cmdargs.out}_perturb.csv', index = False)
    with open(f'{cmdargs.out}.json', 'w', encoding = 'utf-8') as f:
        json.dump(meta, f, indent = 4)

    # figure: hard-tower error along the ODE path; perturbation response
    (fig, axes) = plt.subplots(1, 3, figsize = (10, 2.9))
    for (k, label) in enumerate(labels):
        d = tr[tr.model == label]
        axes[0].plot(d.t, d.hard_share_err, color = f'C{k}', marker = 'o', ms = 2,
                     label = f'{label}: ODE path')
        if k == 0:
            axes[0].plot(d.t, d.ref_share_err, color = 'k', ls = ':',
                         label = 'clean interpolant')
        axes[0].plot(d.t, -d.hard_share_below, color = f'C{k}', ls = '--', lw = 0.8)
        p = pe[(pe.model == label) & (pe.delta == 0) & (pe.direction == 'hard')]
        axes[1].plot(p.t0, p.hard_share_err, color = f'C{k}', marker = 's', label = label)
        for (dname, ls) in [ ('hard', '-'), ('random', '--') ]:
            q = pe[(pe.model == label) & (pe.delta == 0) & (pe.direction == dname)]
            axes[2].plot(q.t0, q.response_hard, color = f'C{k}', ls = ls, marker = 'o',
                         label = f'{label}, {dname}')
    axes[1].axhline(meta[labels[0]]['endpoint_hard_share_err'], color = 'C0', lw = 0.6, ls = ':')
    if len(labels) > 1:
        axes[1].axhline(meta[labels[1]]['endpoint_hard_share_err'], color = 'C1', lw = 0.6,
                        ls = ':')
    axes[0].set_title('hard towers (S > 5 GeV): sum(B(x_t) - B) / sum(S)\n'
                      '(dashed: minus the share of towers below B)', fontsize = 7)
    axes[0].set_xlabel('t')
    axes[1].set_title('endpoint leakage from the clean interpolant at t0\n'
                      '(dotted: from t = 0, the ordinary ODE)', fontsize = 7)
    axes[1].set_xlabel('t0')
    axes[2].set_title('endpoint response to a start displacement\n'
                      '(1: kept, 0: removed; hard towers)', fontsize = 7)
    axes[2].set_xlabel('t0')
    for ax in axes:
        ax.grid(alpha = 0.3)
        ax.axhline(0, color = 'k', lw = 0.5)
        ax.legend(fontsize = 5.5)
    fig.tight_layout()
    fig.savefig(f'{cmdargs.out}.png', dpi = 150)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(pd.DataFrame(solver_rows).round(4).to_string(index = False))
        print(pe.round(4).to_string(index = False))
    print(json.dumps(meta, indent = 2))

if __name__ == '__main__':
    main()
