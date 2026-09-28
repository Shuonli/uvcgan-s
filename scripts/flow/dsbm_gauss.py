#!/usr/bin/env python
"""Gaussian sanity check of the alpha-DSBM code (dsbm.py) before any jet run
(FLOW_NOTES.md, "alpha-DSBM closure test").

pi0 = N(0, s0^2 I_d), pi1 = N(0, s1^2 I_d), reference process sqrt(eps) B,
the same noise convention as the jets (bridge variance eps t (1 - t), SDE
increments sqrt(eps h) z). The static Schrodinger bridge is the entropic OT
coupling of (1/2)|x - y|^2 with regularisation eps (paper eq. 1-2). For these
Gaussians it is Gaussian with, per dimension, Var X0 = s0^2, Var X1 = s1^2
and cross-covariance

    c* = (sqrt(4 s0^2 s1^2 + eps^2) - eps) / 2

(maximise -(s0^2 + s1^2 - 2c)/(2 eps) + (1/2) log(s0^2 s1^2 - c^2) over c:
c^2 + eps c - s0^2 s1^2 = 0). References: OT, c = s0 s1; independent, 0.

The run uses dsbm.py's own functions (bridge_loss via pretrain_loss and
refine_loss, rollout, sample_sde, optimizer_step) with the Gaussian network
of the paper's Appendix K.2 (BridgeMLP), in the same two stages: bridge
matching on independent pairs, then online alpha-DSBM with EMA rollouts.
Every --every updates, on 10k fresh samples with the SDE sampler (--steps
Euler-Maruyama steps, the jets' endpoint convention): the forward coupling
Cov(X0, X1_hat), the backward one Cov(X0_hat, X1) (diagonal mean, largest
off-diagonal), and the generated marginals. At the end, --steps against
2 x --steps with coupled Brownian increments.

Pass: both couplings within 5% of c* and both generated variances within 5%
of the target. A marginal-matching but wrong coupling fails, e.g. the
pretrained model's (it should be far from c*), or X1 = -(s1/s0) X0 (c = -s0
s1, both marginals exact), also scored.

    dsbm_gauss.py [--dim 16 --s0 1 --s1 2 --eps 1.0] [--out docs/flow/dsbm/gauss]
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

import dsbm

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'alpha-DSBM Gaussian check')
    parser.add_argument('--dim', type = int, default = 16)
    parser.add_argument('--s0', type = float, default = 1.0)
    parser.add_argument('--s1', type = float, default = 2.0)
    parser.add_argument('--eps', type = float, default = 1.0)
    parser.add_argument('--steps', type = int, default = 30)
    parser.add_argument('--batch', type = int, default = 256)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--ema', type = float, default = 0.999)
    parser.add_argument('--pretrain', type = int, default = 10000)
    parser.add_argument('--refine', type = int, default = 20000)
    parser.add_argument('--every', type = int, default = 2000)
    parser.add_argument('--n-eval', type = int, default = 10000)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--warmup', type = int, default = 0,
        help = 'linear learning-rate warm-up (updates) at the start of each stage')
    parser.add_argument('--refine-lr', type = float, default = None)
    parser.add_argument('--rollout-net', default = 'ema', choices = [ 'ema', 'online' ])
    parser.add_argument('--model', default = 'mlp', choices = [ 'mlp', 'linear' ])
    parser.add_argument('--precond-scales', action = 'store_true',
        help = "precondition with the endpoints' variances s0^2, s1^2 (else 1, 1)")
    parser.add_argument('--init-coupling', default = 'independent',
                        choices = [ 'independent', 'anti' ],
        help = "pretraining pairs: independent (as the jets), or the paper's"
               ' robustness test X1 = -(s1/s0) X0 (a wrong starting coupling)')
    parser.add_argument('--out', default = 'docs/flow/dsbm/gauss')
    return parser.parse_args()

class BridgeLinear(dsbm.Preconditioned):
    """Diagnostic model: raw(s, u, x) = g(s, u) x with a scalar slope g from
    a small MLP of the time and direction features, so endpoint and drift are
    exactly linear in x: the function class of this Gaussian problem, with
    nothing to extrapolate."""

    def __init__(self, eps, hidden = 256):
        super().__init__()
        self.eps = eps
        self.g = torch.nn.Sequential(
            torch.nn.Linear(2 * dsbm.EMB, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1))

    def raw(self, s, u, x):
        return self.g(torch.cat([ dsbm.embed(u), dsbm.embed(s) ], dim = 1)) * x

def exact_reference(s0, s1, eps, n_steps, init = 'independent', iters = 60):
    """Network-free values of the same procedure in 1-D (the problem is
    isotropic): linear drifts from exact Markovian projections, the same
    discretised sampler (n_steps Euler-Maruyama, the last step returning the
    endpoint). Returns the pretrained stage (projection of the initial
    coupling) and the fixed point of the bidirectional iteration."""
    def slope(a, b, c, t, direction):
        v = (1 - t)**2 * a + t**2 * b + 2 * t * (1 - t) * c + eps * t * (1 - t)
        return ((1 - t) * c + t * b) / v if direction == 1 else ((1 - t) * a + t * c) / v
    def sample(a, b, c, direction, var_start):
        h = 1 / n_steps
        (var, cov) = (var_start, var_start)
        for k in range(n_steps):
            u = k * h
            m = slope(a, b, c, u if direction == 1 else 1 - u, direction)
            if k == n_steps - 1:
                return (m * m * var, m * cov)
            g = 1 + h * (m - 1) / (1 - u)
            (var, cov) = (g * g * var + eps * h, g * cov)
        return (var, cov)
    c0 = 0.0 if init == 'independent' else -s0 * s1
    (cf, cb) = ([ s0**2, s1**2, c0 ], [ s0**2, s1**2, c0 ])
    out = {}
    for n in range(iters):
        (vb, kb) = sample(*cb, 0, s1**2)
        (vf, kf) = sample(*cf, 1, s0**2)
        if n == 0:
            out['pretrained'] = { 'c_fwd' : kf, 'c_bwd' : kb, 'var_x1_hat' : vf,
                                  'var_x0_hat' : vb }
        (cf, cb) = ([ 0.5 * (x + y) for (x, y) in zip(cf, (vb, s1**2, kb)) ],
                    [ 0.5 * (x + y) for (x, y) in zip(cb, (s0**2, vf, kf)) ])
    out['fixed_point'] = { 'c_fwd' : kf, 'c_bwd' : kb, 'var_x1_hat' : vf, 'var_x0_hat' : vb }
    return out

def coupling(x, y):
    """(diagonal mean, largest off-diagonal) of Cov(x, y)."""
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    cov = (xc.T @ yc) / (len(x) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return (float(torch.diag(cov).mean()), float(off.abs().max()))

@torch.no_grad()
def measure(model, cmdargs, gen, increments = None):
    n = cmdargs.n_eval
    x0 = cmdargs.s0 * torch.randn(n, cmdargs.dim, device = 'cuda', generator = gen)
    x1 = cmdargs.s1 * torch.randn(n, cmdargs.dim, device = 'cuda', generator = gen)
    inc = (None, None) if increments is None else increments
    x1_hat = dsbm.sample_sde(model, x0, 1, cmdargs.eps, cmdargs.steps, gen, inc[0])
    x0_hat = dsbm.sample_sde(model, x1, 0, cmdargs.eps, cmdargs.steps, gen, inc[1])
    (c_f, off_f) = coupling(x0, x1_hat)
    (c_b, off_b) = coupling(x0_hat, x1)
    return { 'c_fwd' : c_f, 'offdiag_fwd' : off_f, 'c_bwd' : c_b, 'offdiag_bwd' : off_b,
             'var_x1_hat' : float(x1_hat.var(0).mean()),
             'var_x0_hat' : float(x0_hat.var(0).mean()),
             'mean_x1_hat' : float(x1_hat.mean(0).abs().max()),
             'mean_x0_hat' : float(x0_hat.mean(0).abs().max()) }, (x1_hat, x0_hat)

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    os.makedirs(os.path.dirname(cmdargs.out) or '.', exist_ok = True)
    torch.manual_seed(cmdargs.seed)
    dev = torch.device('cuda')
    c_star = 0.5 * (math.sqrt(4 * cmdargs.s0**2 * cmdargs.s1**2 + cmdargs.eps**2) - cmdargs.eps)
    print(f'analytic SB cross-covariance c* = {c_star:.4f} (OT {cmdargs.s0 * cmdargs.s1:.4f},'
          ' independent 0)', flush = True)

    model = (dsbm.BridgeMLP(cmdargs.dim, cmdargs.eps) if cmdargs.model == 'mlp'
             else BridgeLinear(cmdargs.eps)).to(dev)
    if cmdargs.precond_scales:
        (model.var0, model.var1) = (cmdargs.s0**2, cmdargs.s1**2)
    ema   = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    g_data = torch.Generator(device = dev).manual_seed(1)
    g_loss = torch.Generator(device = dev).manual_seed(2)
    g_roll = torch.Generator(device = dev).manual_seed(3)
    b = cmdargs.batch // 2
    rows = []
    t0 = time.perf_counter()

    def record(stage, update):
        (m, _) = measure(ema, cmdargs, torch.Generator(device = dev).manual_seed(99))
        rows.append({ 'stage' : stage, 'update' : update, 'c_star' : c_star, **m })
        print(f"{stage:9s} {update:6d}  c_fwd {m['c_fwd']:.4f}  c_bwd {m['c_bwd']:.4f}"
              f"  (c* {c_star:.4f})  var X1_hat {m['var_x1_hat']:.3f} (target"
              f" {cmdargs.s1**2:.3f})  var X0_hat {m['var_x0_hat']:.3f} (target"
              f" {cmdargs.s0**2:.3f})  [{time.perf_counter() - t0:.0f} s]", flush = True)

    for stage in ('pretrain', 'refine'):
        # fresh optimiser in each stage (Appendix K)
        lr_stage = cmdargs.refine_lr if (stage == 'refine' and cmdargs.refine_lr) \
            else cmdargs.lr
        opt = torch.optim.Adam(model.parameters(), lr = lr_stage)
        n_up = cmdargs.pretrain if stage == 'pretrain' else cmdargs.refine
        for update in range(1, n_up + 1):
            if cmdargs.warmup:
                for group in opt.param_groups:
                    group['lr'] = lr_stage * min(1.0, update / cmdargs.warmup)
            if stage == 'pretrain':
                x0 = cmdargs.s0 * torch.randn(cmdargs.batch, cmdargs.dim, device = dev,
                                              generator = g_data)
                x1 = cmdargs.s1 * torch.randn(cmdargs.batch, cmdargs.dim, device = dev,
                                              generator = g_data)
                if cmdargs.init_coupling == 'anti':
                    x1 = -(cmdargs.s1 / cmdargs.s0) * x0
                (loss, _, _) = dsbm.pretrain_loss(model, x0, x1, cmdargs.eps, g_loss)
            else:
                x0 = cmdargs.s0 * torch.randn(b, cmdargs.dim, device = dev, generator = g_data)
                x1 = cmdargs.s1 * torch.randn(b, cmdargs.dim, device = dev, generator = g_data)
                sampler = ema if cmdargs.rollout_net == 'ema' else model
                (x0_hat, x1_hat) = dsbm.rollout(sampler, x0, x1, cmdargs.eps, cmdargs.steps,
                                                g_roll)
                (loss, _, _) = dsbm.refine_loss(model, x0, x1, x0_hat, x1_hat,
                                                cmdargs.eps, g_loss)
            dsbm.optimizer_step(model, ema, opt, loss, 1.0, cmdargs.ema)
            if update % cmdargs.every == 0:
                record(stage, update)

    # integration error: N against 2N steps with coupled Brownian increments
    g = torch.Generator(device = dev).manual_seed(7)
    fine = [ math.sqrt(0.5 / cmdargs.steps) * torch.randn(
        2 * cmdargs.steps, cmdargs.n_eval, cmdargs.dim, device = dev, generator = g)
        for _ in range(2) ]
    coarse = [ dsbm.coarsen(f) for f in fine ]
    g_a = torch.Generator(device = dev).manual_seed(5)
    (m_n, (x1_n, x0_n)) = measure(ema, cmdargs, g_a, coarse)
    steps = cmdargs.steps
    cmdargs.steps = 2 * steps
    g_a = torch.Generator(device = dev).manual_seed(5)
    (m_2n, (x1_2n, x0_2n)) = measure(ema, cmdargs, g_a, fine)
    cmdargs.steps = steps
    step_check = {
        'rms_diff_fwd' : float((x1_n - x1_2n).pow(2).mean().sqrt()),
        'rms_diff_bwd' : float((x0_n - x0_2n).pow(2).mean().sqrt()),
        f'c_fwd_{steps}' : m_n['c_fwd'], f'c_fwd_{2 * steps}' : m_2n['c_fwd'],
        f'c_bwd_{steps}' : m_n['c_bwd'], f'c_bwd_{2 * steps}' : m_2n['c_bwd'],
    }

    # a coupling with both marginals exact but the wrong sign
    x0 = cmdargs.s0 * torch.randn(cmdargs.n_eval, cmdargs.dim, device = dev)
    (c_wrong, _) = coupling(x0, -(cmdargs.s1 / cmdargs.s0) * x0)

    df = pd.DataFrame(rows)
    df.to_csv(f'{cmdargs.out}.csv', index = False)
    final = rows[-1]
    pre   = [ r for r in rows if r['stage'] == 'pretrain' ][-1]
    def ok(r):
        return (abs(r['c_fwd'] - c_star) / c_star < 0.05
                and abs(r['c_bwd'] - c_star) / c_star < 0.05
                and abs(r['var_x1_hat'] / cmdargs.s1**2 - 1) < 0.05
                and abs(r['var_x0_hat'] / cmdargs.s0**2 - 1) < 0.05)
    exact = { n : exact_reference(cmdargs.s0, cmdargs.s1, cmdargs.eps, n,
                                  cmdargs.init_coupling)
              for n in (cmdargs.steps, 2 * cmdargs.steps) }
    result = {
        'config' : vars(cmdargs), 'c_star' : c_star, 'c_ot' : cmdargs.s0 * cmdargs.s1,
        'pretrained' : pre, 'refined' : final, 'pretrained_passes' : ok(pre),
        'refined_passes' : ok(final), 'wrong_sign_coupling_c' : c_wrong,
        'wrong_sign_passes' : False, 'step_check' : step_check,
        'exact_network_free' : exact,
        'seconds' : time.perf_counter() - t0,
    }
    with open(f'{cmdargs.out}.json', 'w', encoding = 'utf-8') as f:
        json.dump(result, f, indent = 4)
    print(json.dumps({ k : v for (k, v) in result.items() if k != 'config' }, indent = 2))

if __name__ == '__main__':
    main()
