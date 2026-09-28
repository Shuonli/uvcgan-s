#!/usr/bin/env python
"""Checks of the noisy training path (fm_train.py --path sine, FLOW_NOTES.md,
"Paired noisy-interpolant pilot") before any GPU training, on real paired
training batches of the background-only arm (otcfm1_paired), CPU:

1. eta = 0 reproduces the straight path: the same times, states x_t and
   targets u_t (bit for bit) and the same global RNG state afterwards;
   eta = 0.1 keeps the times and the global RNG stream too.
2. The endpoints are unchanged: x_0 = a and x_1 = b exactly.
3. u_t is the time derivative of x_t at fixed a, b and eps: central
   differences in float64.

    sine_path_check.py [--out docs/flow/bench/noisy/path_check.json]
"""

import argparse
import json
import os

import numpy as np
import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

import fm_common as fc

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Sine path checks')
    parser.add_argument('--n', type = int, default = 64)
    parser.add_argument('--eta', type = float, default = 0.1)
    parser.add_argument('--out', default = 'docs/flow/bench/noisy/path_check.json')
    return parser.parse_args()

def method(norm, path, eta, seed = 7):
    m = fc.Method('otcfm1_paired', norm, None, path = path, eta = eta)
    m.path_gen = torch.Generator().manual_seed(seed)
    return m

def draw(m, x0, x1, seed, t = None):
    """One training draw of (t, x_t, u_t) from a fixed global RNG state, as
    Method.loss makes it; also the global RNG state afterwards."""
    torch.manual_seed(seed)
    (t, xt, ut) = ConditionalFlowMatcher.sample_location_and_conditional_flow(
        m.matcher, x0, x1, t = t)
    if m.path == 'sine':
        (xt, ut) = m.sine_path(t, xt, ut)
    return (t, xt, ut, torch.get_rng_state())

def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    norm  = fc.Norm.load_or_fit(fc.Norm.path(fc.BIAS), bias = fc.BIAS)
    pairs = np.load(fc.cache_path('embed_pairs'), mmap_mode = 'r')[:cmdargs.n]
    batch = { 'embed_pairs' : torch.from_numpy(np.array(pairs)).float() }
    res   = {}

    # 1. eta = 0 against the straight path, and the RNG streams
    straight = method(norm, 'straight', 0.0)
    (x0, x1, _) = straight.endpoints(batch)
    (t_s, x_s, u_s, rng_s) = draw(straight, x0, x1, 123)
    (t_0, x_0, u_0, rng_0) = draw(method(norm, 'sine', 0.0), x0, x1, 123)
    (t_e, x_e, u_e, rng_e) = draw(method(norm, 'sine', cmdargs.eta), x0, x1, 123)
    res['eta0_same_t']  = bool(torch.equal(t_s, t_0))
    res['eta0_same_xt'] = bool(torch.equal(x_s, x_0))
    res['eta0_same_ut'] = bool(torch.equal(u_s, u_0))
    res['eta0_same_global_rng'] = bool(torch.equal(rng_s, rng_0))
    res['eta_same_t'] = bool(torch.equal(t_s, t_e))
    res['eta_same_global_rng'] = bool(torch.equal(rng_s, rng_e))
    # the added noise: gamma(t) eps, its sd against eta sin(pi t)
    tt = t_s.reshape(-1, 1, 1, 1)
    gamma = cmdargs.eta * torch.sin(torch.pi * tt)
    noise = (x_e - x_s) / gamma.clamp(min = 1e-6)
    res['noise_over_gamma_sd'] = float(noise.std())
    res['ut_minus_straight_over_dgamma_equals_noise'] = float(
        ((u_e - u_s) / (cmdargs.eta * torch.pi * torch.cos(torch.pi * tt))
         - noise).abs().max())

    # 2. endpoints, eta > 0
    for (tv, target, name) in [ (0.0, x0, 't0'), (1.0, x1, 't1') ]:
        t = torch.full((len(x0),), tv)
        (_, xt, _, _) = draw(method(norm, 'sine', cmdargs.eta), x0, x1, 5, t)
        res[f'endpoint_{name}_exact'] = bool(torch.equal(xt, target))
        res[f'endpoint_{name}_max_abs'] = float((xt - target).abs().max())

    # 3. u_t against central differences of x_t (float64, fixed a, b, eps)
    (a, b) = (x0.double(), x1.double())
    h = 1e-5
    worst = 0.0
    for tv in [ 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98 ]:
        def at(t_value):
            # the same eps at every t: the generator reseeded
            m = method(norm, 'sine', cmdargs.eta, seed = 11)
            t = torch.full((len(a),), t_value, dtype = torch.float64)
            (_, xt, ut, _) = draw(m, a, b, 5, t)
            return (xt, ut)
        (xp, _) = at(tv + h)
        (xm, _) = at(tv - h)
        (_, u)  = at(tv)
        err = float(((xp - xm) / (2 * h) - u).abs().max())
        res[f'fd_max_abs_t{tv}'] = err
        worst = max(worst, err)
    res['fd_max_abs'] = worst
    res['fd_scale_max_abs_u'] = float(u.abs().max())

    res['passes'] = bool(
        res['eta0_same_t'] and res['eta0_same_xt'] and res['eta0_same_ut']
        and res['eta0_same_global_rng'] and res['eta_same_t']
        and res['eta_same_global_rng'] and res['endpoint_t0_exact']
        and res['endpoint_t1_exact'] and worst < 1e-6
        and abs(res['noise_over_gamma_sd'] - 1) < 0.05)
    os.makedirs(os.path.dirname(cmdargs.out), exist_ok = True)
    with open(cmdargs.out, 'w', encoding = 'utf-8') as f:
        json.dump({ 'eta' : cmdargs.eta, 'n' : cmdargs.n, **res }, f, indent = 4)
    print(json.dumps(res, indent = 2))

if __name__ == '__main__':
    main()
