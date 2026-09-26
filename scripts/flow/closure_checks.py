#!/usr/bin/env python
"""Checks of the jet -> jet flow pipeline, without training a model
(FLOW_NOTES.md, "Pipeline checks").

On the first 512 test pairs (J, T(J)) of the closure test:

1. The normalisation round trip: energy(z(J)) against J, per tower, for the
   zero towers and the total energy. Done in float32, and for the float16
   training cache.
2. A velocity field that is exactly zero, through the inference path
   (fm_common.JetMapper, 4 Euler steps and 32 midpoint evaluations). It must
   return J.
3. The trained unpaired flows (`--runs`, their selected checkpoints), with
   raw outputs taken before the clip at 0:
   - signed per-tower errors F(J) - T(J), in the cone towers where T(J) is
     exactly zero ("truth-empty") and in the others;
   - the halo, the energy of the truth-empty towers, before and after the
     clip;
   - the energy the output puts on the canvas outside the cone, which the
     evaluation ignores.
4. The probability path: TorchCFM's x_t and u_t at t = 0 and 1 for a
   training pair, against the state the inference starts from.

    closure_checks.py [--runs closure_l2_s0,closure_se1_s0] [--n 512]
"""

import argparse
import os

import numpy as np
import torch

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

import fm_common as fc
from closure_data import OFFSET, cone_mask
from closure_eval import load_pairs, selected_step, checkpoints, ckpt_step

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Closure pipeline checks')
    parser.add_argument('--runs', default = 'closure_l2_s0,closure_se1_s0')
    parser.add_argument('--n', type = int, default = 512)
    return parser.parse_args()

class ZeroVelocity(torch.nn.Module):
    def forward(self, t, x):      # pylint: disable=unused-argument
        return torch.zeros_like(x)

def split(canvas, mask):
    """(cone window, everything else) of (N, 16, 16) canvases."""
    inside = torch.zeros_like(canvas, dtype = torch.bool)
    inside[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9] = mask
    return (canvas * inside, canvas * ~inside)

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    mask    = torch.from_numpy(cone_mask()).to(device)
    pairs   = load_pairs('test', cmdargs.n)
    src = torch.from_numpy(pairs['src']).to(device)
    tgt = torch.from_numpy(pairs['tgt']).to(device)

    with open(fc.Norm.path(method = 'jetflow'), 'r', encoding = 'utf-8') as f:
        import json          # pylint: disable=import-outside-toplevel
        norm = fc.Norm(json.load(f))
    method = fc.Method('jetflow', norm)

    print('1. normalisation round trip (float32 test jets)')
    back = norm.energy(norm.z(src, 'jet'), 'jet')
    zero = src == 0
    print(f'   max |E - J| per tower {float((back - src).abs().max()):.2e} GeV;'
          f' zero towers: max |E| {float(back[zero].abs().max()):.2e} GeV;'
          f' total energy: max relative error'
          f' {float(((back.sum((1, 2)) - src.sum((1, 2))) / src.sum((1, 2))).abs().max()):.2e}')
    cache = np.load(fc.cache_path('closure_src'), mmap_mode = 'r')
    c16 = torch.from_numpy(cache[:cmdargs.n].astype(np.float32)).to(device)
    back16 = norm.energy(norm.z(c16, 'jet'), 'jet')
    print(f'   float16 training cache: max |E - J| per tower'
          f' {float((back16 - c16).abs().max()):.2e} GeV, zero towers exactly'
          f' zero in the cache: {bool((c16[c16 == 0] == 0).all())}')

    print('2. zero velocity through the inference path')
    for (nfe, solver) in [ (4, 'euler'), (32, 'midpoint') ]:
        for clip in (True, False):
            out = fc.JetMapper(method, ZeroVelocity(), nfe, solver, clip)(src)
            print(f'   {solver} {nfe}, clip {clip}: max |F(J) - J|'
                  f' {float((out - src).abs().max()):.2e} GeV')

    print('3. trained flows: signed errors before the clip, halo')
    (t_in, _) = split(tgt, mask)
    empty = (t_in == 0) & split(torch.ones_like(tgt), mask)[0].bool()
    occupied = (t_in > 0)
    e_t = t_in.sum((1, 2))
    print(f'   truth-empty cone towers per jet: {float(empty.sum((1, 2)).float().mean()):.1f}'
          f' of 53 (T(J) exactly 0); occupied {float(occupied.sum((1, 2)).float().mean()):.1f}')
    for run in cmdargs.runs.split(','):
        path = os.path.join(fc.out_root(), run)
        step = selected_step(path)
        ckpt = [ c for c in checkpoints(path) if ckpt_step(c) == step ][0]
        (meth, net, _, _) = fc.load_run(path, ckpt, device, 'ema')
        for (nfe, solver) in [ (4, 'euler'), (32, 'midpoint') ]:
            raw = fc.JetMapper(meth, net, nfe, solver, clip = False)(src)
            (r_in, r_out) = split(raw, mask)
            err = (raw - tgt)
            e_empty = err[empty]
            halo_signed  = (r_in * empty).sum((1, 2))
            halo_clipped = (r_in.clamp(min = 0) * empty).sum((1, 2))
            occ_err = err[occupied]
            e_raw   = r_in.sum((1, 2))
            e_clip  = r_in.clamp(min = 0).sum((1, 2))
            print(f'   {run} {solver} {nfe}:')
            print(f'     truth-empty towers: signed error mean {float(e_empty.mean()):+.4f},'
                  f' sd {float(e_empty.std()):.4f} GeV; share < 0'
                  f' {float((e_empty < 0).float().mean()):.2f}; min {float(e_empty.min()):+.3f}')
            print(f'     halo per jet: signed {float(halo_signed.mean()):+.3f} GeV,'
                  f' after the clip {float(halo_clipped.mean()):.3f} GeV'
                  f' ({float((halo_clipped / e_t).mean()):.3f} of E(T(J)))')
            print(f'     occupied towers: signed error mean {float(occ_err.mean()):+.4f}'
                  f' GeV, rms {float(occ_err.pow(2).mean().sqrt()):.3f} GeV')
            print(f'     cone energy / E(T(J)): raw {float((e_raw / e_t).mean()):.3f},'
                  f' clipped {float((e_clip / e_t).mean()):.3f}; energy outside'
                  f' the cone {float(r_out.sum((1, 2)).mean()):+.3f} GeV'
                  f' (clipped {float(r_out.clamp(min = 0).sum((1, 2)).mean()):.3f})')

    print('3b. per-tower signed error by true tower energy T (cone towers),'
          ' mean / rms in GeV, and the share of the jet energy in each class')
    bins = [ (0, 1e-9), (1e-9, 0.05), (0.05, 0.2), (0.2, 1), (1, 5), (5, 1e9) ]
    names = [ '=0', '<0.05', '0.05-0.2', '0.2-1', '1-5', '>5' ]
    outputs = { 'identity J': src }
    for run in cmdargs.runs.split(','):
        path = os.path.join(fc.out_root(), run)
        step = selected_step(path)
        ckpt = [ c for c in checkpoints(path) if ckpt_step(c) == step ][0]
        (meth, net, _, _) = fc.load_run(path, ckpt, device, 'ema')
        for (nfe, solver) in [ (4, 'euler'), (32, 'midpoint') ]:
            outputs[f'{run} {solver}{nfe}'] = fc.JetMapper(
                meth, net, nfe, solver, clip = False)(src)
    print('   class            ' + ''.join(f'{n:>16s}' for n in names))
    share = [ float(((t_in >= lo) & (t_in < hi) & (t_in > 0) | ((lo == 0) & empty))
                    .float().sum()) for (lo, hi) in bins ]
    print('   towers / jet     ' + ''.join(
        f'{float((((t_in >= lo) & (t_in < hi) & occupied) if lo > 0 else empty).sum()) / len(src):16.1f}'
        for (lo, hi) in bins))
    print('   T energy share   ' + ''.join(
        f'{float((t_in * ((t_in >= lo) & (t_in < hi))).sum() / t_in.sum()):16.3f}'
        for (lo, hi) in bins))
    for (name, out) in outputs.items():
        err = split(out, mask)[0] - t_in
        cells = []
        for (lo, hi) in bins:
            sel = empty if lo == 0 else ((t_in >= lo) & (t_in < hi) & occupied)
            e = err[sel]
            cells.append(f'{float(e.mean()):+7.3f}/{float(e.pow(2).mean().sqrt()):.3f}')
        print(f'   {name[:30]:30s}' + ''.join(f'{c:>16s}' for c in cells))
    del share

    print('4. probability path')
    x0 = method.source(src[:4])
    x1 = method.source(tgt[:4])
    cfm = ConditionalFlowMatcher(sigma = 0.0)
    for t in (0.0, 1.0):
        tt = torch.full((4,), t, device = device)
        (_, xt, ut) = cfm.sample_location_and_conditional_flow(x0, x1, tt)
        print(f'   t = {t}: max |x_t - x_{int(t)}| {float((xt - (x1 if t else x0)).abs().max()):.2e},'
              f' max |u_t - (x1 - x0)| {float((ut - (x1 - x0)).abs().max()):.2e}')
    print(f'   sigma of the path: {method.matcher.sigma} (training, jetflow);'
          f' inference starts at source(J) = z(J): identical to x_0')

if __name__ == '__main__':
    main()
