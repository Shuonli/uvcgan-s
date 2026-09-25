#!/usr/bin/env python
"""Checks before the pilot: shapes, finite gradients, couplings, solver
step counts, a checkpoint round trip, and the cost of a step.

    smoke_checks.py [--batch 256]
"""

import argparse
import os
import tempfile
import time

import numpy as np
import torch

import fm_common as fc

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Pre-pilot checks')
    parser.add_argument('--batch', type = int, default = 256)
    parser.add_argument('--channels', default = '64,96,128')
    return parser.parse_args()

def check(cond, what):
    print(('  ok    ' if cond else '  FAIL  ') + what, flush = True)
    if not cond:
        raise SystemExit(1)

def timed(fn, repeats):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats

def main():
    # pylint: disable=too-many-locals,too-many-statements
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name())

    t0   = time.perf_counter()
    norm = fc.Norm.load_or_fit(
        os.path.join(fc.out_root(), 'norm_n20000_seed0.json')
    )
    print(f'norm ({time.perf_counter() - t0:.1f} s):', norm.stats)

    # data: shapes, and the cost of drawing a batch of all three domains
    t0   = time.perf_counter()
    data = fc.GPUData(fc.DOMAINS['otcfm'], device, 0)
    print(f'data in GPU memory after {time.perf_counter() - t0:.0f} s:',
          data.sizes())
    batch = data.batch(cmdargs.batch)
    for (k, v) in batch.items():
        check(v.shape == (cmdargs.batch, *fc.SHAPE) and v.dtype == torch.float32,
              f'{k}: {tuple(v.shape)} {v.dtype}, min {v.min():.3f},'
              f' mean {v.mean():.3f}, max {v.max():.1f}')
        check(bool((v >= 0).all()), f'{k}: energies >= 0')

    def draw():
        data.batch(cmdargs.batch)
    print(f'drawing a batch of 3 x {cmdargs.batch}: {1000 * timed(draw, 50):.2f} ms')

    batch = data.batch(cmdargs.batch)
    del data
    torch.cuda.empty_cache()

    for name in fc.METHODS:
        print(f'\n{name}')
        method = fc.Method(name, norm)
        torch.manual_seed(0)
        net = fc.construct_net(name).to(device)
        print(f'  parameters {fc.count_params(net) / 1e6:.2f} M')

        (x0, x1, cond) = method.endpoints(batch)
        check(x1.shape == (cmdargs.batch, 2, *fc.SHAPE), f'x1 {tuple(x1.shape)}')
        if x0 is not None:
            check(x0.shape == x1.shape, f'x0 {tuple(x0.shape)}')
        if cond is not None:
            check(cond.shape == (cmdargs.batch, 1, *fc.SHAPE),
                  f'cond {tuple(cond.shape)}')
        for (k, v) in [ ('x0', x0), ('x1', x1), ('cond', cond) ]:
            if v is not None:
                check(bool(torch.isfinite(v).all()),
                      f'{k} finite, mean {v.mean():+.3f} std {v.std():.3f}')

        if method.coupled:
            plan = method.matcher.ot_sampler.get_map(x0, x1)
            check(abs(plan.sum() - 1) < 1e-3, f'plan mass {plan.sum():.6f}')
            nnz = (plan > 1e-3 / cmdargs.batch**2).sum(axis = 1)
            rowp = plan / plan.sum(axis = 1, keepdims = True)
            ent  = -(rowp * np.log(np.clip(rowp, 1e-300, None))).sum(axis = 1)
            print(f'  plan: partners per row {np.exp(ent).mean():.2f}'
                  f' (1 = a permutation), nonzeros per row {nnz.mean():.1f}')
            if name == 'otcfm':
                check(bool((nnz == 1).all()), 'exact plan is a permutation')
            else:
                print(f'  sinkhorn marginal error {method.matcher.ot_sampler.last_err:.1e}')
                # the same plan from POT's log-domain solver, run to 1e-8
                c64 = torch.cdist(x0.flatten(1).double(),
                                  x1.flatten(1).double()) ** 2
                a   = torch.full((cmdargs.batch,), 1 / cmdargs.batch,
                                 dtype = c64.dtype, device = device)
                ref = fc.pot.sinkhorn(
                    a, a, c64, method.matcher.ot_sampler.reg,
                    method = 'sinkhorn_log', numItermax = 40000,
                    stopThr = 1e-8, warn = False
                ).cpu().numpy()
                check(np.abs(plan - ref).sum() < 1e-2,
                      f'plan against POT: L1 {np.abs(plan - ref).sum():.1e}')

            def couple():
                method.couple(x0, x1)
            print(f'  coupling {1000 * timed(couple, 10):.1f} ms per batch')
            (x0, x1) = method.couple(x0, x1)

        loss = method.loss(net, x0, x1, cond)
        loss.backward()
        grads = [ p.grad for p in net.parameters() ]
        check(all(g is not None for g in grads), 'every parameter has a gradient')
        check(all(bool(torch.isfinite(g).all()) for g in grads),
              f'gradients finite, loss {float(loss):.4f}')
        gnorm = torch.norm(torch.stack([ g.norm() for g in grads ]))
        print(f'  gradient norm {float(gnorm):.3f}')

        opt = torch.optim.Adam(net.parameters(), lr = 2e-4)

        def step():
            l = method.loss(net, x0, x1, cond)
            opt.zero_grad(set_to_none = True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        print(f'  train step {1000 * timed(step, 20):.1f} ms'
              f' (batch {cmdargs.batch}, no coupling, no data)')
        print(f'  peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB')

        # solver: exactly nfe evaluations
        calls = [ 0 ]
        hook  = net.register_forward_hook(
            lambda *_: calls.__setitem__(0, calls[0] + 1)
        )
        dec = fc.Decomposer(method, net, nfe = 8, solver = 'midpoint')
        y   = dec(batch['embed'][:16].unsqueeze(1))
        hook.remove()
        want = 1 if name == 'regress' else 8
        check(calls[0] == want, f'decomposition took {calls[0]} evaluations')
        check(y.shape == (16, 2, *fc.SHAPE) and bool(torch.isfinite(y).all()),
              f'decomposition {tuple(y.shape)}, finite')

        # checkpoint round trip
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'c.pt')
            fc.save_checkpoint(path, raw = net.state_dict())
            net2 = fc.construct_net(name).to(device)
            net2.load_state_dict(torch.load(path, map_location = device)['raw'])
            net.eval()
            net2.eval()
            t   = torch.rand(8, device = device)
            inp = torch.randn(8, net.in_channels, *fc.SHAPE, device = device)
            with torch.no_grad():
                diff = (net(t, inp) - net2(t, inp)).abs().max()
            check(float(diff) == 0.0, f'checkpoint round trip, max diff {float(diff)}')

        torch.cuda.reset_peak_memory_stats()

    # the width of the network against the cost of a step
    print('\nwidth against step time (condcfm, batch', cmdargs.batch, ')')
    method = fc.Method('condcfm', norm)
    (x0, x1, cond) = method.endpoints(batch)
    for ch in [ int(x) for x in cmdargs.channels.split(',') ]:
        net = fc.construct_net('condcfm', channels = ch).to(device)
        opt = torch.optim.Adam(net.parameters(), lr = 2e-4)

        def step():
            l = method.loss(net, x0, x1, cond)
            opt.zero_grad(set_to_none = True)
            l.backward()
            opt.step()
        dt = timed(step, 20)
        print(f'  channels {ch:4d}: {fc.count_params(net) / 1e6:6.2f} M parameters,'
              f' {1000 * dt:6.1f} ms per step')

    print('\nall checks passed')

if __name__ == '__main__':
    main()
