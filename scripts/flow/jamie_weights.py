#!/usr/bin/env python
"""Weights and rollout accuracy of a toy continuation's extra terms
(FLOW_NOTES.md, "Jamie's toy exercises with OT flow matching"), on training
data at the base checkpoint, before the continuation; then frozen.

    jamie_weights.py BASE_RUN --label NAME --terms abs,bal,ring -- <the
        continuation's fm_train toy arguments>

- lambda of each term: the median over 8 training batches (fixed seed) of
  |grad L_FM| / |grad L_term| for the base checkpoint's raw weights, rounded
  to one significant digit (the hybrid's unpaired term keeps lambda_U).
- solved-endpoint terms (energy, inv_global, inv_far, profile, div): the
  term's value with a midpoint solve of NFE K against 2K on the same 64
  training images (and noise), K = 8, 16, 32: the smallest K whose value
  moves by less than 5%.
Written to docs/flow/jamie_otfm/weights/NAME.json.

    jamie_weights.py --transfer RUN --label NAME

After a continuation: its own solved-endpoint terms with its final EMA
weights at the training rollout's NFE and at the run's frozen test solve,
on the same training batches and noise (4 batches of --roll-n images). A
constraint learned through a coarse rollout need not hold for the accurate
solve that the scored outputs come from; weights/NAME_transfer.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import fm_common as fc
import jamie_methods as jm

ROLLOUT_TERMS = ('energy', 'inv_global', 'inv_far', 'profile', 'div')

def parse_cmdargs():
    argv = sys.argv[1:]
    rest = argv[argv.index('--') + 1:] if '--' in argv else []
    argv = argv[:argv.index('--')] if '--' in argv else argv
    parser = argparse.ArgumentParser(description = 'Toy continuation weights')
    parser.add_argument('base')
    parser.add_argument('--label', required = True)
    parser.add_argument('--terms', required = True)
    parser.add_argument('--batches', type = int, default = 8)
    parser.add_argument('--out', default = 'docs/flow/jamie_otfm/weights')
    a = parser.parse_args(argv)
    # the continuation's own toy arguments, parsed as fm_train parses them
    sys.argv = [ 'fm_train.py', '--label', a.label ] + rest
    import fm_train                                  # pylint: disable=import-outside-toplevel
    targs = fm_train.parse_cmdargs()
    return (a, targs)

def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph = True, allow_unused = True)
    return float(torch.sqrt(sum((x**2).sum() for x in g if x is not None)))

def transfer(run, label, out, batches = 4):
    """The terms at the training rollout's NFE and at the test solve
    (module docstring)."""
    # pylint: disable=too-many-locals
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(run, 'config.json'), encoding = 'utf-8') as f:
        c = json.load(f)
    terms = [ t for t in (c.get('toy_extra') or '').split(',') if t in ROLLOUT_TERMS ]
    targs = argparse.Namespace(**c)
    targs.toy_extra = None
    targs.toy_lambda = None
    method = jm.ToyMethod(targs, jm.load_norm(), device)
    method.need_ref = any(t in ('profile', 'div') for t in terms)
    net = fc.construct_net(c['method'], c['channels'], c['res_blocks'], c['attn'],
                           c.get('backbone', 'unet')).to(device)
    state = torch.load(fc.list_checkpoints(run)[-1], map_location = device, weights_only = False)
    net.load_state_dict(state['ema'])
    net.eval()
    with open(os.path.join('docs/flow/jamie_otfm/solver', f'{label}_frozen.json'),
              encoding = 'utf-8') as f:
        n_test = int(json.load(f)['nfe'])
    nfes = { 'train' : int(c['roll_nfe']), 'test' : n_test }
    vals = { t : { k : [] for k in nfes } for t in terms }
    with torch.no_grad():
        for b in range(batches):
            for (key, nfe) in nfes.items():
                method.set_streams(777 + b, 0)
                batch = method.draw(None, 256)
                (x0, x1, cond) = method.endpoints(batch)
                if method.coupled:
                    if method.name == 'toycond':
                        (cond, x1) = method.pair(cond, x1)
                    else:
                        (x0, x1) = method.couple(x0, x1)
                method.roll_nfe = nfe
                for t in terms:
                    vals[t][key].append(float(getattr(method, f'term_{t}')(net, x0, x1, cond)))
    res = { 'run' : run, 'label' : label, 'nfe' : nfes,
            'terms' : { t : { k : float(np.mean(v)) for (k, v) in d.items() } for (t, d) in vals.items() },
            'per_batch' : vals }
    os.makedirs(out, exist_ok = True)
    with open(os.path.join(out, f'{label}_transfer.json'), 'w', encoding = 'utf-8') as f:
        json.dump(res, f, indent = 4)
    print(json.dumps({ k : v for (k, v) in res.items() if k != 'per_batch' }, indent = 4))

def main():
    # pylint: disable=too-many-locals
    if '--transfer' in sys.argv:
        q = argparse.ArgumentParser(description = 'Rollout transfer of a continuation')
        q.add_argument('--transfer', required = True)
        q.add_argument('--label', required = True)
        q.add_argument('--out', default = 'docs/flow/jamie_otfm/weights')
        r = q.parse_args()
        transfer(r.transfer, r.label, r.out)
        return
    (a, targs) = parse_cmdargs()
    device = torch.device('cuda')
    norm = jm.load_norm()
    terms = a.terms.split(',')
    targs.toy_extra = None
    targs.toy_lambda = None
    method = jm.ToyMethod(targs, norm, device)
    method.need_ref = any(t in ('profile', 'div') for t in terms)
    net = fc.construct_net(targs.method, backbone = 'uvcgan').to(device)
    state = torch.load(os.path.join(a.base, 'resume.pt'), map_location = device,
                       weights_only = False)
    net.load_state_dict(state['raw'])
    net.train()
    params = [ p for p in net.parameters() if p.requires_grad ]
    method.set_streams(12345, 0)
    ratios = { t : [] for t in terms }
    for _ in range(a.batches):
        batch = method.draw(None, 256)
        (x0, x1, cond) = method.endpoints(batch)
        if method.coupled:
            if method.name == 'toycond':
                (cond, x1) = method.pair(cond, x1)
            else:
                (x0, x1) = method.couple(x0, x1)
        lfm = method.loss(net, x0, x1, cond)
        gfm = grad_norm(lfm, params)
        for t in terms:
            lt = getattr(method, f'term_{t}')(net, x0, x1, cond)
            ratios[t].append(gfm / max(grad_norm(lt, params), 1e-12))
    lam = { t : float(f'{np.median(r):.1g}') for (t, r) in ratios.items() }
    out = { 'base' : a.base, 'label' : a.label, 'terms' : terms, 'lambda' : lam,
            'ratios' : { t : [ float(x) for x in r ] for (t, r) in ratios.items() } }
    # rollout accuracy
    roll = [ t for t in terms if t in ROLLOUT_TERMS ]
    if roll:
        method.roll_n = 64
        checks = {}
        chosen = None
        with torch.no_grad():
            for k in (8, 16, 32):
                vals = {}
                for nfe in (k, 2 * k):
                    method.set_streams(54321, 0)
                    batch = method.draw(None, 256)
                    (x0, x1, cond) = method.endpoints(batch)
                    if method.coupled:
                        if method.name == 'toycond':
                            (cond, x1) = method.pair(cond, x1)
                        else:
                            (x0, x1) = method.couple(x0, x1)
                    method.roll_nfe = nfe
                    vals[nfe] = { t : float(getattr(method, f'term_{t}')(net, x0, x1, cond))
                                  for t in roll }
                rel = { t : abs(vals[k][t] - vals[2 * k][t]) / max(abs(vals[2 * k][t]), 1e-12)
                        for t in roll }
                checks[k] = { 'values_K' : vals[k], 'values_2K' : vals[2 * k], 'rel_change' : rel }
                print(k, checks[k], flush = True)
                if chosen is None and all(v < 0.05 for v in rel.values()):
                    chosen = k
        out['rollout_checks'] = checks
        out['roll_nfe'] = chosen or 32
        out['rollout_resolved'] = chosen is not None
    os.makedirs(a.out, exist_ok = True)
    with open(os.path.join(a.out, f'{a.label}.json'), 'w', encoding = 'utf-8') as f:
        json.dump(out, f, indent = 4)
    print(json.dumps({ k : v for (k, v) in out.items() if k != 'ratios' }, indent = 4))

if __name__ == '__main__':
    main()
