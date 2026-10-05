#!/usr/bin/env python
"""GPU side of the toy study's evaluation (FLOW_NOTES.md, "Jamie's toy
exercises with OT flow matching"): solver checks on validation, then the
frozen-solver outputs on the held-out sets. jamie_report.py scores them.

    jamie_eval.py RUN --check [--nfe 32]       step doubling on validation
    jamie_eval.py RUN --generate --nfe N       test outputs at the frozen solve

A run is scored with its final checkpoint (the budget's), EMA weights. D
(toyflow) solves dx/dt = v(t, x) from the source image; C (toycond) from
Gaussian noise of a fixed seed (one (n, 1, 24, 64) draw per set, so an image
keeps its noise whatever the batching) with the condition held fixed. The
midpoint rule; NFE is the number of network evaluations (two per step).
Outputs are raw GeV (exp(sd x + mu) - 0.1, >= -0.1), not clipped.

Sets by the run's state transform (--toy-kind) and target pool:
    sub    t1_mix, t1_low, t1_high, t1_beyond, t1_ue (M -> B_hat), and
           t2_pair's M_vac and M_med (Exercise 2 and the 2 x 2), val v1_mix
    clean  t3_pair (or t3g_pair for the girth rule), its bank (128 draws of
           100 parents, 512 of 4 illustration parents; D: one output), the
           condition swap (C), val v3_pair / v3g_pair
    ue     t4_pair's J_vac + B_a, t4_bank, the swap (C), val v4_pair

Solver check: the validation set's first --n-check inputs at NFE N and 2N
from the same input and, for C, the same noise; for C also a second noise at
2N. Adequate when the per-image key energy (sub: the true-axis cone energy
of M - B_hat; clean: the cone energy; ue: the cone energy minus the image's
own rho R > 1.2 times the cone area) moves by less than 1% of the rms change
the model makes (output - input), every W1/sigma of the key observables
moves by less than its bootstrap sd, and, for C, the per-image change is
below 1/10 of that between two noises. Otherwise N is doubled
(condfm_post.sbatch's rule).
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

import fm_common as fc
import jamie_methods as jm
import jamie_obs as jo

OBS_CHECK = [ 'cone', 'ring', 'girth', 'mass', 'ptd', 'zlead' ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Toy study: outputs')
    parser.add_argument('run')
    parser.add_argument('--check', action = 'store_true')
    parser.add_argument('--curves', action = 'store_true',
        help = 'every checkpoint on --n-check validation inputs at --nfe (descriptive)')
    parser.add_argument('--generate', action = 'store_true')
    parser.add_argument('--nfe', type = int, default = 32)
    parser.add_argument('--n-check', type = int, default = 1000)
    parser.add_argument('--batch', type = int, default = 1000)
    parser.add_argument('--bank-draws', type = int, default = 128)
    parser.add_argument('--illu-draws', type = int, default = 512)
    parser.add_argument('--label', default = None)
    parser.add_argument('--out', default = 'docs/flow/jamie_otfm/solver')
    parser.add_argument('--sets', default = None, help = 'comma separated subset')
    return parser.parse_args()

def cache(name):
    with np.load(os.path.join(jm.jamie_root(), 'cache', f'{name}.npz')) as f:
        return { k : f[k] for k in f.files }

def outputs_dir(label):
    d = os.path.join(jm.jamie_root(), 'outputs', label)
    os.makedirs(d, exist_ok = True)
    return d

class Model:
    """A toy run's final checkpoint (or `ckpt`), EMA weights."""

    def __init__(self, run, device, ckpt = None):
        with open(os.path.join(run, 'config.json'), encoding = 'utf-8') as f:
            self.config = json.load(f)
        ckpt = ckpt or fc.list_checkpoints(run)[-1]
        state = torch.load(ckpt, map_location = device, weights_only = False)
        self.norm = fc.Norm(state['norm'])
        self.kind = self.config['toy_kind']
        self.cond = self.config['method'] == 'toycond'
        self.net = fc.construct_net(self.config['method'], self.config['channels'],
                                    self.config['res_blocks'], self.config['attn'],
                                    self.config.get('backbone', 'unet')).to(device)
        self.net.load_state_dict(state['ema'])
        self.net.eval()
        self.device = device
        self.info = { 'run' : os.path.basename(run.rstrip('/')), 'checkpoint' : os.path.basename(ckpt),
                      'step' : int(state['stats']['step']),
                      'train_time' : float(state['stats']['train_time']) }

    def noise(self, n, seed):
        g = torch.Generator(device = self.device).manual_seed(seed)
        return torch.randn((n, 1, 24, 64), device = self.device, generator = g)

    @torch.no_grad()
    def __call__(self, images, nfe, seed = 0, batch = 1000, noise = None):
        """Raw GeV outputs for (n, 24, 64) input images."""
        out = []
        if self.cond and noise is None:
            noise = self.noise(len(images), seed)
        for s in range(0, len(images), batch):
            x = torch.as_tensor(np.asarray(images[s:s + batch], np.float32), device = self.device)
            z = self.norm.z(x, self.kind).unsqueeze(1)
            if self.cond:
                y = fc.integrate(self.net, noise[s:s + batch], z, nfe, 'midpoint')
            else:
                y = fc.integrate(self.net, z, None, nfe, 'midpoint')
            out.append(self.norm.energy(y[:, 0], self.kind).cpu())
        return torch.cat(out).numpy()

def family(model):
    """(sets, bank, val) of a run, from its transform and target pool."""
    girth = (model.config.get('toy_target') or '') == 'e3g_med'
    if model.kind == 'sub':
        return ([ 't1_mix', 't1_low', 't1_high', 't1_beyond', 't1_ue', 't2_pair' ], None, 'v1_mix')
    if model.kind == 'clean':
        return (([ 't3g_pair' ], 't3g_bank', 'v3g_pair') if girth
                else ([ 't3_pair' ], 't3_bank', 'v3_pair'))
    return ([ 't4_pair' ], 't4_bank', 'v4_pair')

def inputs(model, name, d):
    """{tag: (input images, axes)} of a set for this run."""
    ax = np.stack([ d['eta0'], d['phi0'] ], 1) if 'eta0' in d else None
    if model.kind == 'sub':
        if name == 't1_ue':
            return { 'ue' : (d['ue'], None) }
        if name in ('t2_pair', 'v2_pair'):
            return { 'vac' : (d['jet'] + d['ue'], ax), 'med' : (d['med'] + d['ue'], ax) }
        return { 'mix' : (d['jet'] + d['ue'], ax) }
    if model.kind == 'clean':
        return { 'vac' : (d['jet'], ax) }
    return { 'vac' : (d['jet'] + d['ue'], ax) }

def key_energy(model, out, inp, ax, device):
    """The per-image energy the solver check watches (module docstring)."""
    if model.kind == 'sub':
        o = jo.observables(inp - out, ax, device, substructure = False)
        i = jo.observables(inp, ax, device, substructure = False)
        return (o['cone'], i['cone'])
    if model.kind == 'clean':
        return (jo.observables(out, ax, device, substructure = False)['cone'],
                jo.observables(inp, ax, device, substructure = False)['cone'])
    return (rho_cone(out, ax, device), rho_cone(inp, ax, device))

def rho_cone(images, ax, device):
    """Cone energy minus the image's own rho (R > 1.2) times the cone area."""
    x = torch.as_tensor(np.asarray(images, np.float32), device = device)
    (_, _, dr) = jo.offsets(ax, device)
    far = dr > 1.2
    rho = (x * far).sum((1, 2)) / far.sum((1, 2))
    cone = dr < 0.4
    return ((x * cone).sum((1, 2)) - rho * cone.sum((1, 2))).cpu().numpy()

def check(cmdargs, model, device):
    # pylint: disable=too-many-locals
    (_, _, val) = family(model)
    d = cache(val)
    n = cmdargs.n_check
    tag, (inp, ax) = next(iter(inputs(model, val, { k : v[:n] for (k, v) in d.items() }).items()))
    ax = ax[:n]
    nfe = cmdargs.nfe
    rows = []
    outs = { 'N' : model(inp, nfe, seed = 0), '2N' : model(inp, 2 * nfe, seed = 0) }
    if model.cond:
        outs['2N noise b'] = model(inp, 2 * nfe, seed = 1)
    rd = model.kind == 'sub'
    obs = { k : jo.observables(inp - v if rd else v, ax, device) for (k, v) in outs.items() }
    (e_n, e_in) = key_energy(model, outs['N'], inp, ax, device)
    (e_2n, _) = key_energy(model, outs['2N'], inp, ax, device)
    change = float(np.sqrt(np.mean((e_n - e_in)**2)))
    row = { 'run' : model.info['run'], 'set' : val, 'input' : tag, 'n' : n, 'nfe' : nfe,
            'key_rms_N_vs_2N' : float(np.sqrt(np.mean((e_n - e_2n)**2))),
            'key_change_rms' : change,
            'tower_rms_N_vs_2N' : float(np.sqrt(np.mean((outs['N'] - outs['2N'])**2))) }
    row['key_over_change'] = row['key_rms_N_vs_2N'] / max(change, 1e-9)
    rng = np.random.default_rng(0)
    shifts_ok = True
    for q in OBS_CHECK:
        (a, b) = (obs['N'][q], obs['2N'][q])
        (a, b) = (a[np.isfinite(a)], b[np.isfinite(b)])
        sd = max(float(b.std()), 1e-9)
        dw = wasserstein_distance(a, b) / sd
        bs = np.std([ wasserstein_distance(rng.choice(b, len(b)), rng.choice(b, len(b))) / sd
                      for _ in range(100) ])
        row[f'w1_{q}_N_vs_2N'] = dw
        row[f'w1_{q}_boot_sd'] = float(bs)
        shifts_ok &= dw < bs
    row['w1_ok'] = bool(shifts_ok)
    if model.cond:
        (e_b, _) = key_energy(model, outs['2N noise b'], inp, ax, device)
        row['key_rms_two_noises'] = float(np.sqrt(np.mean((e_2n - e_b)**2)))
        row['tower_rms_two_noises'] = float(np.sqrt(np.mean((outs['2N'] - outs['2N noise b'])**2)))
        row['key_over_two_noises'] = row['key_rms_N_vs_2N'] / max(row['key_rms_two_noises'], 1e-9)
    row['adequate'] = bool(row['key_over_change'] < 0.01 and shifts_ok
                           and (not model.cond or row['key_over_two_noises'] < 0.1))
    rows.append(row)
    os.makedirs(cmdargs.out, exist_ok = True)
    path = os.path.join(cmdargs.out, f"{cmdargs.label or model.info['run']}.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    pd.concat([ old, pd.DataFrame(rows) ]).to_csv(path, index = False)
    print(json.dumps(row, indent = 2), flush = True)
    return row['adequate']

def generate(cmdargs, model, device):
    # pylint: disable=too-many-locals
    label = cmdargs.label or model.info['run']
    od = outputs_dir(label)
    (sets, bank, _) = family(model)
    if cmdargs.sets:
        sets = [ s for s in sets if s in cmdargs.sets.split(',') ]
    meta = { **model.info, 'nfe' : cmdargs.nfe, 'solver' : 'midpoint', 'cond' : model.cond,
             'kind' : model.kind, 'gpu' : torch.cuda.get_device_name(), 'sets' : {} }
    for name in sets:
        d = cache(name)
        for (tag, (inp, _)) in inputs(model, name, d).items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(inp, cmdargs.nfe, seed = 0, batch = cmdargs.batch)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            np.save(os.path.join(od, f'{name}_{tag}.npy'), out.astype(np.float32))
            meta['sets'][f'{name}_{tag}'] = { 'n' : len(out), 'seconds' : dt,
                                              'ms_per_image' : 1000 * dt / len(out), 'seed' : 0 }
            print(f'{label} {name}_{tag}: {out.shape}, {1000 * dt / len(out):.2f} ms/image',
                  flush = True)
    if bank is not None and (not cmdargs.sets or bank in cmdargs.sets.split(',')):
        b = cache(bank)
        src = cache(family(model)[0][0])
        tag = 'vac'
        def bank_inputs(parents):
            if model.kind == 'clean':
                return src['jet'][parents]
            return src['jet'][parents] + src['ue'][parents]
        for (key, parents, draws) in [ ('bank', b['parents'], cmdargs.bank_draws),
                                       ('illu', b['illu_parents'], cmdargs.illu_draws) ]:
            x = bank_inputs(parents)
            if model.cond:
                rep = np.repeat(x, draws, axis = 0)
                out = model(rep, cmdargs.nfe, seed = 2000 + (key == 'illu'), batch = cmdargs.batch)
                out = out.reshape(len(parents), draws, 24, 64)
            else:
                out = model(x, cmdargs.nfe, batch = cmdargs.batch)[:, None]
            np.save(os.path.join(od, f'{bank}_{key}.npy'), out.astype(np.float16))
            meta['sets'][f'{bank}_{key}'] = { 'parents' : len(parents), 'draws' : int(out.shape[1]),
                                              'seed' : 2000 + (key == 'illu') }
            print(f'{label} {bank}_{key}: {out.shape}', flush = True)
        if model.cond:
            # the condition swap: input i's noise with input perm[i]'s condition
            d = cache(family(model)[0][0])
            x = inputs(model, '', { k : v[:1000] for (k, v) in d.items() })[tag][0]
            noise = model.noise(1000, 3000)
            shift = 1 + int(np.random.default_rng(2).integers(999))
            perm = (np.arange(1000) + shift) % 1000
            own = model(x, cmdargs.nfe, noise = noise)
            swap = model(x[perm], cmdargs.nfe, noise = noise)
            again = model(x, cmdargs.nfe, noise = noise)
            other = model(x, cmdargs.nfe, seed = 3001)
            np.savez(os.path.join(od, 'swap.npz'), own = own.astype(np.float32),
                     swap = swap.astype(np.float32), other = other.astype(np.float32),
                     perm = perm, shift = shift)
            meta['swap'] = { 'n' : 1000, 'shift' : shift, 'noise_seed' : 3000,
                             'other_noise_seed' : 3001,
                             'repeat_max_abs_gev' : float(np.abs(own - again).max()) }
            print(f"{label} swap: repeat max |diff| {meta['swap']['repeat_max_abs_gev']:.3g}",
                  flush = True)
    with open(os.path.join(od, 'meta.json'), 'w', encoding = 'utf-8') as f:
        json.dump(meta, f, indent = 4)

def curve_row(model, inp, ax, ref, nfe, device):
    """Validation scores of one checkpoint: population W1/sigma of the key
    observables against the validation target (sub: the true J; clean and
    ue: the validation quenched images), and the key energy's correlation
    with the input."""
    out = model(inp, nfe, seed = 0)
    rd = model.kind == 'sub'
    o = jo.observables(inp - out if rd else out, ax, device, substructure = False)
    r = jo.observables(ref, ax, device, substructure = False)
    row = {}
    for q in [ 'cone', 'ring', 'far', 'girth', 'ptd', 'zlead' ]:
        (a, b) = (o[q][np.isfinite(o[q])], r[q][np.isfinite(r[q])])
        row[f'w1_{q}'] = wasserstein_distance(a, b) / max(float(b.std()), 1e-9)
    (e_out, e_in) = key_energy(model, out, inp, ax, device)
    row['corr_key_input'] = float(np.corrcoef(e_out, e_in)[0, 1])
    if rd:
        row['cone_offset'] = float((o['cone'] - r['cone']).mean())
        row['cone_rms'] = float((o['cone'] - r['cone']).std())
    return row

def curves(cmdargs, device):
    run = cmdargs.run
    rows = []
    for ckpt in fc.list_checkpoints(run):
        model = Model(run, device, ckpt)
        (_, _, val) = family(model)
        d = { k : v[:cmdargs.n_check] for (k, v) in cache(val).items() }
        (inp, ax) = next(iter(inputs(model, val, d).values()))
        if model.kind == 'sub':
            ref = d['jet']
        elif model.kind == 'clean':
            ref = d['med']
        else:
            ref = d['med'] + d['ue']
        row = { 'run' : model.info['run'], 'step' : model.info['step'],
                'train_time' : model.info['train_time'], 'nfe' : cmdargs.nfe,
                **curve_row(model, inp, ax, ref, cmdargs.nfe, device) }
        rows.append(row)
        print(row, flush = True)
    pd.DataFrame(rows).to_csv(os.path.join(run, 'curves.csv'), index = False)

def main():
    cmdargs = parse_cmdargs()
    device = torch.device('cuda')
    if cmdargs.curves:
        curves(cmdargs, device)
        return
    model = Model(cmdargs.run, device)
    if cmdargs.check:
        check(cmdargs, model, device)
    if cmdargs.generate:
        generate(cmdargs, model, device)

if __name__ == '__main__':
    main()
