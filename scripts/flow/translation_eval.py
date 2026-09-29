#!/usr/bin/env python
"""GPU side of the PYTHIA -> JEWEL translation pilot's evaluation
(FLOW_NOTES.md, "PYTHIA -> JEWEL translation pilot"): the models' outputs,
and the validation scores that fix the solver and draw the curves.
translation_report.py computes every test metric from the saved outputs.

Models. An ODE run (fm_train.py --method jetflow, OT-CFM) maps a PYTHIA jet
by integrating its velocity from the jet (fm_common.JetMapper; the primary
setting is --ode-setting, midpoint with 32 network evaluations; 4 Euler
steps are the secondary speed setting). A bridge run (dsbm.py) maps it with
its SDE sampler (dsbm.sample_sde: Euler-Maruyama, --steps steps, noise
sqrt(eps h) at every step, the last step returning the endpoint prediction),
forward direction, EMA parameters: one sample per jet from a fixed seed over
a fixed order and batch size. The physical readout is the models' own: E =
exp(sd z + mu) - 0.1, then clipped at 0, nothing else (no threshold).

Scores (validation: PYTHIA val inputs, JEWEL val as the target sample):
fixed-crop observables of the 53 cone towers around the canvas centre
(closure_eval.Jets: E, mass, girth, p_T^D, z_lead, z_g, R_g); W1 / sigma of
each against JEWEL after the common final selection (cone E >= 10 GeV, the
input selection, applied to both), the share of outputs passing it, and the
input-output correlation of E, girth, mass, core fraction and leading-tower
energy.

Modes:
    --curves RUN ...     every checkpoint on --n-val validation inputs, at
                         the primary setting (ODE runs also at 4 Euler
                         steps): RUN/evals/translation_val.csv
    --solver-check RUN ...  the final checkpoint on --n-check validation
                         inputs: ODE midpoint 32 against 64 (and 4 Euler);
                         bridge N against 2N steps with coupled Brownian
                         increments, against two independent samples at N;
                         with bootstrap errors of the validation W1:
                         OUT/solver_check.csv
    --generate RUN ...   the final checkpoint on every PYTHIA test input at
                         the frozen setting (ODE: also 4 Euler steps); bridge
                         runs also --multi samples of the first --n-multi test
                         inputs; latency on one A6000 (batch --batch):
                         OUTDIR/sphenix/flow/translation/outputs/NAME.npy and
                         NAME.json

    translation_eval.py --curves RUN [--ode-setting 32:midpoint --steps 30]
    translation_eval.py --solver-check RUN ... --labels A,B --out DIR
    translation_eval.py --generate RUN ... --labels A,B
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

import dsbm
import fm_common as fc
from closure_eval import Jets, checkpoints, ckpt_step

MIN_JET = 10.0
# population comparison (fixed crop), and the input-dependence observables
OBS = [ 'E', 'mass', 'girth', 'ptd', 'zlead', 'zg', 'rg' ]
DEP = [ 'E', 'girth', 'mass', 'core', 'lead' ]
MONITOR_OBS = [ 'E', 'mass', 'girth', 'ptd', 'zlead' ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'PYTHIA -> JEWEL outputs')
    parser.add_argument('runs', nargs = '+')
    parser.add_argument('--curves', action = 'store_true')
    parser.add_argument('--solver-check', action = 'store_true')
    parser.add_argument('--generate', action = 'store_true')
    parser.add_argument('--labels', default = None)
    parser.add_argument('--ode-setting', default = '32:midpoint')
    parser.add_argument('--steps', type = int, default = 30)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--n-val', type = int, default = 10000)
    parser.add_argument('--n-check', type = int, default = 2000)
    parser.add_argument('--n-multi', type = int, default = 1000)
    parser.add_argument('--multi', type = int, default = 8)
    parser.add_argument('--boot', type = int, default = 200)
    parser.add_argument('--batch', type = int, default = 2000)
    parser.add_argument('--out', default = 'docs/flow/translation')
    return parser.parse_args()

# --- data

def load_set(split, domain):
    """A held-out set: canvas (N, 16, 16) GeV, key, parent_file,
    parent_event, row, col, e_cone, e_event, e_ring (translation_data.py)."""
    with np.load(os.path.join(fc.translation_root(), 'cache',
                              f'{split}_{domain}.npz')) as f:
        return { k : f[k] for k in f.files }

def train_meta(domain):
    with np.load(os.path.join(fc.translation_root(), 'cache',
                              f'meta_train_{domain}.npz')) as f:
        return { k : f[k] for k in f.files }

def output_dir():
    path = os.path.join(fc.translation_root(), 'outputs')
    os.makedirs(path, exist_ok = True)
    return path

def runs_dir():
    return os.path.join(fc.translation_root(), 'runs')

def resolve(run):
    return run if os.path.isabs(run) or os.path.exists(run) else \
        os.path.join(runs_dir(), run)

# --- observables and scores

def jet_observables(canvas, rows, device):
    """Fixed-crop observables of canvases (GeV, clipped at 0 here): the
    closure's (closure_eval.Jets: E, et, mass, girth, ptd, core, zlead, n1,
    zg, rg), the leading-tower energy, and the soft-tower summaries."""
    jets = Jets(rows, device)
    o    = jets.observables(canvas)
    w    = jets.window(canvas)
    o['lead']   = o['zlead'] * o['E']
    o['n_pos']  = (w > 0).sum((1, 2)).cpu().numpy()
    o['n_0p01'] = (w > 0.01).sum((1, 2)).cpu().numpy()
    o['n_0p1']  = (w > 0.1).sum((1, 2)).cpu().numpy()
    o['soft_0p5'] = (w * (w < 0.5)).sum((1, 2)).cpu().numpy()
    return o

def finite(x):
    x = np.asarray(x, dtype = np.float64)
    return x[np.isfinite(x)]

def w1_sigma(a, b, sd, boot = 0, rng = None):
    """W1(a, b) / sd, and the sd of its bootstrap (both samples resampled)."""
    (a, b) = (finite(a), finite(b))
    value = wasserstein_distance(a, b) / sd
    if not boot:
        return (value, np.nan)
    rng = rng or np.random.default_rng(0)
    bs  = [ wasserstein_distance(rng.choice(a, len(a)), rng.choice(b, len(b))) / sd
            for _ in range(boot) ]
    return (value, float(np.std(bs)))

def population(o_out, o_ref, boot = 0, obs = OBS):
    """W1 / sigma_ref of each observable after the final selection (E >= 10
    GeV on both), and the share of outputs that pass it."""
    sel_o = o_out['E'] >= MIN_JET
    sel_r = o_ref['E'] >= MIN_JET
    row = { 'acceptance' : float(sel_o.mean()), 'ref_acceptance' : float(sel_r.mean()) }
    rng = np.random.default_rng(1)
    for q in obs:
        ref = finite(o_ref[q][sel_r])
        (w, e) = w1_sigma(o_out[q][sel_o], ref, ref.std(), boot, rng)
        row[f'w1_{q}'] = w
        if boot:
            row[f'w1_{q}_boot_sd'] = e
    return row

def dependence(o_in, o_out):
    """Pearson correlation of input and output (no selection)."""
    row = {}
    for q in DEP:
        (a, b) = (np.asarray(o_in[q], float), np.asarray(o_out[q], float))
        ok = np.isfinite(a) & np.isfinite(b)
        row[f'corr_{q}'] = float(np.corrcoef(a[ok], b[ok])[0, 1]) \
            if b[ok].std() > 0 else np.nan
    return row

# --- models

def run_config(run):
    with open(os.path.join(run, 'config.json'), encoding = 'utf-8') as f:
        return json.load(f)

def is_bridge(run):
    return 'stage' in run_config(run)

class Translator:
    """PYTHIA canvases (GeV) -> raw outputs (GeV, >= -0.1) of one checkpoint
    of an ODE run or a bridge run."""

    def __init__(self, run, ckpt, device, cmdargs):
        self.bridge = is_bridge(run)
        self.device = device
        self.batch  = cmdargs.batch
        if self.bridge:
            (self.model, self.state, self.config) = dsbm.load_bridge(run, ckpt, device)
            self.norm = fc.Norm(self.state['norm'])
            self.eps  = self.config['eps']
            self.steps = cmdargs.steps
        else:
            (self.method, self.net, self.state, self.config) = \
                fc.load_run(run, ckpt, device, 'ema')
            self.norm = self.method.norm
        stats = self.state['stats']
        self.info = { 'step' : int(stats['step']),
                      'updates' : int(stats.get('updates', stats['step'])),
                      'train_time' : float(stats.get('total_train_time',
                                                     stats['train_time'])) }

    def settings(self, cmdargs):
        """(name, nfe, solver) of the primary setting and the secondary ones."""
        if self.bridge:
            return [ (f'sde{self.steps}', self.steps, 'sde') ]
        (nfe, solver) = cmdargs.ode_setting.split(':')
        return [ (f'{solver}{nfe}', int(nfe), solver), ('euler4', 4, 'euler') ]

    @torch.no_grad()
    def __call__(self, src, nfe = None, solver = None, seed = 0, increments = None):
        out = []
        gen = torch.Generator(device = self.device).manual_seed(seed)
        if not self.bridge:
            mapper = fc.JetMapper(self.method, self.net, nfe, solver, clip = False)
        for start in range(0, len(src), self.batch):
            j = torch.as_tensor(src[start:start + self.batch], device = self.device).float()
            if self.bridge:
                x   = self.norm.z(j, 'jet').unsqueeze(1)
                inc = None if increments is None else \
                    increments[:, start:start + self.batch]
                y   = dsbm.sample_sde(self.model, x, 1, self.eps, nfe or self.steps,
                                      gen, inc)
                out.append(self.norm.energy(y[:, 0], 'jet').cpu())
            else:
                out.append(mapper(j).cpu())
        return torch.cat(out).numpy()

def clip(x):
    return np.clip(x, 0, None)

# --- the validation monitor of dsbm.py (translation runs)

class MonitorSets:
    """The first n PYTHIA val inputs and every JEWEL val jet, for the
    population monitor of a training run (dsbm.py --monitor translation)."""

    def __init__(self, n, device):
        self.src = load_set('val', 'pythia')
        self.src = { k : v[:n] for (k, v) in self.src.items() }
        self.tgt = load_set('val', 'jewel')
        self.device = device
        self.o_in  = jet_observables(self.src['canvas'], self.src['row'], device)
        self.o_ref = jet_observables(self.tgt['canvas'], self.tgt['row'], device)

    @torch.no_grad()
    def monitor(self, model, data, eps, steps, seed):
        model.eval()
        gen = torch.Generator(device = self.device).manual_seed(seed)
        x   = data.z(torch.as_tensor(self.src['canvas'], device = self.device))
        y   = dsbm.sample_sde(model, x, 1, eps, steps, gen)
        out = data.method.norm.energy(y[:, 0], 'jet').clamp(min = 0).cpu().numpy()
        o   = jet_observables(out, self.src['row'], self.device)
        row = population(o, self.o_ref, obs = MONITOR_OBS)
        row.update(dependence(self.o_in, o))
        return { k : row[k] for k in [ 'acceptance' ]
                 + [ f'w1_{q}' for q in MONITOR_OBS ] + [ 'corr_E', 'corr_girth' ] }

# --- modes

def curves(cmdargs, device):
    src = load_set('val', 'pythia')
    src = { k : v[:cmdargs.n_val] for (k, v) in src.items() }
    tgt = load_set('val', 'jewel')
    o_in  = jet_observables(src['canvas'], src['row'], device)
    o_ref = jet_observables(tgt['canvas'], tgt['row'], device)
    for run in cmdargs.runs:
        rows = []
        path = os.path.join(run, 'evals', 'translation_val.csv')
        done = set()
        if os.path.exists(path):
            old  = pd.read_csv(path)
            rows = old.to_dict('records')
            done = set(zip(old.step, old.setting))
        for ckpt in checkpoints(run):
            tr = Translator(run, ckpt, device, cmdargs)
            for (name, nfe, solver) in tr.settings(cmdargs):
                if (ckpt_step(ckpt), name) in done:
                    continue
                out = clip(tr(src['canvas'], nfe, solver, cmdargs.seed))
                o   = jet_observables(out, src['row'], device)
                row = { 'run' : os.path.basename(run), 'step' : ckpt_step(ckpt),
                        **tr.info, 'setting' : name, 'n' : len(out),
                        **population(o, o_ref), **dependence(o_in, o),
                        'E_out_over_in' : float(np.mean(o['E'] / o_in['E'])) }
                rows.append(row)
                print(f"{row['run']} step {row['step']:7d} "
                      f"{row['train_time'] / 60:6.1f} min {name:>11s}  acc "
                      f"{row['acceptance']:.3f}  " + ' '.join(
                          f"{q} {row[f'w1_{q}']:.3f}" for q in OBS)
                      + f"  corr E {row['corr_E']:.3f} girth {row['corr_girth']:.3f}",
                      flush = True)
        os.makedirs(os.path.dirname(path), exist_ok = True)
        pd.DataFrame(rows).sort_values([ 'setting', 'step' ]).to_csv(path, index = False)
        print(f'wrote {path}', flush = True)

def labels_of(cmdargs):
    if cmdargs.labels:
        return cmdargs.labels.split(',')
    return [ os.path.basename(r.rstrip('/')) for r in cmdargs.runs ]

def final(run, device, cmdargs):
    return Translator(run, checkpoints(run)[-1], device, cmdargs)

def solver_check(cmdargs, device):
    """Is the primary solver resolved? On validation only."""
    # pylint: disable=too-many-locals
    src = load_set('val', 'pythia')
    src = { k : v[:cmdargs.n_check] for (k, v) in src.items() }
    tgt = load_set('val', 'jewel')
    o_in  = jet_observables(src['canvas'], src['row'], device)
    o_ref = jet_observables(tgt['canvas'], tgt['row'], device)
    rows = []

    def describe(label, tr, name, out, o):
        return { 'model' : label, **tr.info, 'setting' : name, 'n' : len(out),
                 **population(o, o_ref, cmdargs.boot), **dependence(o_in, o) }

    for (run, label) in zip(cmdargs.runs, labels_of(cmdargs)):
        tr = final(run, device, cmdargs)
        outs = {}
        if tr.bridge:
            n = tr.steps
            g = torch.Generator(device = device).manual_seed(4242)
            fine = np.sqrt(0.5 / n) * torch.randn(2 * n, len(src['canvas']), 1, 16, 16,
                                                  device = device, generator = g)
            outs[f'sde{n} (coupled)']  = clip(tr(src['canvas'], n, increments = dsbm.coarsen(fine)))
            outs[f'sde{2 * n} (coupled)'] = clip(tr(src['canvas'], 2 * n, increments = fine))
            outs[f'sde{n} (seed a)'] = clip(tr(src['canvas'], n, seed = cmdargs.seed + 1))
            outs[f'sde{n} (seed b)'] = clip(tr(src['canvas'], n, seed = cmdargs.seed + 2))
            pairs = [ ('N vs 2N, same noise', f'sde{n} (coupled)', f'sde{2 * n} (coupled)'),
                      ('two samples at N', f'sde{n} (seed a)', f'sde{n} (seed b)') ]
        else:
            (nfe, solver) = cmdargs.ode_setting.split(':')
            nfe = int(nfe)
            for (name, k, s) in [ ('euler4', 4, 'euler'), (f'{solver}{nfe}', nfe, solver),
                                  (f'{solver}{2 * nfe}', 2 * nfe, solver) ]:
                outs[name] = clip(tr(src['canvas'], k, s))
            pairs = [ (f'{nfe} vs {2 * nfe} NFE', f'{solver}{nfe}', f'{solver}{2 * nfe}'),
                      (f'4 Euler vs {2 * nfe} NFE', 'euler4', f'{solver}{2 * nfe}') ]
        obs = { name : jet_observables(out, src['row'], device) for (name, out) in outs.items() }
        scored = { name : describe(label, tr, name, out, obs[name])
                   for (name, out) in outs.items() }
        rows += scored.values()
        # the size of the translation itself, for scale: rms of E_out - E_in
        e_change = np.sqrt(np.mean((obs[pairs[0][1]]['E'] - o_in['E'])**2))
        for (what, a, b) in pairs:
            row = { 'model' : label, 'setting' : what, 'n' : len(src['canvas']),
                    'tower_rms_gev' : float(np.sqrt(np.mean((outs[a] - outs[b])**2))),
                    'E_rms_gev' : float(np.sqrt(np.mean((obs[a]['E'] - obs[b]['E'])**2))),
                    'E_change_rms_gev' : float(e_change) }
            for q in OBS:
                row[f'dw1_{q}'] = scored[a][f'w1_{q}'] - scored[b][f'w1_{q}']
                row[f'w1_{q}_boot_sd'] = scored[b][f'w1_{q}_boot_sd']
            rows.append(row)
        print(f'{label}: solver check done', flush = True)
    df = pd.DataFrame(rows)
    os.makedirs(cmdargs.out, exist_ok = True)
    df.to_csv(os.path.join(cmdargs.out, 'solver_check.csv'), index = False)
    with pd.option_context('display.width', 250, 'display.max_columns', 60):
        print(df.round(4).to_string(index = False))

@torch.no_grad()
def latency(tr, src, nfe, solver, device, repeats = 5):
    """ms per jet at batch tr.batch on one GPU, after one warm-up batch."""
    x = src[:tr.batch]
    tr(x, nfe, solver)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(repeats):
        tr(x, nfe, solver, seed = k)
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / (repeats * len(x))

def generate(cmdargs, device):
    src = load_set('test', 'pythia')
    for (run, label) in zip(cmdargs.runs, labels_of(cmdargs)):
        tr = final(run, device, cmdargs)
        for (k, (name, nfe, solver)) in enumerate(tr.settings(cmdargs)):
            torch.cuda.synchronize()
            t0  = time.perf_counter()
            raw = tr(src['canvas'], nfe, solver, cmdargs.seed)
            torch.cuda.synchronize()
            dt  = time.perf_counter() - t0
            stem = f'{label}__{name}'
            np.save(os.path.join(output_dir(), f'{stem}.npy'), raw.astype(np.float32))
            meta = { 'run' : os.path.basename(run), 'label' : label, 'setting' : name,
                     'primary' : k == 0, 'nfe' : nfe, 'solver' : solver,
                     'sampler_seed' : cmdargs.seed if tr.bridge else None,
                     **tr.info, 'n' : int(len(raw)), 'raw_min_gev' : float(raw.min()),
                     'generate_seconds' : dt,
                     'ms_per_jet' : latency(tr, src['canvas'], nfe, solver, device),
                     'batch' : tr.batch, 'gpu' : torch.cuda.get_device_name() }
            if tr.bridge:
                meta['eps'] = tr.eps
            with open(os.path.join(output_dir(), f'{stem}.json'), 'w',
                      encoding = 'utf-8') as f:
                json.dump(meta, f, indent = 4)
            print(f'{stem}: {meta}', flush = True)
        if tr.bridge and cmdargs.multi > 1:
            sub = src['canvas'][:cmdargs.n_multi]
            outs = np.stack([ tr(sub, tr.steps, 'sde', 1000 + k)
                              for k in range(cmdargs.multi) ]).astype(np.float32)
            stem = f'{label}__sde{tr.steps}_multi{cmdargs.multi}'
            np.save(os.path.join(output_dir(), f'{stem}.npy'), outs)
            with open(os.path.join(output_dir(), f'{stem}.json'), 'w',
                      encoding = 'utf-8') as f:
                json.dump({ 'run' : os.path.basename(run), 'label' : label,
                            'samples' : cmdargs.multi, 'inputs' : 'first'
                            f' {cmdargs.n_multi} PYTHIA test jets',
                            'seeds' : [ 1000 + k for k in range(cmdargs.multi) ],
                            **tr.info }, f, indent = 4)
            print(f'{stem}: {outs.shape}', flush = True)

def main():
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    cmdargs.runs = [ resolve(r) for r in cmdargs.runs ]
    if cmdargs.curves:
        curves(cmdargs, device)
    elif cmdargs.solver_check:
        solver_check(cmdargs, device)
    elif cmdargs.generate:
        generate(cmdargs, device)
    else:
        raise SystemExit('one of --curves, --solver-check, --generate')

if __name__ == '__main__':
    main()
