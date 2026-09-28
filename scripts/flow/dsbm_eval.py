#!/usr/bin/env python
"""Closure-test scores of the alpha-DSBM runs (dsbm.py) and their references
(FLOW_NOTES.md, "alpha-DSBM closure test"), with the stochastic outputs
evaluated as such.

Scoring is closure_eval.py's, unchanged: the normalised-shape EMD
(unit-energy jets, dR / 0.4 per unit energy; dimensionless, not the GeV
decomposition metric), the full EMD in GeV, the energy response and error,
observables, marginals, the per-tower error by true tower energy
(control_scores). The physical readout is the existing one: E = exp(sd z +
mu) - 0.1 of the model coordinates, clipped at 0 (raw values are >= -0.1
GeV by construction).

Outputs of a bridge run come from the SDE (dsbm.sample_sde: Euler-Maruyama,
--steps steps, noise sqrt(eps h) at every step, the last step returning the
endpoint prediction), forward direction, EMA parameters. Every jet gets ONE
sample, from a fixed generator seed (--seed) over a fixed order and batch
size, so the same event sees the same Brownian increments in every model.
ODE runs (fm_train.py jetflow, e.g. the OT-CFM comparator) use JetMapper at
--ode-setting (32 midpoint evaluations), as their earlier reports.

Modes:
    --select      every checkpoint of each bridge run on the first --n-val
                  validation pairs (and --train-n training pairs): one SDE
                  sample per jet; the checkpoint of lowest mean full EMD (GeV),
                  the closure test's selection metric, is marked
    (default)     each run's selected checkpoint on the test pairs (--n, all
                  20k by default): the closure report, and the identity
    --steps-check the first --n-check validation pairs: --steps against
                  2 x --steps with coupled Brownian increments (integration
                  error), against two independent samples at --steps
                  (sampling noise)
    --multi K     the first --n-multi test pairs, K samples each: per-jet
                  spread of the observables, shape EMD between samples of a
                  jet, each sample's scores, and the scores of the average
                  of the K images (reported separately, never as the result)
    --figure PNG  preselected test jets (four energy quantiles, as
                  closure_eval.py): J, T(J), every run, and three samples plus
                  the K-sample average of the first bridge run; plus
                  PNG_residuals.png (error by true tower energy) and
                  PNG_distributions.png (marginals against the target)

    dsbm_eval.py RUN ... --select [--steps 30]
    dsbm_eval.py RUN ... --out PREFIX
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import dsbm
import fm_common as fc
from closure_eval import (load_pairs, Jets, scores, control_scores, apply,
                          selected_step, checkpoints, ckpt_step, TOWER_CLASSES,
                          OFFSET)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'alpha-DSBM closure scores')
    parser.add_argument('runs', nargs = '+')
    parser.add_argument('--select', action = 'store_true')
    parser.add_argument('--steps-check', action = 'store_true')
    parser.add_argument('--multi', type = int, default = 0)
    parser.add_argument('--figure', default = None)
    parser.add_argument('--steps', type = int, default = 30)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--ode-setting', default = '32:midpoint')
    parser.add_argument('--n-val', type = int, default = 5000)
    parser.add_argument('--train-n', type = int, default = 0)
    parser.add_argument('--n', type = int, default = None)
    parser.add_argument('--n-check', type = int, default = 1000)
    parser.add_argument('--n-multi', type = int, default = 1000)
    parser.add_argument('--batch', type = int, default = 2000)
    parser.add_argument('--labels', default = None,
        help = 'comma separated display names of the runs')
    parser.add_argument('--out', default = os.path.join(fc.out_root(), 'dsbm_eval'))
    return parser.parse_args()

def is_bridge(run):
    with open(os.path.join(run, 'config.json'), encoding = 'utf-8') as f:
        return 'stage' in json.load(f)

class Mapper:
    """Source jets (GeV) -> outputs (GeV, raw) of a bridge or an ODE run."""

    def __init__(self, run, ckpt, cmdargs, device):
        self.bridge = is_bridge(run)
        self.device = device
        self.batch  = cmdargs.batch
        if self.bridge:
            (self.model, self.state, self.config) = dsbm.load_bridge(run, ckpt, device)
            with open(fc.Norm.path(method = 'jetflow'), encoding = 'utf-8') as f:
                self.norm = fc.Norm(json.load(f))
            self.eps = self.config['eps']
            self.nfe = cmdargs.steps
            self.setting = f'SDE {cmdargs.steps} steps'
        else:
            (method, net, self.state, self.config) = fc.load_run(run, ckpt, device, 'ema')
            (nfe, solver) = cmdargs.ode_setting.split(':')
            self.ode = fc.JetMapper(method, net, int(nfe), solver, clip = False)
            self.nfe = int(nfe)
            self.setting = f'ODE {solver} {nfe}'

    @torch.no_grad()
    def __call__(self, src, seed = 0, steps = None, increments = None):
        if not self.bridge:
            return apply(self.ode, src, self.batch, self.device)
        steps = steps or self.nfe
        gen = torch.Generator(device = self.device).manual_seed(seed)
        out = []
        for start in range(0, len(src), self.batch):
            j = torch.as_tensor(src[start:start + self.batch], device = self.device).float()
            x = self.norm.z(j, 'jet').unsqueeze(1)
            inc = None if increments is None else increments[:, start:start + self.batch]
            y = dsbm.sample_sde(self.model, x, 1, self.eps, steps, gen, inc)
            out.append(self.norm.energy(y[:, 0], 'jet').cpu())
        return torch.cat(out).numpy()

def resolve(run):
    return run if os.path.isabs(run) or os.path.exists(run) else \
        os.path.join(fc.out_root(), run)

def selected(run, cmdargs):
    if is_bridge(run):
        return selected_step(run, f'_sde{cmdargs.steps}')
    return selected_step(run, '_' + cmdargs.ode_setting.replace(':', ''))

def mapper_of(run, cmdargs, device):
    step = selected(run, cmdargs)
    ckpt = [ c for c in checkpoints(run) if ckpt_step(c) == step ][0]
    return Mapper(run, ckpt, cmdargs, device)

def select(cmdargs, device):
    sets = { 'val' : load_pairs('val', cmdargs.n_val) }
    if cmdargs.train_n:
        sets['train'] = load_pairs('train', cmdargs.train_n)
    jets = { k : Jets(v['row'], device) for (k, v) in sets.items() }
    for run in cmdargs.runs:
        rows = { k : [] for k in sets }
        for ckpt in checkpoints(run):
            m = Mapper(run, ckpt, cmdargs, device)
            for (k, pairs) in sets.items():
                pred = np.clip(m(pairs['src'], cmdargs.seed), 0, None)
                (row, _) = scores(pred, pairs, jets[k], m.config['label'])
                row.update({ 'step' : ckpt_step(ckpt),
                             'train_time' : m.state['stats']['train_time'],
                             'total_train_time' : m.state['stats'].get(
                                 'total_train_time', m.state['stats']['train_time']) })
                rows[k].append(row)
                print(f"{m.config['label']} {k:5s} step {row['step']:7d} "
                      f"{row['total_train_time'] / 60:6.1f} min  emd {row['emd_gev']:.3f}"
                      f"  shape {row['shape_emd']:.4f}  response {row['response_mean']:.3f}",
                      flush = True)
        df = pd.DataFrame(rows['val'])
        best = int(df.sort_values('emd_gev').step.iloc[0])
        df['selected'] = df.step == best
        os.makedirs(os.path.join(run, 'evals'), exist_ok = True)
        df.to_csv(os.path.join(run, 'evals', f'closure_val_sde{cmdargs.steps}.csv'),
                  index = False)
        if 'train' in rows:
            pd.DataFrame(rows['train']).to_csv(os.path.join(
                run, 'evals', f'closure_train_sde{cmdargs.steps}.csv'), index = False)
        print(f'{run}: selected step {best}', flush = True)

def labels_of(cmdargs):
    if cmdargs.labels:
        return cmdargs.labels.split(',')
    return [ os.path.basename(r.rstrip('/')) for r in cmdargs.runs ]

def report(cmdargs, device):
    # pylint: disable=too-many-locals
    pairs = load_pairs('test', cmdargs.n)
    jets  = Jets(pairs['row'], device)
    rows, obs = [], []
    row = control_scores(pairs['src'], pairs, jets, 'identity F(J) = J')
    (_, o) = scores(pairs['src'], pairs, jets, 'identity F(J) = J')
    for x in o:
        row[f"rel_rmse_{x['observable']}"] = x['rel_rmse']
        row[f"rel_bias_{x['observable']}"] = x['rel_bias']
        row[f"delta_frac_{x['observable']}"] = x['delta_frac']
    rows.append(row)
    obs += o
    for (run, label) in zip(cmdargs.runs, labels_of(cmdargs)):
        m = mapper_of(run, cmdargs, device)
        torch.cuda.synchronize()
        t0  = time.perf_counter()
        raw = m(pairs['src'], cmdargs.seed)
        torch.cuda.synchronize()
        dt  = time.perf_counter() - t0
        row = control_scores(raw, pairs, jets, label)
        (_, o) = scores(np.clip(raw, 0, None), pairs, jets, label)
        for x in o:
            row[f"rel_rmse_{x['observable']}"] = x['rel_rmse']
            row[f"rel_bias_{x['observable']}"] = x['rel_bias']
            row[f"delta_frac_{x['observable']}"] = x['delta_frac']
        stats = m.state['stats']
        row.update({ 'run' : os.path.basename(run), 'setting' : m.setting,
                     'nfe' : m.nfe, 'step' : ckpt_step_of(m),
                     'updates' : stats.get('updates', stats['step']),
                     'train_time_min' : stats['train_time'] / 60,
                     'total_train_min' : stats.get('total_train_time',
                                                   stats['train_time']) / 60,
                     'ms_per_jet' : 1000 * dt / len(pairs['src']),
                     'raw_min_gev' : float(raw.min()) })
        rows.append(row)
        obs += o
        print(f'{label}: done ({dt:.1f} s)', flush = True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{cmdargs.out}.csv', index = False)
    pd.DataFrame(obs).to_csv(f'{cmdargs.out}_observables.csv', index = False)
    show = [ 'model', 'shape_emd', 'shape_emd_median', 'shape_emd_p90', 'shape_emd_p99',
             'emd_gev', 'E_out_over_E_in', 'E_out_over_E_in_sd', 'E_rmse_gev',
             'dmass_rmse', 'dgirth_rmse', 'tower>5_mean', 'tower>5_rms' ]
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(df[[ c for c in show if c in df ]].round(4).to_string(index = False))
        print(df[[ 'model' ] + [ c for c in df if c.startswith('w1_') or
                                 c.startswith('rel_rmse_') ]].round(3).to_string(index = False))
    print(f'wrote {cmdargs.out}.csv')

def ckpt_step_of(m):
    return int(m.state['stats']['step'])

def steps_check(cmdargs, device):
    """Integration error (N vs 2N steps, the same Brownian path) against
    sampling noise (two independent paths at N), on validation jets."""
    pairs = load_pairs('val', cmdargs.n_check)
    jets  = Jets(pairs['row'], device)
    rows  = []
    n = cmdargs.steps
    for (run, label) in zip(cmdargs.runs, labels_of(cmdargs)):
        m = mapper_of(run, cmdargs, device)
        if not m.bridge:
            continue
        g = torch.Generator(device = device).manual_seed(4242)
        fine = np.sqrt(0.5 / n) * torch.randn(2 * n, len(pairs['src']), 1, 16, 16,
                                              device = device, generator = g)
        out_n  = np.clip(m(pairs['src'], steps = n, increments = dsbm.coarsen(fine)), 0, None)
        out_2n = np.clip(m(pairs['src'], steps = 2 * n, increments = fine), 0, None)
        out_a  = np.clip(m(pairs['src'], cmdargs.seed + 1), 0, None)
        out_b  = np.clip(m(pairs['src'], cmdargs.seed + 2), 0, None)
        row = { 'model' : label, 'steps' : n }
        for (name, a, b) in [ ('N_vs_2N_coupled', out_n, out_2n),
                              ('two_samples_N', out_a, out_b) ]:
            (full, shape) = jets.emds(a, b)
            wa = jets.window(a).cpu().numpy()
            wb = jets.window(b).cpu().numpy()
            row[f'{name}_shape_emd'] = float(np.nanmean(shape))
            row[f'{name}_emd_gev'] = float(full.mean())
            row[f'{name}_tower_rms_gev'] = float(np.sqrt(np.mean((wa - wb)**2)))
            row[f'{name}_E_rms_gev'] = float(np.sqrt(np.mean(
                (wa.sum((1, 2)) - wb.sum((1, 2)))**2)))
        for (tag, out) in [ ('N', out_n), ('2N', out_2n) ]:
            (sc, _) = scores(out, pairs, jets, label)
            row[f'{tag}_shape_emd_to_truth'] = sc['shape_emd']
            row[f'{tag}_emd_to_truth_gev'] = sc['emd_gev']
            row[f'{tag}_response'] = sc['response_mean']
        rows.append(row)
        print(row, flush = True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{cmdargs.out}_steps.csv', index = False)
    print(df.round(4).T.to_string())

OBS_SPREAD = [ 'E', 'mass', 'girth', 'ptd', 'zlead' ]

def multi(cmdargs, device):
    """K samples per test jet: sampling variability, and the average image."""
    # pylint: disable=too-many-locals
    pairs = load_pairs('test', cmdargs.n_multi)
    jets  = Jets(pairs['row'], device)
    k_max = cmdargs.multi
    rows  = []
    for (run, label) in zip(cmdargs.runs, labels_of(cmdargs)):
        m = mapper_of(run, cmdargs, device)
        if not m.bridge:
            continue
        outs = [ np.clip(m(pairs['src'], 1000 + k), 0, None) for k in range(k_max) ]
        per_sample = []
        for (k, out) in enumerate(outs):
            (sc, _) = scores(out, pairs, jets, label)
            per_sample.append(sc)
        o = [ jets.observables(out) for out in outs ]
        o_t = jets.observables(pairs['tgt'])
        row = { 'model' : label, 'samples' : k_max, 'n' : len(pairs['src']) }
        for q in OBS_SPREAD:
            v = np.stack([ x[q] for x in o ])          # (K, n)
            sd_truth = np.nanstd(o_t[q])
            row[f'spread_{q}_over_sd'] = float(np.nanmean(np.nanstd(v, axis = 0)) / sd_truth)
        # shape EMD between two samples of the same jet
        (_, s01) = jets.emds(outs[0], outs[1])
        row['shape_emd_between_samples'] = float(np.nanmean(s01))
        sh = np.array([ s['shape_emd'] for s in per_sample ])
        row.update({ 'shape_emd_single_mean_over_samples' : float(sh.mean()),
                     'shape_emd_single_sd_over_samples' : float(sh.std()),
                     'emd_gev_single_mean' : float(np.mean([ s['emd_gev'] for s in per_sample ])),
                     'response_single_mean' : float(np.mean([ s['response_mean']
                                                              for s in per_sample ])) })
        avg = np.mean(outs, axis = 0)
        (sc, _) = scores(avg, pairs, jets, label)
        row.update({ 'avg_image_shape_emd' : sc['shape_emd'],
                     'avg_image_emd_gev' : sc['emd_gev'],
                     'avg_image_response' : sc['response_mean'] })
        rows.append(row)
        print(row, flush = True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{cmdargs.out}_multi.csv', index = False)
    print(df.round(4).T.to_string())

def figure(cmdargs, device):
    """Preselected jets, residuals by true tower energy, marginals."""
    # pylint: disable=import-outside-toplevel,too-many-locals,too-many-statements
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pairs = load_pairs('test', cmdargs.n or 2000)
    jets  = Jets(pairs['row'], device)
    e     = pairs['src'][:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9].sum((1, 2))
    pick  = [ int(np.argmin(np.abs(e - np.quantile(e, q)))) for q in (0.25, 0.5, 0.75, 0.95) ]
    sel   = { k : v[pick] for (k, v) in pairs.items() }
    jsel  = Jets(sel['row'], device)
    columns = [ ('J', sel['src']), ('T(J)', sel['tgt']) ]
    full_outputs = {}
    for (i, (run, label)) in enumerate(zip(cmdargs.runs, labels_of(cmdargs))):
        m = mapper_of(run, cmdargs, device)
        out = np.clip(m(pairs['src'], cmdargs.seed), 0, None)
        full_outputs[label] = out
        if m.bridge and i == 0:
            samples = [ np.clip(m(pairs['src'], 1000 + k), 0, None) for k in range(8) ]
            for k in range(3):
                columns.append((f'{label}\nsample {k + 1}', samples[k][pick]))
            columns.append((f'{label}\nmean of 8', np.mean(samples, axis = 0)[pick]))
        else:
            columns.append((label, out[pick]))

    (fig, axes) = plt.subplots(len(pick), len(columns),
                               figsize = (1.55 * len(columns), 1.75 * len(pick)))
    for (c, (name, img)) in enumerate(columns):
        w = jsel.window(img).cpu().numpy()
        (_, shape) = jsel.emds(img, sel['tgt'])
        for r in range(len(pick)):
            ax = axes[r][c]
            ax.imshow(np.log10(w[r] + 0.1), cmap = 'viridis', vmin = -1, vmax = 1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            title = f'{w[r].sum():.1f} GeV' + (f'\nshape {shape[r]:.3f}' if c > 0 else '')
            ax.set_title(title, fontsize = 5.5)
            if r == 0:
                ax.set_xlabel(name, fontsize = 5.5)
                ax.xaxis.set_label_position('top')
    fig.suptitle('Closure test jets, log10(E + 0.1): single samples (fixed seed) unless'
                 ' marked; shape EMD to T(J), unit-energy jets', fontsize = 7)
    fig.tight_layout()
    fig.savefig(cmdargs.figure, dpi = 150)
    plt.close(fig)

    # residuals by true tower energy, and the marginals
    base = os.path.splitext(cmdargs.figure)[0]
    w_t = jets.window(pairs['tgt'])
    classes = [ c for c in TOWER_CLASSES if c[0] != '=0' ]
    (fig, axes) = plt.subplots(1, 2, figsize = (7.5, 2.8))
    xs = np.arange(len(classes))
    width = 0.8 / (len(full_outputs) + 1)
    all_out = { 'identity': pairs['src'], **full_outputs }
    for (k, (label, out)) in enumerate(all_out.items()):
        err = (jets.window(out) - w_t)
        means, rmss = [], []
        for (_, lo, hi) in classes:
            sel_t = (w_t >= lo) & (w_t < hi) & (w_t > 0)
            v = err[sel_t]
            means.append(float(v.mean()))
            rmss.append(float(v.pow(2).mean().sqrt()))
        axes[0].bar(xs + (k - len(all_out) / 2 + 0.5) * width, means, width, label = label)
        axes[1].bar(xs + (k - len(all_out) / 2 + 0.5) * width, rmss, width, label = label)
    for (ax, t) in zip(axes, [ 'mean error, GeV', 'RMS error, GeV' ]):
        ax.set_xticks(xs)
        ax.set_xticklabels([ c[0] for c in classes ])
        ax.set_xlabel('true tower energy in T(J), GeV')
        ax.set_ylabel(t)
        ax.axhline(0, color = 'k', lw = 0.6)
        ax.grid(alpha = 0.3, axis = 'y')
    axes[0].legend(fontsize = 5)
    fig.tight_layout()
    fig.savefig(f'{base}_residuals.png', dpi = 150)
    plt.close(fig)

    o_t = jets.observables(pairs['tgt'])
    o_j = jets.observables(pairs['src'])
    (fig, axes) = plt.subplots(1, len(OBS_SPREAD), figsize = (2.1 * len(OBS_SPREAD), 2.4))
    for (ax, q) in zip(axes, OBS_SPREAD):
        vals = o_t[q][np.isfinite(o_t[q])]
        bins = np.linspace(*np.quantile(vals, [ 0.005, 0.995 ]), 31)
        ax.hist(vals, bins, histtype = 'step', color = 'k', lw = 1.6, density = True,
                label = 'T(J) (target)')
        ax.hist(o_j[q][np.isfinite(o_j[q])], bins, histtype = 'step', color = 'grey',
                ls = ':', density = True, label = 'J (identity)')
        for (label, out) in full_outputs.items():
            v = jets.observables(out)[q]
            ax.hist(v[np.isfinite(v)], bins, histtype = 'step', density = True, label = label)
        ax.set_title(q, fontsize = 7)
        ax.tick_params(labelsize = 6)
    axes[0].legend(fontsize = 5)
    fig.tight_layout()
    fig.savefig(f'{base}_distributions.png', dpi = 150)
    plt.close(fig)
    print(f'wrote {cmdargs.figure}, {base}_residuals.png, {base}_distributions.png')

def main():
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    cmdargs.runs = [ resolve(r) for r in cmdargs.runs ]
    if cmdargs.select:
        select(cmdargs, device)
    elif cmdargs.steps_check:
        steps_check(cmdargs, device)
    elif cmdargs.multi:
        multi(cmdargs, device)
    elif cmdargs.figure:
        figure(cmdargs, device)
    else:
        report(cmdargs, device)

if __name__ == '__main__':
    main()
