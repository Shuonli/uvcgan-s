#!/usr/bin/env python
"""Single-event fidelity of the extracted jets: per-jet jet energy and
substructure against truth, and the energy mover's distance between the
extracted and the true jet.

Each jet is the R = 0.4 cone around the true leading-jet axis of a mixed
event, towers as constituents, as in substructure.py (whose observables are
used). Events: val (PYTHIA in HIJING) 0 .. n-1 for the development scores,
val n .. 2n-1 as the calibration set, JEWEL 0 .. n-1 as the frozen test.
Models are given as

    LABEL=uvcgan                 the published UVCGAN-S generator (EMA)
    LABEL=single:RUN:DECODE      a flow run at its selection setting
                                 (fm_common.SELECTION), read `direct` or as
                                 `mixture` - background
    LABEL=sampler:RUN            a posterior sampler (conditional CFM,
                                 postflow): K samples at 8 NFE

(RUN under OUTDIR/sphenix/flow, its val-selected checkpoint, EMA network).
Read-outs:

    image       a single-image model's output, raw
    thr0.5      the same, towers below 0.5 GeV dropped: the one post-
                processing applied identically to every single-image model,
                compared with the truth thresholded the same way
    sample      one posterior sample
    mean-image  the observables of the mean image of the K samples
    mean-obs    each observable averaged over the K samples

The first pass (`--models`) stores per-jet values in
OUTDIR/sphenix/flow/jet_fidelity/LABEL_{val,jewel}.npz, so that models can
be computed in parallel jobs; `--report` makes the tables and the figure:

    jet energy    raw response (`jes`, mean ratio), raw bias and RMSE; and
                  after a linear calibration fitted on the calibration set
                  (extracted = a + b true, inverted) and frozen: its bias,
                  resolution (sigma) and RMSE on val and on JEWEL, GeV;
                  `jer_cal` (the in-sample calibrated resolution of
                  eval_val_truth.py) for continuity. Always against the full
                  true cone energy.
    observables   bias, RMSE, sigma(estimate - truth), each in units of
                  sigma(truth); correlation; W1 / sigma(truth) of the
                  distributions; the JEWEL - PYTHIA difference of the mean for
                  jets of true cone energy 20-30 GeV as a fraction of the true
                  difference
    EMD           the energy mover's distance (Komiske, Metodiev, Thaler),
                  EMD = min_f sum f_ij dR_ij / R + |E - E'|, R = 0.4: the least
                  energy x angle that turns one jet into the other, small only
                  if energy and shape agree; `emd_shape`: the EMD after
                  scaling the extracted jet to the true energy
    samplers      the per-jet error bars of mean-obs (spread of the samples):
                  share of jets within +-1 sigma (68% if honest), correlation
                  of sigma with |error|

    jet_fidelity.py --models SPEC [SPEC ...] [--samples 16] [--n-events 10000]
    jet_fidelity.py --report [--show LABEL,...] [--figure PNG]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

import fm_common as fc
from readout_test import fc_best_step, PUBLISHED
from substructure import Geometry, observables, OBSERVABLES, WINDOW, \
    decoded_images

ev = fc.ev

R_EMD     = 0.4
THRESHOLD = 0.5
TRUTHS    = [ 'val', 'jewel' ]
READOUTS  = [ 'image', 'thr0.5', 'sample', 'mean-image', 'mean-obs' ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Single-event fidelity')
    parser.add_argument('--models', nargs = '*', default = [])
    parser.add_argument('--report', action = 'store_true')
    parser.add_argument('--show', default = None,
        help = 'comma separated labels to report (default: all)')
    parser.add_argument('--samples', type = int, default = 16)
    parser.add_argument('--nfe', type = int, default = 8)
    parser.add_argument('--n-events', type = int, default = 10000)
    parser.add_argument('--batch', type = int, default = 500)
    parser.add_argument('--force', action = 'store_true')
    parser.add_argument('--dir', default = os.path.join(
        fc.out_root(), 'jet_fidelity'))
    parser.add_argument('--out', default = os.path.join(
        fc.out_root(), 'jet_fidelity'))
    parser.add_argument('--figure', default = None)
    return parser.parse_args()

def threshold(img):
    return torch.where(img > THRESHOLD, img, torch.zeros_like(img))

# --- energy mover's distance

class EMD:
    """Per-jet EMD between (N, 9, 9) patches on the cone towers."""

    def __init__(self, geo):
        # pylint: disable=import-outside-toplevel
        import ot
        self.emd2 = ot.emd2

        self.mask = geo.mask.flatten().cpu().numpy()
        deta = geo.deta.flatten().cpu().numpy()[self.mask].astype(np.float64)
        dphi = geo.dphi.flatten().cpu().numpy()[self.mask].astype(np.float64)
        n    = len(deta)

        # the extra node takes the energy difference at no transport cost
        self.cost = np.zeros((n + 1, n + 1))
        self.cost[:n, :n] = np.sqrt(
            (deta[:, None] - deta[None, :])**2
            + (dphi[:, None] - dphi[None, :])**2
        ) / R_EMD

    def __call__(self, pa, pb):
        """(EMD, shape part: the EMD of the extracted jet scaled to the
        true energy), both (N,), GeV."""
        a = pa.clamp(min = 0).flatten(1).cpu().numpy().astype(np.float64)
        b = pb.clamp(min = 0).flatten(1).cpu().numpy().astype(np.float64)
        (a, b) = (a[:, self.mask], b[:, self.mask])

        full  = np.empty(len(a))
        shape = np.full(len(a), np.nan)
        for i in range(len(a)):
            (ea, eb) = (a[i].sum(), b[i].sum())
            if min(ea, eb) <= 0:
                full[i] = abs(ea - eb)
                continue
            wa = np.append(a[i], max(eb - ea, 0.0))
            wb = np.append(b[i], max(ea - eb, 0.0))
            full[i]  = self.emd2(wa, wb, self.cost) + abs(ea - eb)
            shape[i] = self.emd2(
                np.append(a[i] * eb / ea, 0.0), np.append(b[i], 0.0),
                self.cost
            )

        return (full, shape)

# --- the models' images

@torch.no_grad()
def uvcgan_images(embed, batch, device):
    # pylint: disable=import-outside-toplevel
    from uvcgan_s.config import Args
    from uvcgan_s.cgan   import construct_model

    outdir = os.environ.get('UVCGAN_S_OUTDIR', 'outdir')
    args   = Args.load(os.path.join(outdir, PUBLISHED))
    model  = construct_model(args.savedir, args.config, is_train = False,
                             device = device)
    model.load(None)
    (gen, norm) = (model.models.ema_gen_ba.eval(), model.data_norm)

    out = []
    for start in range(0, len(embed), batch):
        m = torch.from_numpy(embed[start:start + batch]).to(device)
        out.append(
            norm.denormalize(gen(norm.normalize(m.float().unsqueeze(1))))
            [:, 1].cpu()
        )
    return torch.cat(out)

def load_sampler(run, device):
    run  = os.path.join(fc.out_root(), run)
    step = fc_best_step(run)
    ckpt = [ c for c in fc.list_checkpoints(run)
             if c.endswith(f'step_{step:08d}.pt') ][0]
    (method, net, _, _) = fc.load_run(run, ckpt, device, 'ema')
    assert method.name in fc.SAMPLERS, method.name
    return (method, net, step, run)

@torch.no_grad()
def one_sample(dec, embed, batch, device):
    out = []
    for start in range(0, len(embed), batch):
        m = torch.from_numpy(embed[start:start + batch]).to(device).float()
        out.append(dec(m.unsqueeze(1))[:, 1].cpu())
    return torch.cat(out)

# --- events

class Events:
    """Mixtures, truth and jets of one set of events."""

    def __init__(self, truth_name, start, n, device, geo):
        (embed, signal) = ev.load_pairs(
            os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0,
            truth = truth_name
        )
        self.embed  = embed [start:start + n]
        self.signal = signal[start:start + n]
        self.truth  = ev.Truth(self.embed, self.signal,
                               ev.cone_kernel(ev.R_JET), 10.0, device)
        self.idx    = torch.nonzero(self.truth.jets).flatten()
        self.geo    = geo

        sig = torch.from_numpy(self.signal).float()
        self.t_patch = {
            'full' : geo.patches(sig, self.truth, self.idx),
            'thr'  : geo.patches(threshold(sig), self.truth, self.idx),
        }

    def e_cone(self, img):
        return self.truth.at_axis(img.to(self.truth.row.device))[self.idx] \
            .cpu().numpy()

    def values(self, img, emd, ref = 'full', with_emd = True):
        """Observables, cone energy (with negative towers, as the jet
        scores) and EMD to the truth `ref` of the jets of the images."""
        p = self.geo.patches(img, self.truth, self.idx)
        v = observables(p, self.truth, self.idx, self.geo)
        v['e_cone'] = self.e_cone(img)
        if with_emd:
            (v['emd'], v['emd_shape']) = emd(p, self.t_patch[ref])
        return v

def truth_file(cmdargs, truth_name, events, calib):
    path = os.path.join(cmdargs.dir, f'truth_{truth_name}.npz')
    if os.path.exists(path) and not cmdargs.force:
        return

    geo = events.geo
    out = {}
    for (ref, patch) in events.t_patch.items():
        for (q, x) in observables(patch, events.truth, events.idx,
                                  geo).items():
            out[f'{ref}|{q}'] = x
    out['e_cone'] = events.truth.e_true[events.idx].cpu().numpy()
    if calib is not None:
        out['calib|e_cone'] = calib.truth.e_true[calib.idx].cpu().numpy()
    np.savez_compressed(path, **out)

# --- per-model values

def single_values(kind, parts, events, calib, emd, cmdargs, device):
    def images(embed):
        if kind == 'uvcgan':
            return uvcgan_images(embed, cmdargs.batch, device)
        return decoded_images(parts[1], parts[2], embed, cmdargs.batch,
                              device)

    img = images(events.embed)
    out = {}
    for (k, v) in events.values(img, emd).items():
        out[f'image|{k}'] = v
    for (k, v) in events.values(threshold(img), emd, 'thr').items():
        out[f'thr0.5|{k}'] = v

    if calib is not None:
        img = images(calib.embed)
        out['image|calib_e_cone']  = calib.e_cone(img)
        out['thr0.5|calib_e_cone'] = calib.e_cone(threshold(img))

    return out

def sampler_calibration(run, step, calib, cmdargs, dec_args, device):
    """Calibration-set cone energies of the mean of K samples and of one
    sample: from fm_eval.py's per-event energies of the same checkpoint and
    setting where present (val events n .. 2n-1 of its 20k), else drawn."""
    (method, net) = dec_args
    start = len(calib.embed)
    out   = {}

    for (name, k) in [ ('mean-image', cmdargs.samples), ('sample', 1) ]:
        path = os.path.join(
            run, 'evals', 'val_truth',
            f's{step:08d}_ema_midpoint{cmdargs.nfe}_direct_x{k}.npy'
        )
        if os.path.exists(path):
            e = np.load(path)[start:2 * start]
            out[f'{name}|calib_e_cone'] = e[calib.idx.cpu().numpy()]
            continue

        print(f'{path} missing: drawing the calibration set', flush = True)
        dec  = fc.Decomposer(method, net, nfe = cmdargs.nfe, samples = k,
                             seed = 2)
        img  = one_sample(dec, calib.embed, cmdargs.batch, device)
        out[f'{name}|calib_e_cone'] = calib.e_cone(img)

    out['mean-obs|calib_e_cone'] = out['mean-image|calib_e_cone']
    return out

def sampler_values(label, parts, events, calib, emd, cmdargs, device):
    # pylint: disable=too-many-locals
    (method, net, step, run) = load_sampler(parts[1], device)
    dec  = fc.Decomposer(method, net, nfe = cmdargs.nfe, samples = 1,
                         seed = 1)
    out  = { 'step' : step, 'samples' : cmdargs.samples }
    per  = {}
    mean = None

    for k in range(cmdargs.samples):
        img  = one_sample(dec, events.embed, cmdargs.batch, device)
        mean = img / cmdargs.samples if mean is None \
            else mean + img / cmdargs.samples
        v = events.values(img, emd, with_emd = (k == 0))
        if k == 0:
            for (q, x) in v.items():
                out[f'sample|{q}'] = x
        for (q, x) in v.items():
            if not q.startswith('emd'):
                per.setdefault(q, []).append(x)
        print(f'{label}: sample {k + 1}/{cmdargs.samples}', flush = True)

    for (q, x) in events.values(mean, emd).items():
        out[f'mean-image|{q}'] = x

    with np.errstate(all = 'ignore'):
        for (q, xs) in per.items():
            s = np.stack(xs)
            out[f'mean-obs|{q}'] = np.nanmean(s, axis = 0)
            out[f'mean-obs-sd|{q}'] = np.nanstd(s, axis = 0)

    if calib is not None:
        out.update(sampler_calibration(run, step, calib, cmdargs,
                                       (method, net), device))
    return out

def compute_model(spec, cmdargs, device):
    (label, rest) = spec.split('=', 1)
    parts = rest.split(':')
    kind  = parts[0]
    geo   = Geometry(device)
    emd   = EMD(geo)
    os.makedirs(cmdargs.dir, exist_ok = True)

    for truth_name in TRUTHS:
        path = os.path.join(cmdargs.dir, f'{label}_{truth_name}.npz')
        if os.path.exists(path) and not cmdargs.force:
            print(f'{path} exists, skipped')
            continue

        n      = cmdargs.n_events
        events = Events(truth_name, 0, n, device, geo)
        calib  = Events('val', n, n, device, geo) if truth_name == 'val' \
            else None
        truth_file(cmdargs, truth_name, events, calib)

        if kind in ('uvcgan', 'single'):
            out = single_values(kind, parts, events, calib, emd, cmdargs,
                                device)
        elif kind == 'sampler':
            out = sampler_values(f'{label} {truth_name}', parts, events,
                                 calib, emd, cmdargs, device)
        else:
            raise ValueError(f"unknown model kind '{kind}'")

        out.update({ 'kind' : kind, 'spec' : spec })
        np.savez_compressed(path, **out)
        print(f'wrote {path}', flush = True)

# --- report

def calibration(y, x):
    """(a, b) of extracted y = a + b true x."""
    b = np.mean((x - x.mean()) * (y - y.mean())) / x.var()
    return (y.mean() - b * x.mean(), b)

def energy_scores(y, x, cal):
    (a, b) = calibration(y, x)
    row = {
        'jes'      : float(np.mean(y / x)),
        'raw_bias' : float(np.mean(y - x)),
        'raw_rmse' : float(np.sqrt(np.mean((y - x)**2))),
        'jer_cal'  : float((y - a - b * x).std() / b),
    }
    if cal is not None:
        xhat = (y - cal[0]) / cal[1]
        d    = xhat - x
        row.update({ 'cal_bias' : float(d.mean()), 'cal_res' : float(d.std()),
                     'cal_rmse' : float(np.sqrt(np.mean(d**2))) })
    return row

def compare(x, t):
    ok = np.isfinite(x) & np.isfinite(t)
    (x, t) = (x[ok], t[ok])
    sd = t.std()
    d  = x - t
    return {
        'bias'      : float(d.mean()),
        'rmse'      : float(np.sqrt(np.mean(d**2))),
        'rel_bias'  : float(d.mean() / sd),
        'rel_rmse'  : float(np.sqrt(np.mean(d**2)) / sd),
        'rel_res'   : float(d.std() / sd),
        'corr'      : float(np.corrcoef(x, t)[0, 1]),
        'w1_sigma'  : float(wasserstein_distance(x, t) / sd),
    }

def load_all(directory, show):
    truth = { t : dict(np.load(os.path.join(directory, f'truth_{t}.npz')))
              for t in TRUTHS }
    models = {}
    for path in sorted(glob.glob(os.path.join(directory, '*_val.npz'))):
        label = os.path.basename(path)[:-len('_val.npz')]
        if (label == 'truth') or (show and label not in show):
            continue
        models[label] = {
            t : dict(np.load(os.path.join(directory, f'{label}_{t}.npz'),
                             allow_pickle = True)) for t in TRUTHS
        }
    if show:
        models = { k : models[k] for k in show if k in models }
    return (models, truth)

def report(cmdargs):
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    show = cmdargs.show.split(',') if cmdargs.show else None
    (models, truth) = load_all(cmdargs.dir, show)
    rows  = []
    jets  = []
    means = []

    for (label, per_truth) in models.items():
        cals = {}
        for r in READOUTS:
            key = f'{r}|calib_e_cone'
            if key in per_truth['val']:
                cals[r] = calibration(per_truth['val'][key],
                                      truth['val']['calib|e_cone'])

        for (t, d) in per_truth.items():
            tt = truth[t]
            for r in READOUTS:
                if f'{r}|et' not in d:
                    continue
                ref  = 'thr' if r.startswith('thr') else 'full'
                name = label if r == 'image' else f'{label}, {r}'
                row  = { 'truth' : t, 'model' : label, 'readout' : r,
                         'name' : name, 'n_jets' : len(tt['e_cone']),
                         **energy_scores(d[f'{r}|e_cone'], tt['e_cone'],
                                         cals.get(r)) }
                if f'{r}|emd' in d:
                    (e, f) = (d[f'{r}|emd'], d[f'{r}|emd_shape'])
                    row.update({ 'emd_mean' : float(e.mean()),
                                 'emd_median' : float(np.median(e)),
                                 'emd_shape_mean' : float(np.nanmean(f)) })
                jets.append(row)

                window = (tt['e_cone'] >= WINDOW[0]) & (tt['e_cone'] < WINDOW[1])
                for q in OBSERVABLES:
                    x = d[f'{r}|{q}']
                    c = compare(x, tt[f'{ref}|{q}'])
                    if r == 'mean-obs':
                        sd  = d[f'mean-obs-sd|{q}']
                        err = np.abs(x - tt[f'{ref}|{q}'])
                        ok  = np.isfinite(err) & np.isfinite(sd) & (sd > 0)
                        c['within_1sd'] = float(np.mean(err[ok] <= sd[ok]))
                        c['sd_err_corr'] = float(
                            np.corrcoef(sd[ok], err[ok])[0, 1])
                    rows.append({ 'truth' : t, 'model' : label,
                                  'readout' : r, 'name' : name,
                                  'reference' : ref, 'observable' : q, **c })
                    means.append({ 'truth' : t, 'name' : name,
                                   'reference' : ref, 'observable' : q,
                                   'mean_window' : float(np.nanmean(x[window])) })

    for (t, tt) in truth.items():
        window = (tt['e_cone'] >= WINDOW[0]) & (tt['e_cone'] < WINDOW[1])
        for ref in [ 'full', 'thr' ]:
            for q in OBSERVABLES:
                means.append({ 'truth' : t, 'name' : f'truth ({ref})',
                               'reference' : ref, 'observable' : q,
                               'mean_window' : float(np.nanmean(
                                   tt[f'{ref}|{q}'][window])) })

    obs = pd.DataFrame(rows)
    jet = pd.DataFrame(jets)
    mod = pd.DataFrame(means)
    pv  = mod.pivot_table(index = [ 'name', 'reference', 'observable' ],
                          columns = 'truth', values = 'mean_window')
    pv['delta'] = pv['jewel'] - pv['val']
    truth_delta = { ref : pv.loc[(f'truth ({ref})', ref), 'delta']
                    for ref in [ 'full', 'thr' ] }
    pv['fraction'] = [ d / truth_delta[ref][q]
                       for ((_, ref, q), d) in pv['delta'].items() ]
    pv = pv.reset_index()

    obs.to_csv(f'{cmdargs.out}.csv', index = False)
    jet.to_csv(f'{cmdargs.out}_jets.csv', index = False)
    pv.to_csv(f'{cmdargs.out}_modification.csv', index = False)

    main_obs = [ 'mass', 'girth', 'ptd', 'zlead', 'zg', 'rg' ]
    with pd.option_context('display.width', 250, 'display.max_rows', 300):
        print('jet energy, GeV: raw response and, with the calibration fitted'
              ' on val events n..2n-1 and frozen, bias / resolution / RMSE;'
              ' EMD to truth (thr0.5 against the thresholded truth):')
        print(jet.set_index([ 'truth', 'name' ])
              [[ 'n_jets', 'jes', 'raw_bias', 'cal_bias', 'cal_res',
                 'cal_rmse', 'jer_cal', 'emd_mean', 'emd_shape_mean' ]]
              .round(3).to_string())
        for (value, title) in [
            ('rel_rmse', 'RMSE / sigma(truth)'),
            ('rel_bias', 'bias / sigma(truth)'),
            ('rel_res', 'sigma(estimate - truth) / sigma(truth)'),
            ('w1_sigma', 'W1 / sigma(truth) of the distributions'),
        ]:
            print(f'\n{title} (thr0.5 against the thresholded truth):')
            print(obs.pivot_table(index = [ 'truth', 'name' ],
                                  columns = 'observable', values = value)
                  [OBSERVABLES].round(3).to_string())
        x = obs[obs.readout == 'mean-obs']
        if len(x):
            print('\nsamplers, per-jet error bars of mean-obs: share within'
                  ' +-1 sd (68% if honest) | correlation of sd with |error|')
            print(x.pivot_table(index = [ 'truth', 'name' ],
                                columns = 'observable', values = 'within_1sd')
                  [OBSERVABLES].round(2).to_string())
            print(x.pivot_table(index = [ 'truth', 'name' ],
                                columns = 'observable', values = 'sd_err_corr')
                  [OBSERVABLES].round(2).to_string())
        print('\nJEWEL - PYTHIA of the mean, true cone energy 20-30 GeV, as a'
              ' fraction of the true difference (same truth reference):')
        print(pv.pivot_table(index = 'name', columns = 'observable',
                             values = 'fraction')[main_obs].round(2)
              .to_string())
    print(f'wrote {cmdargs.out}.csv, {cmdargs.out}_jets.csv,'
          f' {cmdargs.out}_modification.csv')

    if cmdargs.figure:
        plot(obs, jet, cmdargs.figure)

def plot(obs, jet, out):
    # pylint: disable=import-outside-toplevel,too-many-locals
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    show  = [ 'et', 'mass', 'girth', 'ptd', 'core', 'zlead', 'n1', 'zg', 'rg' ]
    names = list(dict.fromkeys(obs.name))
    (fig, axes) = plt.subplots(2, 2, figsize = (15, 8.5),
                               gridspec_kw = { 'width_ratios' : [ 3, 1 ] })
    width = 0.8 / len(names)
    cmap  = plt.get_cmap('tab20')

    for (row, t) in enumerate(TRUTHS):
        ax = axes[row][0]
        for (k, name) in enumerate(names):
            v = obs[(obs.truth == t) & (obs.name == name)].set_index(
                'observable').reindex(show)['rel_rmse']
            ax.bar(np.arange(len(show)) + (k - len(names) / 2 + 0.5) * width,
                   v.values, width, label = name, color = cmap(k % 20))
        ax.set_xticks(np.arange(len(show)))
        ax.set_xticklabels(show)
        ax.set_ylabel(f'{t}: per-jet RMSE / sigma(truth)')
        ax.set_ylim(0, 1.4)
        ax.grid(axis = 'y', alpha = 0.3)
        if row == 0:
            ax.legend(fontsize = 6, ncol = 3)

        ax = axes[row][1]
        j  = jet[jet.truth == t].set_index('name').reindex(names)
        has = [ k for (k, n) in enumerate(names)
                if np.isfinite(j.loc[n, 'emd_mean']) ]
        y   = np.arange(len(has))
        ax.barh(y - 0.2, j.emd_mean.values[has], 0.4, label = 'EMD',
                color = [ cmap(k % 20) for k in has ])
        ax.barh(y + 0.2, j.emd_shape_mean.values[has], 0.4,
                label = 'shape part', color = [ cmap(k % 20) for k in has ],
                alpha = 0.45, hatch = '//')
        ax.set_yticks(y)
        ax.set_yticklabels([ names[k] for k in has ], fontsize = 6)
        ax.invert_yaxis()
        ax.set_xlabel(f'{t}: mean per-jet EMD to truth, GeV')
        ax.grid(axis = 'x', alpha = 0.3)
        if row == 0:
            ax.legend(fontsize = 7)

    n = int(jet.n_jets.max())
    fig.suptitle('Single-event fidelity: R = 0.4 cone around the true jet'
                 f' axis, up to {n} jets per set (lower is better; thr0.5'
                 ' against the thresholded truth)', fontsize = 9)
    fig.tight_layout()
    fig.savefig(out, dpi = 110)
    print(f'wrote {out}')

def main():
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda') if cmdargs.models else None

    for spec in cmdargs.models:
        compute_model(spec, cmdargs, device)

    if cmdargs.report:
        report(cmdargs)

if __name__ == '__main__':
    main()
