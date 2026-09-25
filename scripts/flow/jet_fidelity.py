#!/usr/bin/env python
"""Single-event fidelity of the extracted jets: per-jet estimates of the
jet energy and substructure against truth, and the energy mover's distance
between the extracted and the true jet.

Each jet is the R = 0.4 cone around the true leading-jet axis of a val
(PYTHIA) or JEWEL mixed event, towers as constituents, as in
substructure.py (whose observables are used). Models are given as

    LABEL=uvcgan                 the published UVCGAN-S generator (EMA)
    LABEL=single:RUN:DECODE      a flow run at its selection setting
                                 (fm_common.SELECTION), read `direct` or as
                                 `mixture` - background
    LABEL=sampler:RUN            a posterior sampler (conditional CFM,
                                 postflow): K samples at 8 NFE, read out as
                                     sample      one sample
                                     mean-image  the observables of the mean
                                                 image of the K samples
                                     mean-obs    each observable averaged
                                                 over the samples

(RUN under OUTDIR/sphenix/flow, its val-selected checkpoint, EMA). The
first pass (`--models`) stores the per-jet values of every model in
OUTDIR/sphenix/flow/jet_fidelity/LABEL_{val,jewel}.npz, so that models can
be computed in parallel jobs; `--report` makes the tables and the figure:

    per read-out and observable   per-jet resolution sigma(estimate - truth)
                                  / sigma(truth), correlation, relative shift
                                  of the mean, W1 / sigma(truth), and the
                                  JEWEL - PYTHIA difference of the mean for
                                  jets of true cone energy 20-30 GeV as a
                                  fraction of the true difference
    per read-out                  `jer_cal` of the cone energy (as
                                  eval_val_truth.py, on these events) and the
                                  energy mover's distance (Komiske, Metodiev,
                                  Thaler), EMD = min_f sum f_ij dR_ij / R +
                                  |E - E'| with R = 0.4: the least energy x
                                  angle needed to turn one jet into the other,
                                  small only if energy and shape both agree;
                                  and its shape part `emd_shape`, the EMD
                                  after scaling the extracted jet to the true
                                  energy (GeV at the true energy)
    samplers, mean-obs            the per-jet error bars (spread of the
                                  samples): share of jets within +-1 sigma
                                  (68% if honest), correlation of sigma with
                                  |error|

    jet_fidelity.py --models SPEC [SPEC ...] [--samples 16] [--n-events 10000]
    jet_fidelity.py --report [--figure docs/flow/jet_fidelity.png]
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

R_EMD    = 0.4
TRUTHS   = [ 'val', 'jewel' ]
READOUTS = [ 'image', 'sample', 'mean-image', 'mean-obs' ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Single-event fidelity')
    parser.add_argument('--models', nargs = '*', default = [])
    parser.add_argument('--report', action = 'store_true')
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
    return (method, net, step)

@torch.no_grad()
def one_sample(dec, embed, batch, device):
    out = []
    for start in range(0, len(embed), batch):
        m = torch.from_numpy(embed[start:start + batch]).to(device).float()
        out.append(dec(m.unsqueeze(1))[:, 1].cpu())
    return torch.cat(out)

# --- per-jet values

def jet_values(img, truth, idx, geo, emd, t_patch, with_emd = True):
    """Observables, cone energy (with negative towers, as the jet scores)
    and EMD to truth of the jets idx of the images img."""
    p = geo.patches(img, truth, idx)
    v = observables(p, truth, idx, geo)
    v['e_cone'] = truth.at_axis(img.to(truth.row.device))[idx].cpu().numpy()
    if with_emd:
        (v['emd'], v['emd_shape']) = emd(p, t_patch)
    return v

def compute_model(spec, cmdargs, device):
    # pylint: disable=too-many-locals
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

        (embed, signal) = ev.load_pairs(
            os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0,
            truth = truth_name
        )
        (embed, signal) = (embed[:cmdargs.n_events], signal[:cmdargs.n_events])
        truth = ev.Truth(embed, signal, ev.cone_kernel(ev.R_JET), 10.0, device)
        idx   = torch.nonzero(truth.jets).flatten()
        sig   = torch.from_numpy(signal).float()
        t_patch = geo.patches(sig, truth, idx)

        tpath = os.path.join(cmdargs.dir, f'truth_{truth_name}.npz')
        if not os.path.exists(tpath):
            t = observables(t_patch, truth, idx, geo)
            t['e_cone'] = truth.e_true[idx].cpu().numpy()
            np.savez_compressed(tpath, **t)

        out = { 'kind' : kind, 'spec' : spec }

        if kind in ('uvcgan', 'single'):
            if kind == 'uvcgan':
                img = uvcgan_images(embed, cmdargs.batch, device)
            else:
                img = decoded_images(parts[1], parts[2], embed,
                                     cmdargs.batch, device)
            for (k, v) in jet_values(img, truth, idx, geo, emd,
                                     t_patch).items():
                out[f'image|{k}'] = v

        elif kind == 'sampler':
            (method, net, step) = load_sampler(parts[1], device)
            out['step'] = step
            dec  = fc.Decomposer(method, net, nfe = cmdargs.nfe, samples = 1,
                                 seed = 1)
            per  = {}
            mean = None

            for k in range(cmdargs.samples):
                img  = one_sample(dec, embed, cmdargs.batch, device)
                mean = img / cmdargs.samples if mean is None \
                    else mean + img / cmdargs.samples
                v = jet_values(img, truth, idx, geo, emd, t_patch,
                               with_emd = (k == 0))
                if k == 0:
                    for (q, x) in v.items():
                        out[f'sample|{q}'] = x
                for (q, x) in v.items():
                    if not q.startswith('emd'):
                        per.setdefault(q, []).append(x)
                print(f'{label} {truth_name}: sample {k + 1}/{cmdargs.samples}',
                      flush = True)

            for (q, x) in jet_values(mean, truth, idx, geo, emd,
                                     t_patch).items():
                out[f'mean-image|{q}'] = x

            with np.errstate(all = 'ignore'):
                for (q, xs) in per.items():
                    s = np.stack(xs)
                    out[f'mean-obs|{q}'] = np.nanmean(s, axis = 0)
                    out[f'mean-obs-sd|{q}'] = np.nanstd(s, axis = 0)
            out['samples'] = cmdargs.samples
        else:
            raise ValueError(f"unknown model kind '{kind}'")

        np.savez_compressed(path, **out)
        print(f'wrote {path}', flush = True)

# --- report

def jer_cal(y, x):
    b = np.mean((x - x.mean()) * (y - y.mean())) / x.var()
    a = y.mean() - b * x.mean()
    return float((y - a - b * x).std() / b)

def compare(x, t):
    ok = np.isfinite(x) & np.isfinite(t)
    (x, t) = (x[ok], t[ok])
    sd = t.std()
    return {
        'rel_res'   : float((x - t).std() / sd),
        'corr'      : float(np.corrcoef(x, t)[0, 1]),
        'rel_shift' : float((x.mean() - t.mean()) / abs(t.mean())),
        'w1_sigma'  : float(wasserstein_distance(x, t) / sd),
    }

def load_all(directory):
    models = {}
    truth  = {}
    for t in TRUTHS:
        truth[t] = dict(np.load(os.path.join(directory, f'truth_{t}.npz')))
    for path in sorted(glob.glob(os.path.join(directory, '*_val.npz'))):
        label = os.path.basename(path)[:-len('_val.npz')]
        if label == 'truth':
            continue
        models[label] = {
            t : dict(np.load(os.path.join(directory, f'{label}_{t}.npz'),
                             allow_pickle = True)) for t in TRUTHS
        }
    return (models, truth)

def report(cmdargs):
    # pylint: disable=too-many-locals
    (models, truth) = load_all(cmdargs.dir)
    rows  = []
    jets  = []
    means = []

    for (label, per_truth) in models.items():
        for (t, d) in per_truth.items():
            tt = truth[t]
            window = (tt['e_cone'] >= WINDOW[0]) & (tt['e_cone'] < WINDOW[1])

            for r in READOUTS:
                if f'{r}|et' not in d:
                    continue
                name = f'{label}' if r == 'image' else f'{label}, {r}'
                row  = {
                    'truth' : t, 'model' : label, 'readout' : r,
                    'name' : name, 'n_jets' : len(tt['e_cone']),
                    'jer_cal' : jer_cal(d[f'{r}|e_cone'], tt['e_cone']),
                }
                if f'{r}|emd' in d:
                    e = d[f'{r}|emd']
                    f = d[f'{r}|emd_shape']
                    row.update({ 'emd_mean' : float(e.mean()),
                                 'emd_median' : float(np.median(e)),
                                 'emd_p90' : float(np.quantile(e, 0.9)),
                                 'emd_shape_mean' : float(np.nanmean(f)),
                                 'emd_shape_median' : float(np.nanmedian(f)) })
                jets.append(row)

                for q in OBSERVABLES:
                    x = d[f'{r}|{q}']
                    c = compare(x, tt[q])
                    if r == 'mean-obs':
                        sd  = d[f'mean-obs-sd|{q}']
                        err = np.abs(x - tt[q])
                        ok  = np.isfinite(err) & np.isfinite(sd) & (sd > 0)
                        c['within_1sd'] = float(np.mean(err[ok] <= sd[ok]))
                        c['sd_err_corr'] = float(
                            np.corrcoef(sd[ok], err[ok])[0, 1])
                    rows.append({ 'truth' : t, 'model' : label,
                                  'readout' : r, 'name' : name,
                                  'observable' : q, **c })
                    means.append({ 'truth' : t, 'name' : name,
                                   'observable' : q,
                                   'mean_window' : float(np.nanmean(x[window])) })

        for (t, tt) in truth.items():
            window = (tt['e_cone'] >= WINDOW[0]) & (tt['e_cone'] < WINDOW[1])
            for q in OBSERVABLES:
                means.append({ 'truth' : t, 'name' : 'truth', 'observable' : q,
                               'mean_window' : float(np.nanmean(tt[q][window])) })

    obs  = pd.DataFrame(rows)
    jet  = pd.DataFrame(jets)
    mod  = pd.DataFrame(means).drop_duplicates([ 'truth', 'name', 'observable' ])
    pv   = mod.pivot_table(index = [ 'name', 'observable' ], columns = 'truth',
                           values = 'mean_window')
    pv['delta'] = pv['jewel'] - pv['val']
    true_delta  = pv.loc['truth', 'delta']
    pv['fraction'] = [ d / true_delta[q] for ((_, q), d) in pv['delta'].items() ]

    obs.to_csv(f'{cmdargs.out}.csv', index = False)
    jet.to_csv(f'{cmdargs.out}_jets.csv', index = False)
    pv.reset_index().to_csv(f'{cmdargs.out}_modification.csv', index = False)

    with pd.option_context('display.width', 250, 'display.max_rows', 200):
        print('jet energy and EMD (GeV) per model and read-out:')
        print(jet.set_index([ 'truth', 'name' ])
              [[ 'n_jets', 'jer_cal', 'emd_mean', 'emd_median', 'emd_p90',
                 'emd_shape_mean', 'emd_shape_median' ]]
              .round(3).to_string())
        print('\nper-jet resolution sigma(estimate - truth) / sigma(truth):')
        print(obs.pivot_table(index = [ 'truth', 'name' ],
                              columns = 'observable', values = 'rel_res')
              [OBSERVABLES].round(3).to_string())
        print('\nW1 / sigma(truth) of the distributions:')
        print(obs.pivot_table(index = [ 'truth', 'name' ],
                              columns = 'observable', values = 'w1_sigma')
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
              ' fraction of the true difference:')
        print(pv['fraction'].unstack()[OBSERVABLES].round(2).to_string())
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
                'observable').reindex(show)['rel_res']
            ax.bar(np.arange(len(show)) + (k - len(names) / 2 + 0.5) * width,
                   v.values, width, label = name, color = cmap(k % 20))
        ax.set_xticks(np.arange(len(show)))
        ax.set_xticklabels(show)
        ax.set_ylabel(f'{t}: sigma(estimate - truth) / sigma(truth)')
        ax.set_ylim(0, 1.3)
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
                 f' axis, up to {n} jets per set (lower is better)',
                 fontsize = 9)
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
