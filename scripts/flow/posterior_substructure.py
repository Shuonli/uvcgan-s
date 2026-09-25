#!/usr/bin/env python
"""Per-jet estimates from posterior samples: does a sampler give the best
single-event fidelity for jet energy and substructure?

Conditional CFM draws K plausible (background, jet) splits of each mixed
event. For every jet (the R = 0.4 cone around the true axis, as in
substructure.py) and observable, compare per-jet estimators against truth:

    sample      the observable of one sample
    mean-image  the observable of the average image of the K samples
    mean-obs    the average of the observable over the K samples (the
                per-event posterior mean of that observable)
    median-obs  the median of the observable over the samples

and check the per-jet error bars: the spread of the observable over the
samples against the actual error (share of jets within +-1 sigma, which is
68% for honest error bars, and the correlation of sigma with |error|). For
reference, the same per-jet resolution of UVCGAN-S and of the OT-CFM
read-outs from substructure.py.

    posterior_substructure.py [--run ext_condcfm_s0] [--samples 16]
                              [--n-events 10000]
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import fm_common as fc
from readout_test import fc_best_step
from substructure import Geometry, observables, OBSERVABLES

ev = fc.ev

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Posterior per-jet estimates')
    parser.add_argument('--run', default = 'ext_condcfm_s0')
    parser.add_argument('--samples', type = int, default = 16)
    parser.add_argument('--nfe', type = int, default = 8)
    parser.add_argument('--n-events', type = int, default = 10000)
    parser.add_argument('--batch', type = int, default = 500)
    parser.add_argument('--out', default = None)
    return parser.parse_args()

@torch.no_grad()
def sample_images(dec, embed, batch, device):
    out = []
    for start in range(0, len(embed), batch):
        m = torch.from_numpy(embed[start:start + batch]).to(device).float()
        out.append(dec(m.unsqueeze(1))[:, 1].cpu())
    return torch.cat(out)

def resolution(x, t):
    ok = np.isfinite(x) & np.isfinite(t)
    return (float((x[ok] - t[ok]).std() / t[ok].std()),
            float(np.corrcoef(x[ok], t[ok])[0, 1]),
            float((x[ok] - t[ok]).mean() / abs(t[ok].mean())))

def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    geo     = Geometry(device)

    run  = os.path.join(fc.out_root(), cmdargs.run)
    step = fc_best_step(run)
    ckpt = [ c for c in fc.list_checkpoints(run)
             if c.endswith(f'step_{step:08d}.pt') ][0]
    (method, net, _, _) = fc.load_run(run, ckpt, device, 'ema')
    rows = []

    for truth_name in [ 'val', 'jewel' ]:
        (embed, signal) = ev.load_pairs(
            os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0,
            truth = truth_name
        )
        (embed, signal) = (embed[:cmdargs.n_events], signal[:cmdargs.n_events])
        truth = ev.Truth(embed, signal, ev.cone_kernel(ev.R_JET), 10.0, device)
        idx   = torch.nonzero(truth.jets).flatten()

        t_obs = observables(
            geo.patches(torch.from_numpy(signal).float(), truth, idx),
            truth, idx, geo
        )

        dec  = fc.Decomposer(method, net, nfe = cmdargs.nfe, samples = 1,
                             seed = 1)
        per  = { q : [] for q in OBSERVABLES }
        mean = None

        for k in range(cmdargs.samples):
            img = sample_images(dec, embed, cmdargs.batch, device)
            mean = img / cmdargs.samples if mean is None \
                else mean + img / cmdargs.samples
            o = observables(geo.patches(img, truth, idx), truth, idx, geo)
            for q in OBSERVABLES:
                per[q].append(o[q])
            print(f'{truth_name}: sample {k + 1}/{cmdargs.samples}',
                  flush = True)

        m_obs = observables(geo.patches(mean, truth, idx), truth, idx, geo)

        for q in OBSERVABLES:
            s = np.stack(per[q])                     # (K, jets)
            t = t_obs[q]
            with np.errstate(all = 'ignore'):
                est = {
                    'sample'     : s[0],
                    'mean-image' : m_obs[q],
                    'mean-obs'   : np.nanmean(s, axis = 0),
                    'median-obs' : np.nanmedian(s, axis = 0),
                }
                sigma = np.nanstd(s, axis = 0)
            for (name, x) in est.items():
                (res, corr, shift) = resolution(x, t)
                row = { 'truth' : truth_name, 'observable' : q,
                        'estimator' : name, 'rel_res' : res, 'corr' : corr,
                        'rel_shift' : shift }
                if name == 'mean-obs':
                    err = np.abs(x - t)
                    ok  = np.isfinite(err) & np.isfinite(sigma) & (sigma > 0)
                    row['within_1sigma'] = float(np.mean(err[ok] <= sigma[ok]))
                    row['sigma_err_corr'] = float(
                        np.corrcoef(sigma[ok], err[ok])[0, 1])
                rows.append(row)

    df  = pd.DataFrame(rows)
    out = cmdargs.out or os.path.join(fc.out_root(), 'posterior_substructure.csv')
    df.to_csv(out, index = False)

    with pd.option_context('display.width', 200):
        print('\nper-jet resolution sigma(estimate - truth) / sigma(truth):')
        print(df.pivot_table(index = [ 'truth', 'estimator' ],
                             columns = 'observable', values = 'rel_res')
              [OBSERVABLES].round(3).to_string())
        print('\nper-jet error bars of mean-obs: share within +-1 sigma'
              ' (68% if honest), correlation of sigma with |error|:')
        x = df[df.estimator == 'mean-obs']
        print(x.pivot_table(index = 'truth', columns = 'observable',
                            values = 'within_1sigma')[OBSERVABLES]
              .round(2).to_string())
        print(x.pivot_table(index = 'truth', columns = 'observable',
                            values = 'sigma_err_corr')[OBSERVABLES]
              .round(2).to_string())
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
