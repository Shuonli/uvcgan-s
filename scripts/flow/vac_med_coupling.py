#!/usr/bin/env python
"""What would a minibatch OT coupling pair, between vacuum (PYTHIA) and
medium-modified (JEWEL) jets? A feasibility check for an unpaired
vacuum -> medium OT-CFM, before training anything.

Jets: the truth signal images of the val (PYTHIA) and test (JEWEL) pair
caches, leading jet >= 10 GeV inside |eta| < 0.7. Representations:

    full        the whole 24 x 64 image, log(E + 0.1), as the decomposition
                flows use it (jets at their own positions)
    centred     the 9 x 9 towers around the jet axis (R = 0.4 cone),
                log(E + 0.1)
    centred-lin the same towers in GeV

For each, exact minibatch OT between n PYTHIA and n JEWEL jets, and what
the pairs look like: distance between the paired jets' axes, rank
correlation of their cone energies, how often the JEWEL partner has less
energy (energy loss), and the correlation of their girths -- against a
random pairing.

    vac_med_coupling.py [--batches 256,1024] [--repeats 4]
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import ot as pot

import fm_common as fc
from substructure import Geometry, observables

ev = fc.ev

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Vacuum-medium coupling')
    parser.add_argument('--batches', default = '256,1024')
    parser.add_argument('--repeats', type = int, default = 4)
    parser.add_argument('--seed', type = int, default = 0)
    return parser.parse_args()

def jets(truth_name, device, geo):
    (embed, signal) = ev.load_pairs(
        os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0, truth = truth_name
    )
    truth = ev.Truth(embed, signal, ev.cone_kernel(ev.R_JET), 10.0, device)
    idx   = torch.nonzero(truth.jets).flatten()
    sig   = torch.from_numpy(signal).float()
    patch = geo.patches(sig, truth, idx)
    obs   = observables(patch, truth, idx, geo)

    return {
        'full'    : torch.log(sig[idx.cpu()] + 0.1).flatten(1),
        'centred' : torch.log(patch.clamp(min = 0) * geo.mask + 0.1)
                        .flatten(1).cpu(),
        'centred-lin' : (patch.clamp(min = 0) * geo.mask).flatten(1).cpu(),
        'row' : truth.row[idx].cpu().numpy(),
        'col' : truth.col[idx].cpu().numpy(),
        'et'  : obs['et'],
        'girth' : obs['girth'],
    }

def axis_distance(a, ia, b, ib):
    deta = (a['row'][ia] - b['row'][ib]) * ev.DETA
    dcol = np.abs(a['col'][ia] - b['col'][ib])
    dphi = np.minimum(dcol, fc.SHAPE[1] - dcol) * ev.DPHI
    return np.sqrt(deta**2 + dphi**2)

def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    geo     = Geometry(device)
    rng     = np.random.default_rng(cmdargs.seed)

    vac = jets('val', device, geo)
    med = jets('jewel', device, geo)
    print(f"jets: PYTHIA {len(vac['et'])} (median cone et"
          f" {np.median(vac['et']):.1f} GeV), JEWEL {len(med['et'])}"
          f" ({np.median(med['et']):.1f} GeV)", flush = True)

    rows = []
    for n in [ int(x) for x in cmdargs.batches.split(',') ]:
        for rep in range(cmdargs.repeats):
            iv = rng.choice(len(vac['et']), n, replace = False)
            im = rng.choice(len(med['et']), n, replace = False)

            plans = { 'random' : np.arange(n) }
            for rep_name in [ 'full', 'centred', 'centred-lin' ]:
                cost = torch.cdist(vac[rep_name][iv], med[rep_name][im]) ** 2
                plan = pot.emd(pot.unif(n), pot.unif(n),
                               cost.double().numpy(), numItermax = 10_000_000)
                plans[rep_name] = plan.argmax(axis = 1)

            for (name, j) in plans.items():
                (ev_, em) = (vac['et'][iv], med['et'][im][j])
                (gv, gm)  = (vac['girth'][iv], med['girth'][im][j])
                rows.append({
                    'batch' : n, 'repeat' : rep, 'coupling' : name,
                    'axis_dr' : float(axis_distance(vac, iv, med, im[j]).mean()),
                    'et_rank_corr' : float(spearmanr(ev_, em)[0]),
                    'frac_med_lower' : float(np.mean(em < ev_)),
                    'mean_et_loss' : float(np.mean(ev_ - em)),
                    'sd_et_loss' : float(np.std(ev_ - em)),
                    'girth_corr' : float(np.corrcoef(gv, gm)[0, 1]),
                })

    df = pd.DataFrame(rows)
    out = os.path.join(fc.out_root(), 'vac_med_coupling.csv')
    df.to_csv(out, index = False)

    # the quantile picture: sorted energies against sorted energies
    q = np.linspace(0.05, 0.95, 10)
    print('\ncone et quantiles, PYTHIA -> JEWEL (the monotone coupling):')
    print(' '.join(f'{a:.0f}->{b:.0f}' for (a, b) in zip(
        np.quantile(vac['et'], q), np.quantile(med['et'], q))))

    with pd.option_context('display.width', 200):
        print(df.groupby([ 'batch', 'coupling' ]).mean(numeric_only = True)
                .drop(columns = 'repeat').round(3).to_string())
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
