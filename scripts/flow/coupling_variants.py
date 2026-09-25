#!/usr/bin/env python
"""Which way of setting up the unpaired matching lets a mixture find its
own jet? Variants of coupling_diag.py, scored the same way against the
held-out truth of val mixtures (the truth only scores the pairing).

Targets are independently drawn HIJING and PYTHIA training events; a
batch holds n real mixtures and n targets. Set-ups:

    empty start      (mixture, empty) -> (HIJING, PYTHIA), as trained
    mixture start    (mixture, mixture) -> (HIJING, PYTHIA): the right
                     panel starts from the mixture too; each target still
                     comes as a (HIJING, PYTHIA) bundle
    mix -> PYTHIA    one panel: mixtures matched with PYTHIA events alone
    mix -> HIJING    one panel: mixtures matched with HIJING events alone,
                     the jet read as mixture - background

each with the difference measured in log units (log(E + 0.1), as trained)
or in GeV. Scores: `hit`, how often the matched PYTHIA event has its jet
where the mixture's jet is (or, for mix -> HIJING, the jet of mixture -
background); `corr`, the correlation of the matched jet's energy in the
true jet cone with the true jet energy; `jer_cal`.

    coupling_variants.py [--batches 256,1024] [--repeats 4]
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import fm_common as fc
from coupling_diag import exact_plan, score

ev = fc.ev

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Matching set-ups')
    parser.add_argument('--batches', default = '256,1024')
    parser.add_argument('--repeats', type = int, default = 4)
    parser.add_argument('--pool', type = int, default = 16384)
    parser.add_argument('--seed', type = int, default = 0)
    return parser.parse_args()

def psi(x):
    return torch.log(x + fc.BIAS)

def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng     = np.random.default_rng(cmdargs.seed)
    kernel  = ev.cone_kernel(ev.R_JET)

    norm = fc.Norm.load_or_fit(
        os.path.join(fc.out_root(), 'norm_n20000_seed0.json')
    )
    (embed, signal) = ev.load_pairs(
        os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0
    )
    bkg_pool = fc.read_random('background', cmdargs.pool, rng)
    sig_pool = fc.read_random('signal', cmdargs.pool, rng)
    rows = []

    for n in [ int(x) for x in cmdargs.batches.split(',') ]:
        for rep in range(cmdargs.repeats):
            pick  = rng.choice(len(embed), n, replace = False)
            truth = ev.Truth(embed[pick], signal[pick], kernel, 10.0, device)
            m = torch.from_numpy(embed[pick]).to(device).float()
            b = torch.from_numpy(bkg_pool[
                rng.choice(cmdargs.pool, n, replace = False)]).to(device)
            s = torch.from_numpy(sig_pool[
                rng.choice(cmdargs.pool, n, replace = False)]).to(device)

            # (start, target, what the matched target gives as the jet)
            setups = {
                'random' : (None, None, 'signal'),
                'empty start, log' : (
                    norm.state(m, torch.zeros_like(m)), norm.state(b, s),
                    'signal'),
                'empty start, GeV' : (
                    torch.stack((m, torch.zeros_like(m)), 1),
                    torch.stack((b, s), 1), 'signal'),
                'mixture start, log' : (
                    norm.state(m, m), norm.state(b, s), 'signal'),
                'mixture start, GeV' : (
                    torch.stack((m, m), 1), torch.stack((b, s), 1),
                    'signal'),
                'mix -> PYTHIA, log' : (psi(m), psi(s), 'signal'),
                'mix -> PYTHIA, GeV' : (m, s, 'signal'),
                'mix -> HIJING, log' : (psi(m), psi(b), 'subtract'),
                'mix -> HIJING, GeV' : (m, b, 'subtract'),
            }

            for (name, (x0, x1, read)) in setups.items():
                if x0 is None:
                    j = np.arange(n)
                else:
                    j = exact_plan(x0, x1).argmax(axis = 1)
                j = torch.from_numpy(np.asarray(j)).to(device)

                jet = s[j] if read == 'signal' else m - b[j]
                e_true = truth.e_true[truth.jets]
                e_pair = truth.at_axis(jet)[truth.jets]
                rows.append({
                    'batch' : n, 'repeat' : rep, 'setup' : name,
                    'corr' : float(np.corrcoef(
                        e_pair.cpu().numpy(), e_true.cpu().numpy())[0, 1]),
                    **score(jet, truth, kernel),
                })

            print(f'batch {n} repeat {rep} done', flush = True)

    df  = pd.DataFrame(rows)
    out = os.path.join(fc.out_root(), 'coupling_variants.csv')
    df.to_csv(out, index = False)

    with pd.option_context('display.width', 200):
        print(df.groupby([ 'batch', 'setup' ], sort = False)
                [[ 'hit', 'corr', 'jer_cal', 'jes' ]].mean().round(3)
                .to_string())
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
