#!/usr/bin/env python
"""Does a minibatch coupling pair a mixture with its own signal?

OT-CFM and SB-CFM learn from pairs (x0, x1) chosen by a minibatch plan
between independently drawn source and target events. For the
decomposition the pair only teaches the right thing if the target signal
looks like the signal inside the source mixture. This scores the plans
themselves, before any training, with the held-out truth of val mixtures
(the truth scores the pairing, it is never trained on):

    hit      the paired signal's leading jet lies within dR < 0.4 of the
             mixture's true jet (random pairing: ~ the cone's share of the
             acceptance)
    jer_cal  calibrated resolution of the paired signal's energy in the
             cone around the true jet -- what a flow that reproduced this
             pairing would score

Plans (squared Euclidean cost in the standardised state of fm_common):

    random   independent pairing (I-CFM)
    ot       exact OT between x0 = (psi(m), psi(0)) and x1 = (psi(b), psi(s))
             (candidate A)
    sb       entropic plan, reg = 2 sigma^2, one partner drawn per row
             (candidate B)
    ot-sum   exact OT with cost |psi(m) - psi(b + s)|^2: uses the additive
             mixing; not a candidate, shows what that knowledge buys

Decodings: `direct` (the paired signal) and `mixture` (m - the paired
background, additivity imposed).

    coupling_diag.py [--batches 256,1024,2048] [--repeats 4] [--sigma 1]
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import ot as pot

import fm_common as fc

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Score minibatch couplings')
    parser.add_argument('--batches', default = '256,1024,2048')
    parser.add_argument('--repeats', type = int, default = 4)
    parser.add_argument('--sigma', type = float, default = 1.0)
    parser.add_argument('--pool', type = int, default = 16384)
    parser.add_argument('--seed', type = int, default = 0)
    return parser.parse_args()

def exact_plan(x0, x1):
    cost = torch.cdist(x0.flatten(1), x1.flatten(1)) ** 2
    n    = x0.shape[0]
    plan = pot.emd(pot.unif(n), pot.unif(n), cost.double().cpu().numpy(),
                   numItermax = 10_000_000)
    return plan

def leading_axis(x, kernel):
    sums = fc.ev.cone_sums(x, kernel)
    flat = sums.flatten(1).argmax(dim = 1)
    return (flat // sums.shape[2], flat % sums.shape[2])

def score(signal_hat, truth, kernel):
    (row, col) = leading_axis(signal_hat, kernel)
    deta = (row - truth.row).float() * fc.ev.DETA
    dcol = (col - truth.col).abs()
    dphi = torch.minimum(dcol, 64 - dcol).float() * fc.ev.DPHI
    hit  = ((deta**2 + dphi**2).sqrt() < fc.ev.R_JET)[truth.jets]

    e_fake = truth.at_axis(signal_hat)
    return {
        'hit' : float(hit.float().mean()),
        **fc.ev.jet_scores(e_fake, truth),
    }

def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    rng     = np.random.default_rng(cmdargs.seed)
    kernel  = fc.ev.cone_kernel(fc.ev.R_JET)

    norm = fc.Norm.load_or_fit(
        os.path.join(fc.out_root(), 'norm_n20000_seed0.json')
    )
    (embed, signal) = fc.ev.load_pairs(
        os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0
    )
    bkg_pool = fc.read_random('background', cmdargs.pool, rng)
    sig_pool = fc.read_random('signal', cmdargs.pool, rng)
    sb       = fc.GPUSinkhornPlanSampler(reg = 2 * cmdargs.sigma**2)
    rows     = []

    for n in [ int(x) for x in cmdargs.batches.split(',') ]:
        for rep in range(cmdargs.repeats):
            pick  = rng.choice(len(embed), n, replace = False)
            truth = fc.ev.Truth(
                embed[pick], signal[pick], kernel, 10.0, device
            )
            m = torch.from_numpy(embed[pick]).to(device).float()
            b = torch.from_numpy(
                bkg_pool[rng.choice(cmdargs.pool, n, replace = False)]
            ).to(device)
            s = torch.from_numpy(
                sig_pool[rng.choice(cmdargs.pool, n, replace = False)]
            ).to(device)

            x0 = norm.state(m, torch.zeros_like(m))
            x1 = norm.state(b, s)

            plans = {}
            plans['random'] = (np.arange(n), 0.0, {})

            t0   = time.perf_counter()
            plan = exact_plan(x0, x1)
            plans['ot'] = (plan.argmax(axis = 1), time.perf_counter() - t0, {})

            torch.cuda.synchronize()
            t0   = time.perf_counter()
            plan = sb.get_map(x0, x1)
            dt   = time.perf_counter() - t0
            rowp = plan / plan.sum(axis = 1, keepdims = True)
            ent  = -(rowp * np.log(np.clip(rowp, 1e-300, None))).sum(axis = 1)
            j    = np.array([
                rng.choice(n, p = rowp[i]) for i in range(n)
            ])
            plans['sb'] = (j, dt, {
                'partners'  : float(np.exp(ent).mean()),
                'sb_err'    : sb.last_err,
                'plan_mass' : float(plan.sum()),
            })

            t0   = time.perf_counter()
            plan = exact_plan(
                norm.z(m, 'syn').unsqueeze(1), norm.z(b + s, 'syn').unsqueeze(1)
            )
            plans['ot-sum'] = (
                plan.argmax(axis = 1), time.perf_counter() - t0, {}
            )

            for (name, (j, dt, extra)) in plans.items():
                j = torch.from_numpy(np.asarray(j)).to(device)

                for decode in [ 'direct', 'mixture' ]:
                    s_hat = s[j] if decode == 'direct' else m - b[j]
                    rows.append({
                        'batch' : n, 'repeat' : rep, 'plan' : name,
                        'decode' : decode, 'plan_time' : dt,
                        **score(s_hat, truth, kernel), **extra,
                    })

            print(f'batch {n} repeat {rep} done', flush = True)

    df   = pd.DataFrame(rows)
    path = os.path.join(fc.out_root(), 'coupling_diag.csv')
    df.to_csv(path, index = False)

    cols = [ 'hit', 'jer_cal', 'jes', 'bias', 'plan_time' ]
    extra = [ c for c in [ 'partners', 'sb_err' ] if c in df ]
    print(df.groupby([ 'batch', 'plan', 'decode' ])[cols + extra]
            .mean().round(4).to_string())

    ref = fc.ev.reference_scores(
        fc.ev.Truth(embed, signal, kernel, 10.0, device), device
    )
    print(f"\nmedian-rho on the 20k val events: jer_cal {ref['jer_cal']:.2f}")
    print(f'wrote {path}')

if __name__ == '__main__':
    main()
