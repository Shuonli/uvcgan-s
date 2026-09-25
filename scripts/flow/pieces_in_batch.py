#!/usr/bin/env python
"""If a batch holds the true pieces of its mixtures, shuffled, does the
least-change matching find them?

    synthetic   n HIJING + n PYTHIA training events, mixtures made by adding
                them (as the idt-aa term of UVCGAN-S does)
    real        n val mixtures with their true pieces (the PYTHIA event from
                the index files, background = mixture - PYTHIA): uses the
                hidden truth, a check only

The pieces are shuffled, as bundles (HIJING, PYTHIA) or each on its own,
and matched to the mixtures with the cost of the trained OT-CFM (mixture
in the background panel, empty signal panel, log units). Score: how often
a mixture is matched to its own background.

    pieces_in_batch.py [--batches 256,1024] [--repeats 4]
"""

import argparse
import os

import numpy as np
import torch

import fm_common as fc
from coupling_diag import exact_plan

ev = fc.ev

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'True pieces in the batch')
    parser.add_argument('--batches', default = '256,1024')
    parser.add_argument('--repeats', type = int, default = 4)
    parser.add_argument('--seed', type = int, default = 0)
    return parser.parse_args()

def main():
    cmdargs = parse_cmdargs()
    rng  = np.random.default_rng(cmdargs.seed)
    norm = fc.Norm.load_or_fit(
        os.path.join(fc.out_root(), 'norm_n20000_seed0.json')
    )
    (embed, signal) = ev.load_pairs(
        os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0
    )
    bkg_pool = fc.read_random('background', 8192, rng)
    sig_pool = fc.read_random('signal', 8192, rng)

    for n in [ int(x) for x in cmdargs.batches.split(',') ]:
        for kind in [ 'synthetic', 'real' ]:
            found = []
            for _ in range(cmdargs.repeats):
                if kind == 'synthetic':
                    b = torch.from_numpy(bkg_pool[rng.choice(8192, n, replace = False)])
                    s = torch.from_numpy(sig_pool[rng.choice(8192, n, replace = False)])
                    m = b + s
                else:
                    pick = rng.choice(len(embed), n, replace = False)
                    m = torch.from_numpy(embed[pick]).float()
                    s = torch.from_numpy(signal[pick]).float()
                    b = (m - s).clamp(min = 0)

                order = rng.permutation(n)        # hide which piece is whose
                x0 = norm.state(m, torch.zeros_like(m))
                x1 = norm.state(b[order], s[order])
                j  = exact_plan(x0, x1).argmax(axis = 1)
                found.append(float(np.mean(order[j] == np.arange(n))))

            print(f'batch {n:5d}, {kind:9s} mixtures: matched to their own'
                  f' pieces {np.mean(found):.1%} (min {min(found):.1%})',
                  flush = True)

if __name__ == '__main__':
    main()
