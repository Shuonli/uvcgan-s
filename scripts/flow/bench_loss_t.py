#!/usr/bin/env python
"""Where along the path a paired flow's training signal lies (consolidated
benchmark, FLOW_NOTES.md): the CFM loss of a run's val-selected checkpoint
(EMA) at fixed t, per output channel, on held-out val pairs (the benchmark
images: M and the detector-level signal S, B = M - S).

For the joint arm (M, 0) -> (B, S) the signal channel of the state at any
t > 0 is (1 - t) z(0) + t z(S): it exposes S, and with it B, so the target
velocity is determined there and only t -> 0 carries the decision. For
the one-panel paired arm M -> B the state (1 - t) z(M) + t z(B) does not
separate M from B.

    bench_loss_t.py --runs bench_paired_bkg_s0 bench_joint_s0 [--n-events 2000]
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import fm_common as fc
from fm_eval import best_step, ckpt_step

T_GRID = [ 0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95 ]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'CFM loss against t')
    parser.add_argument('--runs', nargs = '+', required = True)
    parser.add_argument('--n-events', type = int, default = 2000)
    parser.add_argument('--batch', type = int, default = 250)
    parser.add_argument('--device', default = 'cuda')
    parser.add_argument('--out', default = 'docs/flow/bench/bench_loss_t.csv')
    return parser.parse_args()

@torch.no_grad()
def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    device  = torch.device(cmdargs.device)
    path    = os.path.join(fc.out_root(), 'bench', 'images', 'val')
    m_all   = np.load(os.path.join(path, 'mixture.npy'), mmap_mode = 'r')
    s_all   = np.load(os.path.join(path, 'truth.npy'), mmap_mode = 'r')
    rows = []
    for run in cmdargs.runs:
        run_dir = os.path.join(fc.out_root(), run)
        step = best_step(run_dir)
        ckpt = [ c for c in fc.list_checkpoints(run_dir) if ckpt_step(c) == step ][0]
        (method, net, _, _) = fc.load_run(run_dir, ckpt, device, 'ema')
        sums = { t : np.zeros(2) for t in T_GRID }
        n = 0
        for start in range(0, cmdargs.n_events, cmdargs.batch):
            m = torch.from_numpy(np.asarray(m_all[start:start + cmdargs.batch])).to(device)
            s = torch.from_numpy(np.asarray(s_all[start:start + cmdargs.batch])).to(device)
            b = (m - s).clamp(min = 0)
            x0 = method.source(m)
            x1 = method.norm.z(b, 'bkg').unsqueeze(1) if x0.shape[1] == 1 \
                else method.norm.state(b, s)
            for t in T_GRID:
                tt = torch.full((len(m),), t, device = device)
                xt = (1 - t) * x0 + t * x1
                sq = (net(tt, xt) - (x1 - x0))**2
                per = sq.mean(dim = (2, 3)).sum(dim = 0).cpu().numpy()
                sums[t][:len(per)] += per
            n += len(m)
        for t in T_GRID:
            rows.append({ 'run' : run, 'method' : method.name, 'step' : step, 't' : t,
                          'loss_bkg' : sums[t][0] / n,
                          'loss_sig' : sums[t][1] / n if x0.shape[1] == 2 else np.nan })
        print(pd.DataFrame(rows[-len(T_GRID):]).round(5).to_string(index = False),
              flush = True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cmdargs.out), exist_ok = True)
    df.to_csv(cmdargs.out, index = False)
    print(f'wrote {cmdargs.out}')

if __name__ == '__main__':
    main()
