#!/usr/bin/env python
"""Score flow-matching checkpoints against held-out truth.

Uses the baseline's evaluator (scripts/slurm/eval_val_truth.py) unchanged:
the same 20k val (or JEWEL test) events, the same `score_generator`, the
same columns. The flow is wrapped as a generator (fm_common.Decomposer).

    fm_eval.py RUN_DIR [RUN_DIR ...] [--truth val|jewel] [--steps all|best|last]
               [--nets ema,raw] [--nfe 16] [--solver midpoint]
               [--decode direct] [--samples 1] [--latency]

Rows go to RUN_DIR/evals/{truth}_truth.csv, keyed by (step, net, nfe,
solver, decode, samples); rows already present are skipped. `--steps best`
takes the checkpoint of lowest val jer_cal of the EMA network at the
method's selection setting (`SELECTION`):
checkpoints are selected on val, JEWEL only scores the selected one.

--latency times the decomposition of 500 events (GPU-synchronised, after a
warm-up) for every requested setting and, for comparison, one forward pass
of the published UVCGAN-S generator; written to OUTDIR/sphenix/flow/latency.csv.
"""

import argparse
import fcntl
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch

import fm_common as fc

KEY = [ 'step', 'net', 'nfe', 'solver', 'decode', 'samples' ]

SELECTION = fc.SELECTION

PUBLISHED = os.path.join(
    'sphenix', 'pretrained',
    'model_m(uvcgan-s)_d(resnet)_g(vit-modnet)_sgn_bkg_sub'
)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Score flow checkpoints')
    parser.add_argument('runs', nargs = '*')
    parser.add_argument('--truth', default = 'val', choices = [ 'val', 'jewel' ])
    parser.add_argument('--steps', default = 'all',
        help = "'all', 'last', 'best' (on val) or comma separated steps")
    parser.add_argument('--nets', default = 'ema,raw')
    parser.add_argument('--nfe', default = '16')
    parser.add_argument('--solver', default = 'midpoint')
    parser.add_argument('--decode', default = 'direct')
    parser.add_argument('--samples', default = '1')
    parser.add_argument('--n-events', type = int, default = 20000)
    parser.add_argument('--batch', type = int, default = 500)
    parser.add_argument('--latency', action = 'store_true')
    parser.add_argument('--force', action = 'store_true',
        help = 'score settings already present again (e.g. for new columns)')
    return parser.parse_args()

def ckpt_step(path):
    return int(re.search(r'step_(\d+)\.pt$', path).group(1))

def best_step(run_dir):
    csv = os.path.join(run_dir, 'evals', 'val_truth.csv')
    if not os.path.exists(csv):
        raise RuntimeError(f"'{run_dir}': score the val events first")

    (nfe, solver, decode, samples) = default_setting(run_dir)

    v = pd.read_csv(csv)
    v = v[(v.net == 'ema') & (v.nfe == nfe) & (v.solver == solver)
          & (v.decode == decode) & (v.samples == samples)]

    return int(v.sort_values('jer_cal').step.iloc[0])

def default_setting(run_dir):
    """(nfe, solver, decode, samples) that checkpoints are selected with."""
    with open(os.path.join(run_dir, 'config.json'), 'r', encoding = 'utf-8') as f:
        method = json.load(f)['method']

    return SELECTION[method]

def select_checkpoints(run_dir, spec):
    ckpts = fc.list_checkpoints(run_dir)

    if spec == 'all':
        return ckpts
    if spec == 'last':
        return ckpts[-1:]
    if spec == 'best':
        step = best_step(run_dir)
        return [ c for c in ckpts if ckpt_step(c) == step ]

    steps = { int(x) for x in spec.split(',') }
    return [ c for c in ckpts if ckpt_step(c) in steps ]

def settings(cmdargs, method):
    result = []

    for nfe in [ int(x) for x in cmdargs.nfe.split(',') ]:
        for solver in cmdargs.solver.split(','):
            for decode in cmdargs.decode.split(','):
                for samples in [ int(x) for x in cmdargs.samples.split(',') ]:
                    if fc.is_regression(method):
                        # one evaluation, nothing to solve or to sample
                        (nfe, solver, samples) = (1, 'none', 1)
                    elif (solver == 'midpoint') and (nfe % 2):
                        continue
                    if (method != 'condcfm') and (samples != 1):
                        continue

                    s = (nfe, solver, decode, samples)
                    if s not in result:
                        result.append(s)

    return result

def evaluate_run(run_dir, cmdargs, truth, device):
    # pylint: disable=too-many-locals
    csv  = os.path.join(run_dir, 'evals', f'{cmdargs.truth}_truth.csv')
    done = set()

    if os.path.exists(csv) and not cmdargs.force:
        old  = pd.read_csv(csv)
        done = set(map(tuple, old[KEY].astype(str).values))

    rows = []

    for ckpt in select_checkpoints(run_dir, cmdargs.steps):
        step = ckpt_step(ckpt)

        for net_name in cmdargs.nets.split(','):
            (method, net, state, config) = fc.load_run(
                run_dir, ckpt, device, net_name
            )

            for (nfe, solver, decode, samples) in settings(cmdargs, method.name):
                key = tuple(map(str, (step, net_name, nfe, solver, decode,
                                      samples)))
                if key in done:
                    continue

                dec = fc.Decomposer(
                    method, net, nfe = nfe, solver = solver, decode = decode,
                    samples = samples
                )

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                (scores, e_fake) = fc.ev.score_generator(
                    dec, None, truth, cmdargs.batch, device
                )
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                row = {
                    'label'      : config['label'],
                    'method'     : method.name,
                    'step'       : step,
                    'train_time' : state['stats']['train_time'],
                    'examples'   : state['stats']['examples'],
                    'net'        : net_name,
                    'nfe'        : nfe,
                    'solver'     : solver,
                    'decode'     : decode,
                    'samples'    : samples,
                    **scores,
                    'eval_time'  : dt,
                }
                rows.append(row)

                outdir = os.path.join(
                    run_dir, 'evals', f'{cmdargs.truth}_truth'
                )
                os.makedirs(outdir, exist_ok = True)
                np.save(os.path.join(
                    outdir,
                    f's{step:08d}_{net_name}_{solver}{nfe}_{decode}'
                    f'_x{samples}.npy'
                ), e_fake)

                print(
                    f"{config['label']:>24s} {step:7d}"
                    f" {state['stats']['train_time'] / 60:6.1f} min"
                    f" {net_name} {solver:>8s} {nfe:3d} {decode:>7s}"
                    f" x{samples}  l1_sig {scores['l1_sig']:.4f}"
                    f"  l1_bkg {scores['l1_bkg']:.4f}"
                    f"  mse_sig {scores['mse_sig']:.4f}  jes {scores['jes']:.3f}"
                    f"  jer_cal {scores['jer_cal']:5.2f}  ({dt:.1f} s)",
                    flush = True
                )

    if rows:
        # several scoring jobs may finish on the same run at once
        with open(f'{csv}.lock', 'w', encoding = 'utf-8') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            new = pd.DataFrame(rows)
            if os.path.exists(csv):
                new = pd.concat([ pd.read_csv(csv), new ])
            new = new.drop_duplicates(KEY, keep = 'last')
            new.sort_values(KEY).to_csv(csv, index = False)

@torch.no_grad()
def time_call(fn, x, repeats = 5):
    fn(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return float(np.median(times))

def measure_latency(cmdargs, truth, device):
    # pylint: disable=import-outside-toplevel
    from uvcgan_s.config import Args
    from uvcgan_s.cgan   import construct_model

    x    = torch.from_numpy(truth.embed[:cmdargs.batch]).to(device)
    x    = x.float().unsqueeze(1)
    rows = []

    outdir = os.environ.get('UVCGAN_S_OUTDIR', 'outdir')
    args   = Args.load(os.path.join(outdir, PUBLISHED))
    model  = construct_model(args.savedir, args.config, is_train = False,
                             device = device)
    model.load(None)
    gen  = model.models.ema_gen_ba
    gen.eval()
    norm = model.data_norm

    def baseline(x):
        return norm.denormalize(gen(norm.normalize(x)))

    dt = time_call(baseline, x)
    rows.append({
        'model' : 'uvcgan-s published (ema_gen_ba)', 'nfe' : 1,
        'solver' : 'none', 'samples' : 1,
        'n_params' : fc.count_params(gen),
        'ms_per_batch' : 1000 * dt, 'ms_per_event' : 1000 * dt / len(x),
    })

    for run_dir in cmdargs.runs:
        ckpt = fc.list_checkpoints(run_dir)[-1]
        (method, net, _state, config) = fc.load_run(run_dir, ckpt, device)

        for (nfe, solver, decode, samples) in settings(cmdargs, method.name):
            if decode != 'direct':
                continue

            dec = fc.Decomposer(method, net, nfe = nfe, solver = solver,
                                samples = samples)
            dt  = time_call(dec, x)
            rows.append({
                'model' : f"{config['label']} ({method.name})", 'nfe' : nfe,
                'solver' : solver, 'samples' : samples,
                'n_params' : fc.count_params(net),
                'ms_per_batch' : 1000 * dt,
                'ms_per_event' : 1000 * dt / len(x),
            })

    df = pd.DataFrame(rows)
    df['batch'] = len(x)
    df['gpu']   = torch.cuda.get_device_name()
    print(df.to_string(index = False))

    path = os.path.join(fc.out_root(), 'latency.csv')
    df.to_csv(path, mode = 'a', header = not os.path.exists(path),
              index = False)

def main():
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    runs    = []

    for pattern in cmdargs.runs:
        runs += sorted(glob.glob(pattern.rstrip('/')))
    cmdargs.runs = runs

    (embed, signal) = fc.ev.load_pairs(
        os.environ.get('UVCGAN_S_DATA', 'data'), cmdargs.n_events, 0,
        truth = cmdargs.truth
    )
    truth = fc.ev.Truth(
        embed, signal, fc.ev.cone_kernel(fc.ev.R_JET), 10.0, device
    )

    if cmdargs.latency:
        measure_latency(cmdargs, truth, device)
        return

    for run_dir in runs:
        evaluate_run(run_dir, cmdargs, truth, device)

if __name__ == '__main__':
    main()
