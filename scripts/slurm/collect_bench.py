#!/usr/bin/env python
"""Tabulate the results of `submit_bench.sh` from the `history.csv` files."""

import argparse
import glob
import json
import os
import re

import pandas as pd

from uvcgan_s import ROOT_OUTDIR

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Collect DDP benchmarks')
    parser.add_argument(
        'root', nargs = '?',
        default = os.path.join(ROOT_OUTDIR, 'sphenix', 'bench'),
        help    = 'directory with the benchmark models',
    )
    parser.add_argument(
        '--warmup', type = int, default = 1,
        help = 'number of warm-up epochs to discard',
    )
    parser.add_argument(
        '--prefix', default = 'bench_',
        help = 'only collect models whose label starts with this prefix',
    )
    parser.add_argument(
        '--filter', default = None,
        help = (
            'regex that labels must match, e.g. the job id suffix of one'
            ' series so that speedups are relative to the same node'
        ),
    )
    return parser.parse_args()

def collect_run(path, warmup, prefix, pattern):
    with open(os.path.join(path, 'label'), encoding = 'utf-8') as f:
        label = f.read().strip()

    if not label.startswith(prefix):
        return None

    if (pattern is not None) and (re.search(pattern, label) is None):
        return None

    history = pd.read_csv(os.path.join(path, 'history.csv'))
    history = history[history.epoch > warmup]

    if len(history) == 0:
        return None

    with open(os.path.join(path, 'config.json'), encoding = 'utf-8') as f:
        config = json.load(f)

    batch = config['batch_size']
    steps = config['steps_per_epoch']
    time  = history.epoch_time.mean()
    rate  = history.samples_per_sec.mean()
    world = int(round(rate * time / (steps * batch)))

    return {
        'label'       : label,
        'gpus'        : world,
        'batch/gpu'   : batch,
        'samples/step': batch * world,
        'step_ms'     : 1e3 * time / steps,
        'samples/s'   : rate,
        'epochs'      : len(history),
    }

def main():
    cmdargs = parse_cmdargs()

    runs = []
    for path in sorted(glob.glob(os.path.join(cmdargs.root, 'model_*'))):
        if not os.path.exists(os.path.join(path, 'history.csv')):
            continue

        run = collect_run(
            path, cmdargs.warmup, cmdargs.prefix, cmdargs.filter
        )
        if run is not None:
            runs.append(run)

    if not runs:
        print(f"No finished benchmark runs found under '{cmdargs.root}'")
        return

    table = pd.DataFrame(runs).sort_values([ 'batch/gpu', 'gpus', 'label' ])

    # speedup w.r.t. the single GPU run at the smallest batch (the training
    # configuration), scaling w.r.t. the single GPU run at the same batch
    single = table[table.gpus == 1].groupby('batch/gpu')['samples/s'].max()

    if len(single) > 0:
        base = single.loc[single.index.min()]
        table['speedup'] = table['samples/s'] / base
        table['scaling'] = (
            table['samples/s'] / table['batch/gpu'].map(single) / table.gpus
        )

    with pd.option_context('display.width', 200, 'display.precision', 2):
        print(table.to_string(index = False))

if __name__ == '__main__':
    main()
