#!/usr/bin/env python
"""Compare the per rank NCCL flight recorder dumps of a hung run.

`TORCH_NCCL_DUMP_ON_TIMEOUT=1` makes every rank write the collectives it
issued when the watchdog fires. This script lines the ranks up and reports
the first sequence number at which they disagree, which is where the
deadlock starts.

    python scripts/slurm/analyze_nccl_trace.py slurm_logs/nccl_trace_12345
"""

import argparse
import glob
import pickle

from collections import defaultdict

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Analyze NCCL traces')
    parser.add_argument(
        'prefix', help = 'path prefix of the per rank dump files'
    )
    parser.add_argument(
        '--context', type = int, default = 3,
        help = 'entries to show around the divergence',
    )
    return parser.parse_args()

def load_traces(prefix):
    traces = {}

    for path in sorted(glob.glob(prefix + '*')):
        try:
            with open(path, 'rb') as f:
                dump = pickle.load(f)
        except Exception as e:                # pylint: disable=broad-except
            print(f"  cannot read '{path}': {e}")
            continue

        entries = dump.get('entries', dump) if isinstance(dump, dict) else dump
        rank    = path[len(prefix):].lstrip('_') or '?'
        traces[rank] = entries

    return traces

def describe(entry):
    if not isinstance(entry, dict):
        return str(entry)[:80]

    return '{} {} in={} state={}'.format(
        entry.get('seq_id', entry.get('collective_seq_id', '?')),
        entry.get('profiling_name', entry.get('op', '?')),
        entry.get('input_sizes', '?'),
        entry.get('state', '?'),
    )

def main():
    cmdargs = parse_cmdargs()
    traces  = load_traces(cmdargs.prefix)

    if not traces:
        print(f"No dumps found at '{cmdargs.prefix}*'")
        return

    print(f"loaded {len(traces)} rank dump(s)")
    for (rank, entries) in traces.items():
        print(f"  rank {rank}: {len(entries)} entries")

    # group each rank's last entries by state to find who is stuck where
    print("\nlast entry per rank:")
    for (rank, entries) in sorted(traces.items()):
        if entries:
            print(f"  rank {rank}: {describe(entries[-1])}")

    # first index at which the ranks' collective sequences differ
    length = min(len(e) for e in traces.values())
    ranks  = sorted(traces)

    for i in range(length):
        sigs = defaultdict(list)
        for rank in ranks:
            entry = traces[rank][i]
            key   = (
                entry.get('profiling_name'), str(entry.get('input_sizes'))
            ) if isinstance(entry, dict) else str(entry)
            sigs[key].append(rank)

        if len(sigs) > 1:
            print(f"\nranks diverge at entry {i}:")
            for (key, members) in sigs.items():
                print(f"  {key} <- ranks {members}")

            lo = max(0, i - cmdargs.context)
            print(f"\ncontext for rank {ranks[0]}:")
            for j in range(lo, min(length, i + cmdargs.context)):
                print(f"  [{j}] {describe(traces[ranks[0]][j])}")
            return

    print(f"\nno divergence in the first {length} entries;"
          " ranks differ only in how many entries they recorded:")
    for (rank, entries) in sorted(traces.items()):
        print(f"  rank {rank}: {len(entries)}")

if __name__ == '__main__':
    main()
