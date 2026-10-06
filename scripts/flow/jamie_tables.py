#!/usr/bin/env python
"""The toy study's run and cost ledger (FLOW_NOTES.md, "Jamie's toy exercises
with OT flow matching"): every run's supervision, base, extra terms and
weights, updates, training time, update rate, coupling share, peak memory,
frozen solver and inference cost; docs/flow/jamie_otfm/ledger.csv.
"""

import json
import os

import pandas as pd

import jamie_methods as jm

ROLLOUT_TERMS = { 'energy', 'inv_global', 'inv_far', 'profile', 'div' }
SUPERVISION = { 'unpaired' : 'pure unpaired', 'synthetic' : 'synthetic paired',
                'paired' : 'true-paired control', 'hybrid' : 'hybrid' }

def load(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding = 'utf-8') as f:
        return json.load(f)

def main():
    runs = os.path.join(jm.jamie_root(), 'runs')
    rows = []
    for label in sorted(os.listdir(runs)):
        c = load(os.path.join(runs, label, 'config.json'))
        s = load(os.path.join(runs, label, 'summary.json'))
        if not c:
            continue
        w = load(os.path.join('docs', 'flow', 'jamie_otfm', 'weights', f'{label}.json'))
        f = load(os.path.join('docs', 'flow', 'jamie_otfm', 'solver', f'{label}_frozen.json'))
        m = load(os.path.join(jm.jamie_root(), 'outputs', label, 'meta.json'))
        ms = [ v['ms_per_image'] for v in m.get('sets', {}).values() if 'ms_per_image' in v ]
        rows.append({
            'run' : label, 'model' : 'D' if c['method'] == 'toyflow' else 'C',
            'supervision' : SUPERVISION.get(c.get('toy_pairing'), c.get('toy_pairing')),
            'transform' : c.get('toy_kind'), 'source' : c.get('toy_source') or c.get('toy_prior'),
            'target' : c.get('toy_target') or c.get('toy_ue'), 'data' : c.get('toy_data'),
            'coupling' : c.get('coupling'), 'ot_reg' : c.get('ot_reg'), 'cost' : c.get('toy_cost'),
            'seed' : c.get('seed'), 'init' : os.path.basename(c['init']) if c.get('init') else '',
            'extra' : c.get('toy_extra') or '', 'lambda' : c.get('toy_lambda') or '',
            'lambda_u' : c.get('toy_lambda_u') if c.get('toy_pairing') == 'hybrid' else '',
            # a solve only for the solved-endpoint terms (abs, bal, ring, ueprof:
            # the local surrogate)
            'roll_nfe' : c.get('roll_nfe') if set((c.get('toy_extra') or '').split(','))
                         & ROLLOUT_TERMS else '',
            'updates' : s.get('step'), 'train_h' : s.get('train_time', 0) / 3600 if s else None,
            'updates_per_s' : s.get('steps_per_s'), 'coupling_share' : s.get('coupling_frac'),
            'peak_gb' : s.get('peak_mem_gb'), 'end_to_end_h' : s.get('end_to_end_time', 0) / 3600 if s else None,
            'solver_nfe' : f.get('nfe'), 'solver_resolved' : f.get('resolved'),
            'ms_per_image' : sum(ms) / len(ms) if ms else None,
            'weights_audit' : json.dumps(w.get('lambda')) if w else '',
            'rollout_resolved' : w.get('rollout_resolved', '') if w else '' })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join('docs', 'flow', 'jamie_otfm', 'ledger.csv'), index = False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(df.round(3).to_string(index = False))
    print('GPU h of training:', round(float(df['train_h'].sum()), 2))

if __name__ == '__main__':
    main()
