#!/usr/bin/env python
"""Tables of the alpha-DSBM closure test (FLOW_NOTES.md, "alpha-DSBM
closure test"): the cost table (docs/flow/dsbm/closure_cost.csv) and the
LaTeX tables of the appendix slides (docs/flow/dsbm/slides/tables/*.tex),
from the outputs of dsbm_eval.py (PREFIX.csv, PREFIX_steps.csv,
PREFIX_multi.csv) and the runs' summary.json.

    dsbm_tables.py [--prefix docs/flow/dsbm/closure_test]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import fm_common as fc

# report label -> (TeX name, historical U-Net reference?)
TEX = {
    'identity F(J) = J' : (r'identity $F(J) = J$', False),
    'paired control' : (r'\pai{} control', True),
    'OT-CFM U-Net l2' : (r'\ot{}, U-Net, $L^2$', True),
    'OT-CFM U-Net shape+E' : (r'\ot{}, U-Net, shape + $E$', True),
    'OT-CFM U-Net shape+E pool 1024' : (r'\ot{}, U-Net, shape + $E$, pool 1024', True),
    'OT-CFM UVCGAN l2' : (r'\ot{}, UVCGAN, $L^2$', False),
    'bridge pretrained' : (r'\br{}, pretrained (0.5 h)', False),
    'bridge continued' : (r'\br{}, continued', False),
    'alpha-DSBM' : (r'\dsbm{}', False),
}

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'alpha-DSBM tables')
    parser.add_argument('--prefix', default = 'docs/flow/dsbm/closure_test')
    parser.add_argument('--tables', default = 'docs/flow/dsbm/slides/tables')
    return parser.parse_args()

def f(v, digits):
    return '--' if v is None or not np.isfinite(v) else f'{v:.{digits}f}'

def name(label):
    (tex, hist) = TEX.get(label, (label, False))
    return tex + (r'$^\dagger$' if hist else '')

def write(path, lines):
    with open(path, 'w', encoding = 'utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'wrote {path}')

def fidelity_table(df, path):
    lines = [ r'\scriptsize', r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
              r' & \multicolumn{4}{c}{shape EMD to $T(J)$} & EMD & $E_{out}/E_{in}$ & $E$ RMSE'
              r' & towers $>$ 5 GeV \\',
              r' & mean & median & p90 & p99 & GeV & (0.8) & GeV & mean / rms, GeV \\',
              r'\midrule' ]
    for (_, r) in df.iterrows():
        lines.append(
            f"{name(r.model)} & {f(r.shape_emd, 4)} & {f(r.shape_emd_median, 4)}"
            f" & {f(r.shape_emd_p90, 3)} & {f(r.shape_emd_p99, 3)} & {f(r.emd_gev, 2)}"
            f" & {f(r.E_out_over_E_in, 3)}$\\pm${f(r.E_out_over_E_in_sd, 3)}"
            f" & {f(r.E_rmse_gev, 2)} & {f(r['tower>5_mean'], 2)} / {f(r['tower>5_rms'], 2)} \\\\")
    lines += [ r'\bottomrule', r'\end{tabular}' ]
    write(path, lines)

def observables_table(df, path):
    lines = [ r'\scriptsize', r'\begin{tabular}{lrrrrrrrrr}', r'\toprule',
              r' & \multicolumn{4}{c}{per jet: change or value error} &'
              r' \multicolumn{5}{c}{distribution: $W_1/\sigma$ to $T(J)$} \\',
              r' & $\Delta m$ RMSE & $\Delta g$ RMSE & $p_T^D$ & $z_{lead}$'
              r' & $E$ & $m$ & $g$ & $p_T^D$ & $z_{lead}$ \\',
              r' & GeV & & rel. & rel. & & & & & \\', r'\midrule' ]
    for (_, r) in df.iterrows():
        lines.append(
            f"{name(r.model)} & {f(r.dmass_rmse, 3)} & {f(r.dgirth_rmse, 4)}"
            f" & {f(r.rel_rmse_ptd, 3)} & {f(r.rel_rmse_zlead, 3)}"
            f" & {f(r.w1_E, 3)} & {f(r.w1_mass, 3)} & {f(r.w1_girth, 3)}"
            f" & {f(r.w1_ptd, 3)} & {f(r.w1_zlead, 3)} \\\\")
    lines += [ r'\bottomrule', r'\end{tabular}' ]
    write(path, lines)

def spread_table(steps, multi, path):
    m = multi.set_index('model')
    s = steps.set_index('model')
    lines = [ r'\scriptsize', r'\begin{tabular}{lrrrrrrr}', r'\toprule',
              r' & \multicolumn{3}{c}{val, 1000 jets: shape EMD} &'
              r' \multicolumn{4}{c}{test, 1000 jets, 8 samples each} \\',
              r' & to $T(J)$ & 2 samples & 30 vs 60 steps & one sample & 2 samples'
              r' & mean image & spread $E/\sigma_E$ \\', r'\midrule' ]
    for label in s.index:
        a = s.loc[label]
        b = m.loc[label] if label in m.index else None
        lines.append(
            f"{name(label)} & {f(a.N_shape_emd_to_truth, 3)} & {f(a.two_samples_N_shape_emd, 3)}"
            f" & {f(a.N_vs_2N_coupled_shape_emd, 3)}"
            f" & {f(None if b is None else b.shape_emd_single_mean_over_samples, 3)}"
            f" & {f(None if b is None else b.shape_emd_between_samples, 3)}"
            f" & {f(None if b is None else b.avg_image_shape_emd, 3)}"
            f" & {f(None if b is None else b.spread_E_over_sd, 3)} \\\\")
    lines += [ r'\bottomrule', r'\end{tabular}' ]
    write(path, lines)

def summary_of(run):
    path = os.path.join(fc.out_root(), run)
    with open(os.path.join(path, 'summary.json'), encoding = 'utf-8') as fh:
        summary = json.load(fh)
    with open(os.path.join(path, 'config.json'), encoding = 'utf-8') as fh:
        config = json.load(fh)
    return (summary, config)

def cost(df):
    rows = []
    for (_, r) in df.iterrows():
        if not isinstance(r.get('run'), str):
            continue
        (s, c) = summary_of(r.run)
        bridge = 'stage' in c
        init_h = 0.0
        init_updates = 0
        if bridge and c.get('init'):
            (s0, _) = summary_of(c['init'])
            init_h = s0['train_time'] / 3600
            init_updates = s0['updates']
        updates = s['updates'] if bridge else s['step']
        rows.append({
            'model' : r.model, 'run' : r.run,
            'backbone' : c.get('backbone', 'unet'), 'n_params' : s['n_params'],
            'gpu_h_stage' : s['train_time'] / 3600,
            'gpu_h_total' : s['train_time'] / 3600 + init_h,
            'updates_stage' : updates, 'updates_total' : updates + init_updates,
            'updates_per_s' : s['updates_per_s'] if bridge else s['steps_per_s'],
            'rollout_share' : s.get('rollout_share', 0.0) if bridge else 0.0,
            'coupling_share' : 0.0 if bridge else s.get('coupling_frac', 0.0),
            'rollout_nfe_per_update' : s.get('rollout_nfe_per_update', 0) if bridge else 0,
            'peak_mem_gb' : s['peak_mem_gb'],
            'selected_step' : r.step, 'selected_gpu_h' : r.total_train_min / 60,
            'inference' : r.setting, 'nfe' : r.nfe, 'ms_per_jet' : r.ms_per_jet })
    return pd.DataFrame(rows)

def cost_table(tab, path):
    lines = [ r'\scriptsize', r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
              r' & GPU h & updates & updates/s & rollout share & peak GB & selected at'
              r' & inference NFE & ms / jet \\', r'\midrule' ]
    for (_, r) in tab.iterrows():
        lines.append(
            f"{name(r.model)} & {f(r.gpu_h_total, 2)} & {r.updates_total / 1000:.1f}k"
            f" & {f(r.updates_per_s, 2)} & {f(r.rollout_share, 2)} & {f(r.peak_mem_gb, 1)}"
            f" & {f(r.selected_gpu_h, 2)} h & {int(r.nfe)} & {f(r.ms_per_jet, 3)} \\\\")
    lines += [ r'\bottomrule', r'\end{tabular}' ]
    write(path, lines)

def main():
    cmdargs = parse_cmdargs()
    os.makedirs(cmdargs.tables, exist_ok = True)
    df = pd.read_csv(f'{cmdargs.prefix}.csv')
    fidelity_table(df, os.path.join(cmdargs.tables, 'fidelity.tex'))
    observables_table(df, os.path.join(cmdargs.tables, 'observables.tex'))
    steps = pd.read_csv(f'{cmdargs.prefix}_steps.csv')
    multi = pd.read_csv(f'{cmdargs.prefix}_multi.csv')
    spread_table(steps, multi, os.path.join(cmdargs.tables, 'spread.tex'))
    tab = cost(df)
    tab.to_csv(os.path.join(os.path.dirname(cmdargs.prefix), 'closure_cost.csv'), index = False)
    cost_table(tab, os.path.join(cmdargs.tables, 'cost.tex'))
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(tab.round(3).to_string(index = False))

if __name__ == '__main__':
    main()
