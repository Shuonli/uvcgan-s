#!/usr/bin/env python
"""LaTeX tables of the PYTHIA -> JEWEL translation appendix
(docs/flow/translation/slides/tables/*.tex) from translation_report.py's
CSV files. Every number on the slides comes from those files.

    translation_tables.py [--dir docs/flow/translation]
"""

import argparse
import os

import numpy as np
import pandas as pd

import translation_eval as te

TEX = { 'JEWEL test' : 'JEWEL test (target)', 'JEWEL ref' : 'JEWEL ref (floor)',
        'random JEWEL' : 'random JEWEL', 'identity' : 'identity (PYTHIA)',
        'OT-CFM' : r'\ot{}', 'OT-CFM (4 Euler)' : r'\ot{}, 4 Euler$^\ast$',
        'alpha-DSBM' : r'\dsbm{}' }
OBS_TEX = { 'E' : '$E$', 'mass' : 'mass', 'girth' : 'girth', 'ptd' : '$p_T^D$',
            'zlead' : '$z_\\mathrm{lead}$', 'zg' : '$z_g$', 'rg' : '$R_g$',
            'core' : 'core', 'lead' : 'lead', 'pt' : '$p_T$' }

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Translation appendix tables')
    parser.add_argument('--dir', default = 'docs/flow/translation')
    return parser.parse_args()

def write(path, lines):
    with open(path, 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {path}')

def begin(cols):
    return [ r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{' + cols + '}', r'\toprule' ]

END = [ r'\bottomrule', r'\end{tabular}' ]

def population(d, out):
    pop = pd.read_csv(os.path.join(d, 'population.csv'))
    order = [ 'JEWEL ref', 'random JEWEL', 'identity' ] + [
        s for s in pop['sample'] if s not in ('JEWEL ref', 'random JEWEL', 'identity') ]
    lines = begin('l' + 'r' * (len(te.OBS) + 1))
    lines.append(' & pass & ' + ' & '.join(OBS_TEX[q] for q in te.OBS) + r' \\')
    lines.append(r'\midrule')
    for s in order:
        r = pop[pop['sample'] == s].iloc[0]
        cells = []
        for q in te.OBS:
            v = f"{r[f'w1_{q}']:.3f}"
            if s not in ('JEWEL ref', 'random JEWEL', 'identity') and bool(r[f'moves_{q}']):
                v = r'\textbf{' + v + '}'
            if s not in ('JEWEL ref',) and bool(r[f'consistent_{q}']):
                v += r'$^\circ$'
            cells.append(v)
        lines.append(f"{TEX.get(s, s)} & {r['acceptance']:.3f} & " + ' & '.join(cells) + r' \\')
        if s == 'identity':
            lines.append(r'\midrule')
    lines += END
    write(os.path.join(out, 'population.tex'), lines)

def joint(d, out):
    df = pd.read_csv(os.path.join(d, 'joint.csv'))
    order = [ 'JEWEL ref', 'random JEWEL', 'identity' ] + [
        s for s in df['sample'] if s not in ('JEWEL ref', 'random JEWEL', 'identity') ]
    lines = begin('lrrrrrrr')
    lines.append(r' & \multicolumn{3}{c}{girth $\mid E$ bin} & \multicolumn{3}{c}'
                 r'{$z_\mathrm{lead} \mid E$ bin} & corr. \\')
    lines.append(r' & 10-20 & 20-30 & 30-60 & 10-20 & 20-30 & 30-60 & max $|\Delta|$ \\')
    lines.append(r'\midrule')
    for s in order:
        r = df[df['sample'] == s].iloc[0]
        cells = []
        for q in [ 'girth', 'zlead' ]:
            for (lo, hi) in [ (10, 20), (20, 30), (30, 60) ]:
                cells.append(f"{r[f'w1_{q}_pt{lo}_{hi}']:.3f}")
        lines.append(f'{TEX.get(s, s)} & ' + ' & '.join(cells)
                     + f" & {r['corr_maxdiff']:.2f}" + r' \\')
        if s == 'identity':
            lines.append(r'\midrule')
    lines += END
    write(os.path.join(out, 'joint.tex'), lines)

def dependence(d, out):
    df = pd.read_csv(os.path.join(d, 'dependence.csv'))
    order = [ 'identity', 'random JEWEL' ] + [
        s for s in df['sample'] if s not in ('identity', 'random JEWEL') ]
    lines = begin('lrrrrrrr')
    lines.append(r' & \multicolumn{5}{c}{correlation, input vs output} & own input & '
                 r'$E_\mathrm{out}/E_\mathrm{in}$ \\')
    lines.append(' & ' + ' & '.join(OBS_TEX[q] for q in te.DEP)
                 + r' & preferred & mean $\pm$ sd \\')
    lines.append(r'\midrule')
    for s in order:
        r = df[df['sample'] == s].iloc[0]
        cells = [ f"{r[f'corr_{q}']:.2f}" for q in te.DEP ]
        pref = f"{r['preference_rate']:.3f}" if np.isfinite(r.get('preference_rate', np.nan)) else '--'
        lines.append(f'{TEX.get(s, s)} & ' + ' & '.join(cells) + f' & {pref} & '
                     f"{r['E_ratio_mean']:.2f} $\\pm$ {r['E_ratio_sd']:.2f}" + r' \\')
    lines += END
    write(os.path.join(out, 'dependence.tex'), lines)

def fidelity(d, out):
    df = pd.read_csv(os.path.join(d, 'fidelity.csv'))
    order = [ 'JEWEL test', 'identity' ] + [
        s for s in df['sample'] if s not in ('JEWEL test', 'identity', 'JEWEL ref',
                                             'random JEWEL') ]
    lines = begin('lrrrrrrrl')
    lines.append(r' & \multicolumn{4}{c}{cone towers per jet above} & $E$ in towers & '
                 r'lead tower & core & \\')
    lines.append(r' & 0 & 0.01 & 0.1 & 1 GeV & $<$ 0.5 GeV & GeV & fraction & flags \\')
    lines.append(r'\midrule')
    for s in order:
        r = df[df['sample'] == s].iloc[0]
        flags = ', '.join(f for (f, k) in [ ('soft floor', 'soft_floor'),
                                             ('flattening', 'flattening') ] if bool(r[k]))
        lines.append(f"{TEX.get(s, s)} & {r['n_pos_mean']:.1f} & {r['n_0p01_mean']:.1f} & "
                     f"{r['n_0p1_mean']:.1f} & {r['n1_mean']:.1f} & {r['soft_0p5_mean']:.2f} & "
                     f"{r['lead_mean']:.2f} & {r['core_mean']:.3f} & {flags or '--'}" + r' \\')
    lines += END
    write(os.path.join(out, 'fidelity.tex'), lines)

def cost(d, out):
    df = pd.read_csv(os.path.join(d, 'cost.csv'))
    lines = begin('lrrrrrrrr')
    lines.append(r' & GPU h & stage & updates & updates/s & in rollouts & peak GB & '
                 r'NFE & ms / jet \\')
    lines.append(r'\midrule')
    for (_, r) in df.iterrows():
        if 'DSBM' in r['model']:
            lines.append(f"{TEX.get(r['model'], r['model'])} & {r['pretrain_min'] / 60:.2f} & "
                         f"pretrain & {r['pretrain_updates'] / 1e3:.1f}k & "
                         f"{r['pretrain_updates_per_s']:.1f} & -- & {r['pretrain_peak_gb']:.1f}"
                         r' & & \\')
            lines.append(f" & {r['refine_min'] / 60:.2f} & refine & "
                         f"{r['refine_updates'] / 1e3:.1f}k & {r['refine_updates_per_s']:.2f} & "
                         f"{100 * r['refine_rollout_share']:.0f}\\% & {r['refine_peak_gb']:.1f} & "
                         f"{int(r['nfe_sde30'])} & {r['latency_ms_sde30']:.2f}" + r' \\')
        else:
            lat = [ c for c in r.index if c.startswith('latency_ms_') and np.isfinite(r[c]) ]
            prim = [ c for c in lat if 'euler' not in c ][0]
            nfe = int(r[prim.replace('latency_ms_', 'nfe_')])
            lines.append(f"{TEX.get(r['model'], r['model'])} & {r['gpu_h']:.2f} & train & "
                         f"{r['updates'] / 1e3:.1f}k & {r['train_updates_per_s']:.1f} & "
                         f"{100 * r['coupling_share']:.0f}\\% OT & {r['train_peak_gb']:.1f} & "
                         f"{nfe} & {r[prim]:.2f}" + r' \\')
            if 'latency_ms_euler4' in r and np.isfinite(r['latency_ms_euler4']):
                lines.append(r' & & & & & & & 4$^\ast$ & ' + f"{r['latency_ms_euler4']:.2f}"
                             + r' \\')
    lines += END
    write(os.path.join(out, 'cost.tex'), lines)

def main():
    cmdargs = parse_cmdargs()
    out = os.path.join(cmdargs.dir, 'slides', 'tables')
    os.makedirs(out, exist_ok = True)
    population(cmdargs.dir, out)
    joint(cmdargs.dir, out)
    dependence(cmdargs.dir, out)
    fidelity(cmdargs.dir, out)
    cost(cmdargs.dir, out)

if __name__ == '__main__':
    main()
