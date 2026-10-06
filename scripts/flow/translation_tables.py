#!/usr/bin/env python
"""LaTeX tables of the PYTHIA -> JEWEL translation appendix
(docs/flow/translation/slides/tables/*.tex) from translation_report.py's
CSV files. Every number on the slides comes from those files.

    translation_tables.py [--dir docs/flow/translation]
    translation_tables.py --deck cgan     # a deck's tables (slides/tables/cgan_*.tex)
"""

import argparse
import os

import numpy as np
import pandas as pd

import translation_eval as te

TEX = { 'JEWEL test' : 'JEWEL test', 'JEWEL ref' : 'JEWEL ref',
        'random JEWEL' : 'random JEWEL', 'identity' : 'identity',
        'OT-CFM' : r'\ot{}', 'OT-CFM (4 Euler)' : r'\ot{}, 4 Euler$^\ast$',
        'alpha-DSBM' : r'\dsbm{}' }
OBS_TEX = { 'E' : '$E$', 'mass' : 'mass', 'girth' : 'girth', 'ptd' : '$p_T^D$',
            'zlead' : '$z_\\mathrm{lead}$', 'zg' : '$z_g$', 'rg' : '$R_g$',
            'core' : 'core', 'lead' : 'lead', 'pt' : '$p_T$' }

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Translation appendix tables')
    parser.add_argument('--dir', default = 'docs/flow/translation')
    parser.add_argument('--deck', default = None,
        help = 'report tag (docs/flow/translation/TAG): write that deck\'s tables'
               ' (slides/tables/TAG_*.tex) instead')
    return parser.parse_args()

DECK = { 'JEWEL ref' : 'JEWEL vs JEWEL', 'identity' : 'PYTHIA input',
         'random JEWEL' : r'\keyrj random JEWEL jet' }
DECK_OBS = [ 'E', 'mass', 'girth', 'zlead', 'ptd', 'zg', 'rg' ]
DECK_TEX = { **OBS_TEX, 'E' : '$p_T$' }

def deck_name(s):
    if s in DECK:
        return DECK[s]
    key = r'\keyot ' if s == 'OT-FM' else (r'\keycg ' if 'CycleGAN' in s else '')
    return key + s

def write(path, lines):
    with open(path, 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {path}')

def begin(cols, sep = 3):
    return [ r'\scriptsize', r'\setlength{\tabcolsep}{' + str(sep) + 'pt}',
             r'\begin{tabular}{' + cols + '}', r'\toprule' ]

END = [ r'\bottomrule', r'\end{tabular}' ]

SHAPE_CHANGES = [ ('girth', 'girth'), ('zlead', r'$z_\mathrm{lead}$'), ('ptd', '$p_T^D$'),
                  ('mass', 'mass (GeV)'), ('zg', '$z_g$'), ('rg', '$R_g$') ]

def col_head(s):
    """Two-line column header of a deck table (key line, then the name)."""
    names = { 'JEWEL ref' : ('JEWEL vs', 'JEWEL'), 'identity' : ('PYTHIA', 'input'),
              'random JEWEL' : (r'\keyrj random', 'JEWEL jet') }
    if s in names:
        return names[s]
    key = r'\keyot ' if s == 'OT-FM' else (r'\keycg ' if 'CycleGAN' in s else '')
    parts = s.split(' ', 1)
    return (key + parts[0], parts[1] if len(parts) > 1 else '')

def head_rows(samples):
    tops = [ col_head(s) for s in samples ]
    return [ ' & ' + ' & '.join(t for (t, _) in tops) + r' \\',
             ' & ' + ' & '.join(b for (_, b) in tops) + r' \\', r'\midrule' ]

def shape_change_table(tag, out):
    """Mean and rms of the per-jet change (output minus its own input) of
    each shape observable, all test jets, from the report's observables
    (translation/report/TAG.npz): the models against a random JEWEL jet.
    Observables are rows."""
    import numpy as np                         # pylint: disable=import-outside-toplevel
    import fm_common as fc                     # pylint: disable=import-outside-toplevel
    with np.load(os.path.join(fc.translation_root(), 'report', f'{tag}.npz')) as f:
        obs = { k : f[k] for k in f.files if k.startswith('crop|') }
    names = sorted({ k.split('|')[1] for k in obs })
    models = [ n for n in names if n not in ('JEWEL test', 'JEWEL ref', 'identity',
                                             'random JEWEL') ] + [ 'random JEWEL' ]
    rows = []
    stats = {}
    for m in models:
        for (q, _) in SHAPE_CHANGES:
            dlt = obs[f'crop|{m}|{q}'] - obs[f'crop|identity|{q}']
            dlt = dlt[np.isfinite(dlt)]
            stats[(m, q)] = (float(np.sqrt(np.mean(dlt**2))), float(np.mean(dlt)))
            rows += [ { 'model' : m, 'observable' : q, 'stat' : 'rms',
                        'value' : stats[(m, q)][0] },
                      { 'model' : m, 'observable' : q, 'stat' : 'mean',
                        'value' : stats[(m, q)][1] } ]
    # the rms of the change of each sample, and how much larger the random
    # JEWEL jet's is than each model's (the means: shape_changes.csv)
    n = len(models)
    lines = begin('l' + 'r' * (2 * n - 1), sep = 5)
    lines.append(r' & \multicolumn{%d}{c}{rms of the change} & \multicolumn{%d}{c}{ratio} \\'
                 % (n, n - 1))
    tops = [ col_head(m) for m in models ] + [ ('random /', col_head(m)[0].split()[-1])
                                               for m in models[:-1] ]
    lines.append(' & ' + ' & '.join(t for (t, _) in tops) + r' \\')
    lines.append(' & ' + ' & '.join(b for (_, b) in tops) + r' \\')
    lines.append(r'\midrule')
    for (q, label) in SHAPE_CHANGES:
        cells = [ f'{stats[(m, q)][0]:.3f}' for m in models ] \
            + [ f"{stats[('random JEWEL', q)][0] / stats[(m, q)][0]:.1f}" for m in models[:-1] ]
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    lines += END
    write(os.path.join(out, f'{tag}_shape_changes.tex'), lines)
    pd.DataFrame(rows).to_csv(os.path.join('docs/flow/translation', tag, 'shape_changes.csv'),
                              index = False)

def deck_tables(d, out, tag):
    """A deck's population, dependence and cost tables (TAG_*.tex), with
    observables as rows and samples as columns."""
    pop = pd.read_csv(os.path.join(d, 'population.csv'))
    models = [ s for s in pop['sample'] if s not in DECK ]
    samples = [ 'JEWEL ref', 'identity', 'random JEWEL' ] + models
    # narrower columns once a third model is in
    lines = begin('l' + 'r' * len(samples), sep = 4 if len(samples) <= 5 else 2.5) \
        + head_rows(samples)
    for q in DECK_OBS:
        cells = []
        for s in samples:
            r = pop[pop['sample'] == s].iloc[0]
            v = r[f'w1_{q}']
            text = f'{v:.3f}' if v < 0.1 else f'{v:.2f}'
            if s in models and bool(r[f'consistent_{q}']):
                text = r'\textbf{' + text + '}'
            cells.append(text)
        lines.append(f'{DECK_TEX[q]} & ' + ' & '.join(cells) + r' \\')
    lines += END
    write(os.path.join(out, f'{tag}_population.tex'), lines)

    dep = pd.read_csv(os.path.join(d, 'dependence.csv'))
    samples = [ 'random JEWEL' ] + models
    lines = begin('l' + 'r' * len(samples), sep = 5 if len(samples) <= 3 else 3) \
        + head_rows(samples)
    labels = { 'E' : '$p_T$', 'girth' : 'girth', 'mass' : 'mass', 'core' : 'core fraction',
               'lead' : 'leading tower' }
    for q in te.DEP:
        cells = [ f"{dep[dep['sample'] == s].iloc[0][f'corr_{q}']:.2f}".replace('-0.00', '0.00')
                  for s in samples ]
        lines.append(f'{labels[q]} & ' + ' & '.join(cells) + r' \\')
    lines.append(r'\midrule')
    cells = [ f"{100 * dep[dep['sample'] == s].iloc[0]['preference_rate']:.0f}\\%"
              for s in samples ]
    lines.append('own input preferred & ' + ' & '.join(cells) + r' \\')
    lines += END
    write(os.path.join(out, f'{tag}_dependence.tex'), lines)

    import json                                # pylint: disable=import-outside-toplevel
    with open(os.path.join(d, 'outputs.json'), encoding = 'utf-8') as f:
        info = json.load(f)
    lines = begin('lrrrrr', sep = 4)
    lines.append(r' & GPU h & updates & updates/s & network calls & ms / jet \\')
    lines.append(r'\midrule')
    for s in models:
        m = info[s]
        lines.append(f"{deck_name(s)} & {m['train_time'] / 3600:.1f} & {m['updates'] / 1e3:.0f}k & "
                     f"{m['updates'] / m['train_time']:.1f} & {m['nfe']} & {m['ms_per_jet']:.2f}"
                     + r' \\')
    lines += END
    write(os.path.join(out, f'{tag}_cost.tex'), lines)

def population(d, out):
    pop = pd.read_csv(os.path.join(d, 'population.csv'))
    order = [ 'JEWEL ref', 'random JEWEL', 'identity' ] + [
        s for s in pop['sample'] if s not in ('JEWEL ref', 'random JEWEL', 'identity') ]
    lines = begin('l' + 'r' * (len(te.OBS) + 1), sep = 2)
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
    lines = begin('lrrrrrr')
    lines.append(r' & \multicolumn{5}{c}{correlation, input vs output} & own input \\')
    lines.append(' & ' + ' & '.join(OBS_TEX[q] for q in te.DEP) + r' & preferred \\')
    lines.append(r'\midrule')
    for s in order:
        r = df[df['sample'] == s].iloc[0]
        cells = [ f"{r[f'corr_{q}']:.2f}" for q in te.DEP ]
        pref = f"{r['preference_rate']:.3f}" if np.isfinite(r.get('preference_rate', np.nan)) else '--'
        lines.append(f'{TEX.get(s, s)} & ' + ' & '.join(cells) + f' & {pref}' + r' \\')
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
    lines = begin('lrrrrrr', sep = 2.5)
    lines.append(r' & GPU h & updates & upd./s & overhead & NFE & ms/jet \\')
    lines.append(r'\midrule')
    for (_, r) in df.iterrows():
        if 'DSBM' in r['model']:
            lines.append(f"{TEX.get(r['model'], r['model'])} pretrain & "
                         f"{r['pretrain_min'] / 60:.2f} & {r['pretrain_updates'] / 1e3:.1f}k & "
                         f"{r['pretrain_updates_per_s']:.1f} & --"
                         r' & & \\')
            lines.append(f"\\quad + refine & {r['refine_min'] / 60:.2f} & "
                         f"{r['refine_updates'] / 1e3:.1f}k & {r['refine_updates_per_s']:.2f} & "
                         f"{100 * r['refine_rollout_share']:.0f}\\% roll-out & "
                         f"{int(r['nfe_sde30'])} & {r['latency_ms_sde30']:.2f}" + r' \\')
        else:
            lat = [ c for c in r.index if c.startswith('latency_ms_') and np.isfinite(r[c]) ]
            prim = [ c for c in lat if 'euler' not in c ][0]
            nfe = int(r[prim.replace('latency_ms_', 'nfe_')])
            lines.append(f"{TEX.get(r['model'], r['model'])} & {r['gpu_h']:.2f} & "
                         f"{r['updates'] / 1e3:.1f}k & {r['train_updates_per_s']:.1f} & "
                         f"{100 * r['coupling_share']:.0f}\\% OT & "
                         f"{nfe} & {r[prim]:.2f}" + r' \\')
            if 'latency_ms_euler4' in r and np.isfinite(r['latency_ms_euler4']):
                lines.append(r'\quad 4 Euler$^\ast$ & & & & & 4 & '
                             + f"{r['latency_ms_euler4']:.2f}" + r' \\')
    lines += END
    write(os.path.join(out, 'cost.tex'), lines)

def main():
    cmdargs = parse_cmdargs()
    out = os.path.join(cmdargs.dir, 'slides', 'tables')
    os.makedirs(out, exist_ok = True)
    if cmdargs.deck:
        deck_tables(os.path.join(cmdargs.dir, cmdargs.deck), out, cmdargs.deck)
        shape_change_table(cmdargs.deck, out)
        return
    population(cmdargs.dir, out)
    joint(cmdargs.dir, out)
    dependence(cmdargs.dir, out)
    fidelity(cmdargs.dir, out)
    cost(cmdargs.dir, out)

if __name__ == '__main__':
    main()
