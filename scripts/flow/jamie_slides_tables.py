#!/usr/bin/env python
"""LaTeX tables of the toy study's deck (docs/flow/jamie_otfm/slides/tables/)
from the report CSVs (jamie_report.py) and the ledger (jamie_tables.py).
Every number on the slides comes from those files.
"""

import os

import numpy as np
import pandas as pd

D = 'docs/flow/jamie_otfm'
T = os.path.join(D, 'slides', 'tables')

def csv(*parts):
    path = os.path.join(D, *parts)
    return pd.read_csv(path) if os.path.exists(path) else None

def write(name, lines):
    os.makedirs(T, exist_ok = True)
    with open(os.path.join(T, name), 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {name}')

def begin(cols, sep = 3):
    return [ r'\scriptsize', r'\setlength{\tabcolsep}{' + str(sep) + 'pt}',
             r'\begin{tabular}{' + cols + '}', r'\toprule' ]

END = [ r'\bottomrule', r'\end{tabular}' ]

def f(v, d = 2, pct = False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '--'
    if pct:
        return f'{100 * v:.1f}\\%' if abs(100 * v) < 9.95 else f'{100 * v:.0f}\\%'
    return f'{v:.{d}f}'

def sup(model, ex):
    """The supervision tag of a row (deck macros \\sUP, \\sSP, \\sHY, \\sTP;
    \\sST for the standard rho readouts)."""
    if 'paired control' in model:
        return r'\sTP'
    if model.startswith('rho') or model.startswith('standard'):
        return r'\sST'
    if model.startswith('truth') or model.startswith('identity'):
        return ''
    if ex in ('1', '2'):
        return r'\sUP' if model.startswith('E1-U') else r'\sSP'
    if ex == '22':
        return r'\sHY' if ' x ' in model else r'\sSP'
    return r'\sUP'

def tex(s):
    return str(s).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')

def ex1():
    s = csv('ex1', 'ex1_subtraction.csv')
    fk = csv('ex1', 'ex1_fakes.csv')
    if s is None:
        return
    fk = fk.set_index('model') if fk is not None else None
    lines = begin('llrrrrrrrr') + [
        r' & & offset & RMS & RMS cal. & halo $R<0.4$ & hard-tower & negative & fakes: largest & fakes: \\',
        r' & & GeV & GeV & GeV & GeV, $+$ & loss & towers & cone $>10$ & anti-$k_T$ \\', r'\midrule' ]
    for (_, r) in s.iterrows():
        m = r['model']
        fj = fk.loc[m] if (fk is not None and m in fk.index) else None
        if m.startswith('rho x A'):
            fj = fk.loc['rho x A (flat rho, event mean)'] if fk is not None else None
        lines.append(' & '.join([ tex(m), sup(m, '1'), f(r['offset'], 1), f(r['rms'], 2),
                                  f(r['rms_calibrated'], 2),
                                  f(r.get('halo_cone_positive'), 1),
                                  f(r.get('core_loss_hard_rel'), pct = True),
                                  f(r.get('neg_tower_share'), pct = True),
                                  f(fj['jamie_largest_cone_gt10'], pct = True) if fj is not None else '--',
                                  f(fj['akt_jet_gt10_share'], pct = True) if fj is not None else '--' ])
                     + r' \\')
    write('ex1.tex', lines + END)

def shifts(name, rows_filter = None):
    s = csv(name, f'{name}_shifts.csv')
    if s is None:
        return
    lines = begin('llrrrrrrr') + [
        r' & & cone $E$ & girth & ring 0.4-1 & mass & quenched & recoil & ring of \\',
        r' & & shift & shift & shift & shift & jets lost & in $\hat B$ & vacuum jets \\',
        r'\midrule' ]
    t = s.iloc[0]
    lines.append(' & '.join([ 'truth', '', f(t['cone_truth_shift'], 2) + ' GeV', '+' + f(t['girth_truth_shift'], 3),
                              '+' + f(t['ring_truth_shift'], 2) + ' GeV', f(t['mass_truth_shift'], 3) + ' GeV',
                              '0', '--', f(t['vac_ring_true'], 2) + ' GeV' ]) + r' \\')
    lines.append(r'\midrule')
    for (_, r) in s.iterrows():
        if rows_filter and not rows_filter(r['model']):
            continue
        lines.append(' & '.join([ tex(r['model']), sup(r['model'], '22' if name == 'prior_data' else '2'),
                                  f(r['cone_recovered'], pct = True),
                                  f(r['girth_recovered'], pct = True), f(r['ring_recovered'], pct = True),
                                  f(r['mass_shift'], 2) + ' GeV', f(r['lost_med'], pct = True),
                                  f(r.get('recoil_in_bhat'), 2), f(r['vac_ring_hat'], 2) ]) + r' \\')
    write(f'{name}.tex', lines + END)

def ex3(ex):
    p = csv(f'ex{ex}', f'ex{ex}_population.csv')
    dep = csv(f'ex{ex}', f'ex{ex}_dependence.csv')
    cond = csv(f'ex{ex}', f'ex{ex}_conditional.csv')
    if p is None:
        return
    ue = ex == '4'
    (c, r_) = ('cone_sub', 'ring_sub') if ue else ('cone', 'ring')
    dep = dep.set_index('model')
    lines = begin('llrrrrrrrr') + [
        r' & & \multicolumn{4}{c}{$W_1/\sigma$ to the true quenched} & \multicolumn{3}{c}{shift recovered}'
        r' & $r$(cone, \\', r' & & cone & ring & girth & $R>1$ & cone & ring & girth & input) \\', r'\midrule' ]
    for (_, r) in p.iterrows():
        m = r['model']
        d = dep.loc[m] if m in dep.index else None
        lines.append(' & '.join([ tex(m), sup(m, ex), f(r[f'w1_{c}'], 3), f(r[f'w1_{r_}'], 3), f(r['w1_girth'], 3),
                                  f(r['w1_far'], 3),
                                  f(d[f'{c}_shift_recovered'], pct = True) if d is not None else '',
                                  f(d[f'{r_}_shift_recovered'], pct = True) if d is not None else '',
                                  f(d['girth_shift_recovered'], pct = True) if d is not None else '',
                                  f(d[f'corr_{c}_input'], 2) if d is not None else '' ]) + r' \\')
    write(f'ex{ex}_population.tex', lines + END)
    if cond is None:
        return
    lines = begin('llrrrrrrrrr') + [
        r' & & \multicolumn{3}{c}{conditional $W_1/\sigma$} & \multicolumn{3}{c}{68\% coverage}'
        r' & \multicolumn{3}{c}{conditional sd: model / truth} \\',
        r' & & cone & ring & girth & cone & ring & girth & cone & ring & girth \\', r'\midrule' ]
    for (_, r) in cond.iterrows():
        lines.append(' & '.join([ tex(r['model']), sup(r['model'], ex) ]
                                + [ f(r[f'{q}_cond_w1'], 3) for q in ('cone', 'ring', 'girth') ]
                                + [ f(r[f'{q}_cov68'], pct = True) for q in ('cone', 'ring', 'girth') ]
                                + [ f(r[f'{q}_cond_sd_model'] / r[f'{q}_cond_sd_truth'], 2)
                                    for q in ('cone', 'ring', 'girth') ]) + r' \\')
    write(f'ex{ex}_conditional.tex', lines + END)

def ex4_ue():
    dep = csv('ex4', 'ex4_dependence.csv')
    if dep is None:
        return
    lines = begin('llrrrrr') + [
        r' & & far towers: $r$ & far rms change & core shift & ring shift & oracle ring \\',
        r' & & (output, input) & / UE tower sd & recovered & recovered & GeV \\', r'\midrule' ]
    t = dep.iloc[0]
    lines.append(' & '.join([ 'truth (J$_\\mathrm{med}$ + B$_a$)', '', f(t['truth_far_tower_corr_input'], 3),
                              f(t['truth_far_rms_change_over_ue_sd'], 3), '100\\%', '100\\%',
                              f(t.get('truth_oracle_ring'), 2) ]) + r' \\')
    lines.append(r'\midrule')
    for (_, r) in dep.iterrows():
        if r['model'].startswith('identity'):
            continue
        lines.append(' & '.join([ tex(r['model']), sup(r['model'], '4'), f(r['far_tower_corr_input'], 3),
                                  f(r['far_rms_change_over_ue_sd'], 3),
                                  f(r['cone_sub_shift_recovered'], pct = True),
                                  f(r['ring_sub_shift_recovered'], pct = True),
                                  f(r.get('oracle_ring'), 2) ]) + r' \\')
    write('ex4_ue.tex', lines + END)

def girth():
    g = csv('ex3g', 'ex3g_girth_rule.csv')
    if g is None:
        return
    lines = begin('llrrrrr') + [
        r' & & $r$(loss, input & $r$(loss, input & coefficient: & coefficient: & $r$(loss, \\',
        r' & & girth) & energy) & girth & energy & true loss) \\', r'\midrule' ]
    for (_, r) in g.iterrows():
        lines.append(' & '.join([ tex(r['model']), sup(r['model'], '3g'), f(r['corr_loss_frac_input_girth'], 2),
                                  f(r['corr_loss_frac_input_energy'], 2), f(r['coef_girth'], 3),
                                  f(r['coef_energy'], 3), f(r['corr_loss_with_true_loss'], 2) ]) + r' \\')
    write('ex3g.tex', lines + END)

def seeds():
    """Second seed: the same inputs' changes, seed 0 against seed 1."""
    lines = begin('lrrrrrrr') + [
        r' & \multicolumn{2}{c}{cone change} & \multicolumn{2}{c}{ring change}'
        r' & \multicolumn{2}{c}{$r$(change, true change)} & far towers \\',
        r' & $r$(seeds) & rms diff./rms & $r$(seeds) & rms diff./rms & cone & ring'
        r' & rms diff., GeV \\', r'\midrule' ]
    n = 0
    for ex in ('3', '4'):
        t = csv(f'ex{ex}', f'ex{ex}_seeds.csv')
        if t is None:
            continue
        (c, r_) = ('cone_sub', 'ring_sub') if ex == '4' else ('cone', 'ring')
        for (_, r) in t.iterrows():
            n += 1
            lines.append(' & '.join([ f'Ex.~{ex}: ' + tex(r['pair']), f(r[f'{c}_corr_changes'], 2),
                                      f(r[f'{c}_rms_seed_diff'] / r[f'{c}_rms_change_a'], 2),
                                      f(r[f'{r_}_corr_changes'], 2),
                                      f(r[f'{r_}_rms_seed_diff'] / r[f'{r_}_rms_change_a'], 2),
                                      f(r[f'{c}_corr_change_truth_a'], 2),
                                      f(r[f'{r_}_corr_change_truth_a'], 2),
                                      f(r['towers_far_rms_seed_diff'], 3) ]) + r' \\')
    if n:
        write('seeds.tex', lines + END)

BASES = [ 'E1-U_s0', 'E1-P_s0', 'E3-D_s0', 'E3-D_s1', 'E3-Ch_s0', 'E3-Cs_s0', 'E3-Cp_s0', 'E3g-D_s0',
          'E3g-C_s0', 'E4-D_s0', 'E4-D_s1', 'E4-Dnf_s0', 'E4-C_s0' ]

def solver():
    """The frozen solve of every baseline and the four-step Euler diagnostic."""
    import json                                       # pylint: disable=import-outside-toplevel
    eu = csv('solver', 'euler4.csv')
    eu = eu.set_index('label') if eu is not None else None
    lines = begin('lrlrrrrr') + [
        r' & frozen & & \multicolumn{3}{c}{$N$ against $2N$ (validation)} & \multicolumn{2}{c}{4 Euler steps} \\',
        r'run & NFE & resolved & key / change & $W_1$ shifts & key / two noises'
        r' & key / change & ms / image \\', r'\midrule' ]
    for b in BASES:
        path = os.path.join(D, 'solver', f'{b}_frozen.json')
        if not os.path.exists(path):
            continue
        with open(path, encoding = 'utf-8') as fh:
            fr = json.load(fh)
        # the check of N against 2N; E3-D's beyond 256 is in LABEL_fine.csv
        c = pd.concat([ x for x in (csv('solver', f'{b}.csv'), csv('solver', f'{b}_fine.csv'))
                        if x is not None ])
        c = c[c['nfe'] == fr['nfe']]
        if not len(c):
            continue
        r = c.iloc[-1]
        two = r.get('key_over_two_noises', np.nan)
        e = eu.loc[b] if (eu is not None and b in eu.index) else None
        lines.append(' & '.join([ tex(b), str(fr['nfe']), 'yes' if fr['resolved'] else 'no',
                                  f(r['key_over_change'], 3),
                                  'within sd' if bool(r['w1_ok']) else 'beyond sd',
                                  f(two, 3) if np.isfinite(two) else '--',
                                  f(e['key_over_change'], 2) if e is not None else '--',
                                  (f(e['ms_per_image_euler4'], 1) + ' / ' + f(e['ms_per_image_frozen'], 1))
                                  if e is not None else '--' ]) + r' \\')
    write('solver.tex', lines + END)

def transfer():
    """Solved-endpoint terms at the training rollout's NFE against the test
    solve (weights/*_transfer.json)."""
    import glob, json                                 # pylint: disable=import-outside-toplevel
    lines = begin('llrrrrr') + [
        r'continuation & term & rollout NFE & value & test NFE & value & ratio \\', r'\midrule' ]
    n = 0
    for path in sorted(glob.glob(os.path.join(D, 'weights', '*_transfer.json'))):
        with open(path, encoding = 'utf-8') as fh:
            t = json.load(fh)
        for (term, v) in t['terms'].items():
            n += 1
            lines.append(' & '.join([ tex(t['label']), tex(term), str(t['nfe']['train']), f(v['train'], 3),
                                      str(t['nfe']['test']), f(v['test'], 3),
                                      f(v['test'] / v['train'], 1) if v['train'] else '--' ]) + r' \\')
    if n:
        write('transfer.tex', lines + END)

def onestep():
    """Subtraction: the one-step mean against the solved output."""
    o = csv('ex1', 'ex1_onestep.csv')
    if o is None:
        return
    lines = begin('llrrr') + [
        r' & & hard-tower & cone offset & $\hat B$ on hard towers \\',
        r'run & estimate & loss & GeV & GeV (true UE 0.58) \\', r'\midrule' ]
    for (_, r) in o.iterrows():
        lines.append(' & '.join([ tex(r['label']), tex(r['estimate']), f(r['hard_loss_rel'], pct = True),
                                  f(r['cone_offset'], 1), f(r['bhat_on_hard_mean'], 2) ]) + r' \\')
    write('onestep.tex', lines + END)

def prior_data_effects():
    """The 2 x 2's pre-registered readings (prior_data_effects.csv)."""
    e = csv('prior_data', 'prior_data_effects.csv')
    if e is None:
        return
    lines = begin('lrrrrrrl') + [
        r' & \multicolumn{3}{c}{quenched minus vacuum data: shift (sd)} & \multicolumn{3}{c}{ring of vacuum jets'
        r' above truth, GeV} & data \\',
        r'prior & cone, GeV & girth & ring, GeV & vacuum data & quenched data & broad $-$ vacuum prior & matter? \\',
        r'\midrule' ]
    broad = { r['prior'] : r for (_, r) in e.iterrows() if str(r['prior']).startswith('broad minus') }
    for (_, r) in e.iterrows():
        if r['prior'] not in ('vac', 'broad'):
            continue
        bm = [ broad.get(f'broad minus vac, {d} data') for d in ('vac', 'quench') ]
        bmt = ', '.join(f(x['vac_ring_broad_minus_vac'], 2) for x in bm if x is not None) \
            if r['prior'] == 'broad' else '--'
        lines.append(' & '.join([ {'vac' : 'vacuum', 'broad' : 'broad'}[r['prior']],
                                  f"{f(r['cone_quench_minus_vac_data'], 2)} ({f(r['cone_z'], 1)})",
                                  f"{f(r['girth_quench_minus_vac_data'], 4)} ({f(r['girth_z'], 1)})",
                                  f"{f(r['ring_quench_minus_vac_data'], 2)} ({f(r['ring_z'], 1)})",
                                  f(r['vac_ring_excess_vac_data'], 2), f(r['vac_ring_excess_quench_data'], 2),
                                  bmt, 'yes' if bool(r['data_matter']) else 'no' ]) + r' \\')
    write('prior_data_effects.tex', lines + END)

def coverage():
    """Jamie's changes and their FM counterparts (coverage.csv)."""
    c = csv('coverage.csv')
    if c is None:
        return
    head = [ r'\tiny', r'\setlength{\tabcolsep}{2.5pt}',
             r'\begin{tabular}{p{0.2\linewidth}p{0.22\linewidth}p{0.22\linewidth}p{0.3\linewidth}}',
             r'\toprule', r'Jamie tried (printed slides) & his result & FM counterpart (runs) & our result \\',
             r'\midrule' ]
    rows = [ ' & '.join([ tex(r['jamie_change']) + f" ({tex(r['jamie_slides'])})", tex(r['jamie_result']),
                          tex(r['fm_counterpart']) + (f" ({tex(r['runs'])})" if r['runs'] != '-' else ''),
                          tex(r['our_result']) ]) + r' \\' for (_, r) in c.iterrows() ]
    half = (len(rows) + 1) // 2
    write('coverage_a.tex', head + rows[:half] + END)
    write('coverage_b.tex', head + rows[half:] + END)

def ledger():
    l = csv('ledger.csv')
    if l is None:
        return
    head = begin('llllrrrr') + [
        r'run & model & supervision & from & updates & GPU h & upd./s & NFE \\', r'\midrule' ]
    rows = []
    for (_, r) in l.iterrows():
        rows.append(' & '.join([ tex(r['run']), r['model'], tex(r['supervision']),
                                 tex(r['init']) if isinstance(r['init'], str) and r['init'] else 'scratch',
                                 f'{r["updates"] / 1e3:.1f}k' if np.isfinite(r['updates']) else '--',
                                 f(r['train_h'], 2), f(r['updates_per_s'], 1),
                                 f(r['solver_nfe'], 0) ]) + r' \\')
    write('ledger.tex', head + rows + END)
    half = (len(rows) + 1) // 2
    write('ledger_a.tex', head + rows[:half] + END)
    write('ledger_b.tex', head + rows[half:] + END)

def main():
    ex1()
    shifts('ex2')
    shifts('prior_data')
    ex3('3')
    ex3('4')
    ex4_ue()
    girth()
    seeds()
    solver()
    transfer()
    onestep()
    prior_data_effects()
    coverage()
    ledger()

if __name__ == '__main__':
    main()
