#!/usr/bin/env python
"""Figure and table of the alpha-DSBM Gaussian checks (dsbm_gauss.py outputs):
the forward and backward cross-covariance and the generated variances along
pretraining and refinement, against the analytic Schrodinger-bridge values.

    dsbm_gauss_plot.py LABEL=PREFIX ... --out docs/flow/dsbm/gauss_check
"""

import argparse
import json

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({ 'font.size' : 8, 'legend.fontsize' : 6.5 })

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Gaussian check figure')
    parser.add_argument('runs', nargs = '+', help = 'LABEL=PREFIX of dsbm_gauss.py --out')
    parser.add_argument('--out', default = 'docs/flow/dsbm/gauss_check')
    return parser.parse_args()

def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    runs = [ r.split('=', 1) for r in cmdargs.runs ]
    (fig, axes) = plt.subplots(1, 3, figsize = (9.5, 3.4))
    rows = []
    for (k, (label, prefix)) in enumerate(runs):
        df  = pd.read_csv(f'{prefix}.csv')
        res = json.load(open(f'{prefix}.json', encoding = 'utf-8'))
        cfg = res['config']
        n_pre = df[df.stage == 'pretrain']['update'].max()
        x = df['update'] + (df.stage == 'refine') * n_pre
        color = f'C{k}'
        c_star = res['c_star']
        axes[0].plot(x, df.c_fwd / c_star, color = color, label = label)
        axes[0].plot(x, df.c_bwd / c_star, color = color, ls = '--')
        axes[1].plot(x, df.var_x1_hat / cfg['s1']**2, color = color)
        axes[2].plot(x, df.var_x0_hat / cfg['s0']**2, color = color)
        for ax in axes:
            ax.axvline(n_pre, color = color, lw = 0.4, ls = ':')
        fin = res['refined']
        pre = res['pretrained']
        rows.append({ 'check' : label, 's0' : cfg['s0'], 's1' : cfg['s1'], 'eps' : cfg['eps'],
                      'model' : cfg.get('model', 'mlp'),
                      'c_star' : c_star,
                      'pretrained_c_fwd' : pre['c_fwd'], 'pretrained_c_bwd' : pre['c_bwd'],
                      'refined_c_fwd' : fin['c_fwd'], 'refined_c_bwd' : fin['c_bwd'],
                      'refined_var_x1_rel' : fin['var_x1_hat'] / cfg['s1']**2,
                      'refined_var_x0_rel' : fin['var_x0_hat'] / cfg['s0']**2,
                      'pretrained_passes' : res['pretrained_passes'],
                      'refined_passes' : res['refined_passes'],
                      'exact_fixed_point_c_fwd' :
                          res['exact_network_free'][str(cfg['steps'])]['fixed_point']['c_fwd'],
                      'exact_pretrained_c_fwd' :
                          res['exact_network_free'][str(cfg['steps'])]['pretrained']['c_fwd'] })
    for (ax, title) in zip(axes, [ 'cross-covariance / c* (solid fwd, dashed bwd)',
                                   'Var X1_hat / s1^2 (forward)',
                                   'Var X0_hat / s0^2 (backward)' ]):
        ax.axhline(1, color = 'k', lw = 0.8)
        ax.axhspan(0.95, 1.05, color = 'grey', alpha = 0.15, lw = 0)
        ax.set_title(title, fontsize = 7.5)
        ax.set_xlabel('update (pretraining, then refinement)')
        ax.set_ylim(0.3, 2.3)
        ax.grid(alpha = 0.3)
    (handles, labels) = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'lower center', ncol = 4, fontsize = 6.5, frameon = False)
    fig.suptitle('Gaussian check of the alpha-DSBM code (d = 16, eps = 1, 30 SDE steps): analytic SB'
                 ' value = 1, grey band = 5%\nnetwork-free value after pretraining: coupling 0.88,'
                 ' variances 0.94 / 0.93 (dotted line: pretraining ends, refinement starts)', fontsize = 8)
    fig.tight_layout(rect = (0, 0.1, 1, 1))
    fig.savefig(f'{cmdargs.out}.png', dpi = 150)
    table = pd.DataFrame(rows)
    table.to_csv(f'{cmdargs.out}.csv', index = False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(table.round(4).to_string(index = False))

if __name__ == '__main__':
    main()
