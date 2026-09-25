#!/usr/bin/env python
"""Read-outs of the unpaired OT-CFM (mixture -> background) that trade the
cleanliness of the extracted image against its jet energy, without
retraining.

The flow gives a background b(m) of each mixture m; the signal is read as
m - b. A coarse solve (4 Euler steps) gives the better jet energy, a
converged one (midpoint, 16 NFE) the cleaner image. Read-outs scored here
(none of them uses the truth; parameters are chosen on val, JEWEL confirms):

    coarse, converged      m - b of either solve
    *_clip                 max(m - b, 0): signal energies are not negative
    coarse_thr<t>          m - b where it exceeds t GeV, else 0
    blend                  m - (b_coarse + b_converged) / 2
    seeded<E>_<out>        a jet-seeded subtraction: towers whose coarse cone
                           energy (R = 0.4) exceeds E GeV seed a region (all
                           towers within R = 0.4 of a seed); inside it the
                           coarse subtraction, outside it either nothing
                           (`zero`) or the clipped converged subtraction
                           (`conv`)

Scores: the usual columns of eval_val_truth (l1_sig, jes, jer_cal, ...)
and, as a diagnostic that does use the truth, the per-tower error of the
signal image inside and outside the true leading-jet cone (l1_in, l1_out)
and how often the seeded region covers the true jet axis. References
scored the same way: the published UVCGAN-S generator and the L1
regression.

    readout_test.py OTCFM_RUN [OTCFM_RUN ...] [--truth val|jewel]
                    [--readouts all|name,name] [--n-events 20000]
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import fm_common as fc

ev = fc.ev

PUBLISHED = os.path.join(
    'sphenix', 'pretrained',
    'model_m(uvcgan-s)_d(resnet)_g(vit-modnet)_sgn_bkg_sub'
)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'OT-CFM read-outs')
    parser.add_argument('runs', nargs = '+')
    parser.add_argument('--truth', default = 'val', choices = [ 'val', 'jewel' ])
    parser.add_argument('--readouts', default = 'all')
    parser.add_argument('--n-events', type = int, default = 20000)
    parser.add_argument('--batch', type = int, default = 500)
    parser.add_argument('--references', action = 'store_true',
        help = 'also score the published UVCGAN-S and the L1 regression')
    parser.add_argument('--clean-references', action = 'store_true',
        help = 'with --references: apply the same read-out rules (tower'
               ' threshold, jet seeding) to their signal images, for a'
               ' like-for-like comparison')
    parser.add_argument('--out', default = None)
    parser.add_argument('--figure', default = None,
        help = 'event display of a few val events, with --references')
    parser.add_argument('--show', default = 'coarse,converged,seeded12_zero_thr0.5',
        help = 'read-outs of the first run in the event display')
    return parser.parse_args()

READOUTS = [
    'coarse', 'converged', 'coarse_clip', 'converged_clip',
    'coarse_thr0.2', 'coarse_thr0.3', 'coarse_thr0.5', 'coarse_thr0.7',
    'coarse_thr1', 'blend',
    'seeded8_zero', 'seeded12_zero', 'seeded15_zero', 'seeded20_zero',
    'seeded25_zero', 'seeded30_zero', 'seeded15_zero_clip',
    'seeded20_zero_clip', 'seeded20_conv',
    'seeded12_zero_thr0.3', 'seeded12_zero_thr0.5', 'seeded12_zero_thr0.7',
    'seeded10_zero_thr0.5', 'seeded15_zero_thr0.5',
]

def needs_converged(name):
    """Read-outs that use the flow's converged solve, which a model that
    outputs its signal directly does not have."""
    return name in ('converged', 'converged_clip', 'blend') \
        or name.endswith('_conv')

def clean_image(images, embed, name, batch, device, kernel):
    """A read-out rule applied to a model's own signal image: the rules
    act on the signal, which here is the model's output rather than
    mixture - flow background."""
    (out, regions) = ([], [])
    for start in range(0, len(images), batch):
        s_hat = images[start:start + batch].to(device)
        m = torch.from_numpy(embed[start:start + batch]).to(device).float()
        (s_clean, region) = readout(name, m, m - s_hat, m - s_hat, kernel)
        out.append(s_clean.cpu())
        if region is not None:
            regions.append(region.cpu())
    return (torch.cat(out), torch.cat(regions) if regions else None)

def cone_mask(truth, start, n, device):
    """Towers within R = 0.4 of each event's true jet axis."""
    rows = truth.row[start:start + n].to(device)[:, None, None]
    cols = truth.col[start:start + n].to(device)[:, None, None]

    r = torch.arange(fc.SHAPE[0], device = device)[None, :, None]
    c = torch.arange(fc.SHAPE[1], device = device)[None, None, :]

    deta = (r - rows).float() * ev.DETA
    dcol = (c - cols).abs()
    dphi = torch.minimum(dcol, fc.SHAPE[1] - dcol).float() * ev.DPHI

    return deta**2 + dphi**2 <= ev.R_JET**2 + 1e-9

def readout(name, m, b4, bc, kernel):
    # pylint: disable=too-many-return-statements
    s4 = m - b4
    sc = m - bc

    if name == 'coarse':
        return (s4, None)
    if name == 'converged':
        return (sc, None)
    if name == 'coarse_clip':
        return (s4.clamp(min = 0), None)
    if name == 'converged_clip':
        return (sc.clamp(min = 0), None)
    if name.startswith('coarse_thr'):
        t = float(name[len('coarse_thr'):])
        return (torch.where(s4 > t, s4, torch.zeros_like(s4)), None)
    if name == 'blend':
        return (m - 0.5 * (b4 + bc), None)
    if name.startswith('seeded'):
        # seeded<E>_<outside>[_clip][_thr<t>]
        parts  = name[len('seeded'):].split('_')
        (energy, outside) = parts[:2]
        seeds  = ev.cone_sums(s4, kernel) > float(energy)
        region = ev.cone_sums(seeds.float(), kernel) > 0.5
        rest   = torch.zeros_like(s4) if outside == 'zero' \
            else sc.clamp(min = 0)
        inside = s4
        for opt in parts[2:]:
            if opt == 'clip':
                inside = inside.clamp(min = 0)
            elif opt.startswith('thr'):
                t = float(opt[3:])
                inside = torch.where(inside > t, inside,
                                     torch.zeros_like(inside))
        return (torch.where(region, inside, rest), region)

    raise ValueError(name)

def score_images(images, truth, batch, device, kernel, regions = None):
    """eval_val_truth's columns plus the in/out-of-cone split."""
    # pylint: disable=too-many-locals
    n = len(images)
    (l1, l1_in, l1_out, n_in, n_out, covered) = (0.0, 0.0, 0.0, 0, 0, 0)
    (se, se_in, se_out) = (0.0, 0.0, 0.0)
    e_fake = []

    for start in range(0, n, batch):
        s_hat = images[start:start + batch].to(device)
        s     = torch.from_numpy(truth.signal[start:start + batch]).to(device)
        s     = s.float()
        err   = (s_hat - s).abs()
        mask  = cone_mask(truth, start, len(s), device)

        l1     += float(err.sum())
        l1_in  += float(err[mask].sum())
        l1_out += float(err[~mask].sum())
        se     += float((err**2).sum())
        se_in  += float((err[mask]**2).sum())
        se_out += float((err[~mask]**2).sum())
        n_in   += int(mask.sum())
        n_out  += int((~mask).sum())

        if regions is not None:
            reg  = regions[start:start + batch].to(device)
            rows = truth.row[start:start + len(s)]
            cols = truth.col[start:start + len(s)]
            covered += int(reg[torch.arange(len(s), device = device),
                               rows, cols][
                truth.jets[start:start + len(s)]
            ].sum())

        e_fake.append(truth.at_axis(s_hat, start))

    e_fake = torch.cat(e_fake)
    result = {
        'l1_sig' : l1 / (n * images[0].numel()),
        'l1_in'  : l1_in / n_in,
        'l1_out' : l1_out / n_out,
        'abs_err_out_gev' : l1_out / n,
        'mse_sig' : se / (n * images[0].numel()),
        'mse_in'  : se_in / n_in,
        'mse_out' : se_out / n_out,
        **ev.jet_scores(e_fake, truth),
    }
    if regions is not None:
        result['jet_in_region'] = covered / int(truth.jets.sum())

    return result

@torch.no_grad()
def flow_backgrounds(run_dir, embed, batch, device, truth_name):
    ckpt_step = fc_best_step(run_dir)
    ckpt = [ c for c in fc.list_checkpoints(run_dir)
             if c.endswith(f'step_{ckpt_step:08d}.pt') ][0]
    (method, net, state, config) = fc.load_run(run_dir, ckpt, device, 'ema')

    cache = os.path.join(
        run_dir, 'evals',
        f'readout_{truth_name}_s{ckpt_step:08d}_n{len(embed)}.pt'
    )
    if os.path.exists(cache):
        return (torch.load(cache), config['label'], ckpt_step,
                state['stats']['train_time'] / 60)

    result = {}
    for (name, nfe, solver) in [ ('b4', 4, 'euler'), ('bc', 16, 'midpoint') ]:
        dec = fc.Decomposer(method, net, nfe = nfe, solver = solver)
        out = []
        for start in range(0, len(embed), batch):
            m = torch.from_numpy(embed[start:start + batch]).to(device)
            out.append(dec(m.float().unsqueeze(1))[:, 0].cpu())
        result[name] = torch.cat(out)

    torch.save(result, cache)
    return (result, config['label'], ckpt_step,
            state['stats']['train_time'] / 60)

def fc_best_step(run_dir):
    # pylint: disable=import-outside-toplevel
    from fm_eval import best_step
    return best_step(run_dir)

@torch.no_grad()
def reference_images(embed, batch, device):
    # pylint: disable=import-outside-toplevel
    from uvcgan_s.config import Args
    from uvcgan_s.cgan   import construct_model

    outdir = os.environ.get('UVCGAN_S_OUTDIR', 'outdir')
    args   = Args.load(os.path.join(outdir, PUBLISHED))
    model  = construct_model(args.savedir, args.config, is_train = False,
                             device = device)
    model.load(None)
    (gen, norm) = (model.models.ema_gen_ba.eval(), model.data_norm)

    run = os.path.join(fc.out_root(), 'ext_regress_l1_s0')
    ckpt = [ c for c in fc.list_checkpoints(run)
             if c.endswith(f'step_{fc_best_step(run):08d}.pt') ][0]
    (method, net, _, _) = fc.load_run(run, ckpt, device, 'ema')
    reg = fc.Decomposer(method, net)

    images = { 'uvcgan-s published' : [], 'regress_l1' : [] }
    for start in range(0, len(embed), batch):
        m = torch.from_numpy(embed[start:start + batch]).to(device)
        m = m.float().unsqueeze(1)
        images['uvcgan-s published'].append(
            norm.denormalize(gen(norm.normalize(m)))[:, 1].cpu()
        )
        images['regress_l1'].append(reg(m)[:, 1].cpu())

    return { k : torch.cat(v) for (k, v) in images.items() }

def main():
    # pylint: disable=too-many-locals
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    kernel  = ev.cone_kernel(ev.R_JET).to(device)

    (embed, signal) = ev.load_pairs(
        os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0,
        truth = cmdargs.truth
    )
    embed  = embed[:cmdargs.n_events]
    signal = signal[:cmdargs.n_events]
    truth  = ev.Truth(embed, signal, ev.cone_kernel(ev.R_JET), 10.0, device)

    names = READOUTS if cmdargs.readouts == 'all' \
        else cmdargs.readouts.split(',')
    rows  = []
    shown = {}

    # the error of an empty image outside the jet cone is the true signal
    # there: other particles of the PYTHIA event
    zero = score_images(
        torch.zeros(embed.shape, dtype = torch.float32), truth,
        cmdargs.batch, device, kernel
    )
    rows.append({ 'model' : 'empty image', 'readout' : '-',
                  **{ k : v for (k, v) in zero.items()
                      if k.startswith('l1') or k.startswith('abs') } })
    print(f"empty image: l1_out {zero['l1_out']:.4f}"
          f" ({zero['abs_err_out_gev']:.1f} GeV/event)", flush = True)

    if cmdargs.references:
        for (label, images) in reference_images(
            embed, cmdargs.batch, device
        ).items():
            rows.append({ 'model' : label, 'readout' : '-',
                          **score_images(images, truth, cmdargs.batch,
                                         device, kernel) })
            print(format_row(rows[-1]), flush = True)
            shown[label] = images

            if cmdargs.clean_references:
                for name in names:
                    if needs_converged(name) or name == 'coarse':
                        continue
                    (cleaned, regions) = clean_image(
                        images, embed, name, cmdargs.batch, device, kernel
                    )
                    rows.append({
                        'model' : label, 'readout' : name,
                        **score_images(cleaned, truth, cmdargs.batch, device,
                                       kernel, regions),
                    })
                    print(format_row(rows[-1]), flush = True)

    for run_dir in cmdargs.runs:
        t0 = time.perf_counter()
        (bkg, label, step, minutes) = flow_backgrounds(
            run_dir.rstrip('/'), embed, cmdargs.batch, device, cmdargs.truth
        )
        print(f'{label}: step {step} ({minutes:.0f} min), backgrounds in'
              f' {time.perf_counter() - t0:.0f} s', flush = True)

        for name in names:
            (images, regions) = ([], [])
            for start in range(0, len(embed), cmdargs.batch):
                m  = torch.from_numpy(embed[start:start + cmdargs.batch])
                m  = m.to(device).float()
                b4 = bkg['b4'][start:start + cmdargs.batch].to(device)
                bc = bkg['bc'][start:start + cmdargs.batch].to(device)
                (s_hat, region) = readout(name, m, b4, bc, kernel)
                images.append(s_hat.cpu())
                if region is not None:
                    regions.append(region.cpu())

            images = torch.cat(images)
            rows.append({
                'model' : label, 'step' : step, 'readout' : name,
                **score_images(
                    images, truth, cmdargs.batch, device, kernel,
                    torch.cat(regions) if regions else None
                ),
            })
            print(format_row(rows[-1]), flush = True)

            if (run_dir == cmdargs.runs[0]) and \
                    (name in cmdargs.show.split(',')):
                shown[f'OT-CFM {name}'] = images

    if cmdargs.figure:
        event_display(embed, signal, truth, shown, cmdargs.figure)

    df = pd.DataFrame(rows)
    df['truth'] = cmdargs.truth
    out = cmdargs.out or os.path.join(
        fc.out_root(), f'readout_{cmdargs.truth}.csv'
    )
    df.to_csv(out, mode = 'a', header = not os.path.exists(out),
              index = False)
    print(f'wrote {out}')

def event_display(embed, signal, truth, shown, path, n = 4):
    """Mixture, true signal and extracted signals of a few events with a
    jet, energies in GeV (negative towers in blue)."""
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    events = np.nonzero(truth.jets.cpu().numpy())[0][:n]
    panels = [ ('mixture', embed), ('true signal', signal) ] + [
        (k, v.numpy()) for (k, v) in shown.items()
    ]

    (fig, axes) = plt.subplots(
        n, len(panels), figsize = (2.6 * len(panels), 1.35 * n),
        squeeze = False
    )
    # a scale that shows the stray energy of a few 100 MeV per tower
    norm = TwoSlopeNorm(vmin = -1, vcenter = 0, vmax = 4)

    for (i, e) in enumerate(events):
        for (j, (title, images)) in enumerate(panels):
            ax  = axes[i][j]
            img = np.asarray(images[e], dtype = np.float32)
            im  = ax.imshow(img, cmap = 'RdBu_r', norm = norm,
                            aspect = 'auto', origin = 'lower')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize = 8)
            ax.scatter([ int(truth.col[e]) ], [ int(truth.row[e]) ],
                       marker = '+', color = 'k', s = 30, lw = 0.8)

    fig.colorbar(im, ax = axes, shrink = 0.6, label = 'GeV per tower')
    fig.savefig(path, dpi = 110, bbox_inches = 'tight')
    print(f'wrote {path}')

def format_row(r):
    extra = f"  jet in region {r['jet_in_region']:.3f}" \
        if 'jet_in_region' in r and pd.notna(r.get('jet_in_region')) else ''
    return (f"{r['model']:>20s} {r['readout']:>16s}  l1_sig {r['l1_sig']:.4f}"
            f"  in {r['l1_in']:.3f}  out {r['l1_out']:.4f}"
            f" ({r['abs_err_out_gev']:5.1f} GeV/event)"
            f"  mse in {r['mse_in']:.3f} out {r['mse_out']:.4f}  jes {r['jes']:.3f}"
            f"  jer_cal {r['jer_cal']:.3f}{extra}")

if __name__ == '__main__':
    main()
