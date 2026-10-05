#!/usr/bin/env python
"""Before the stochastic translation pilot's training (FLOW_NOTES.md,
"Stochastic conditional FM pilot"): the coupling audit that fixes the
entropic regularisation, the checks of the condjet code, and a Gaussian
positive control of the conditional sampler. All on training jets or
synthetic data; no validation or test jet is read.

    condfm_checks.py --audit  [--batches 32 --support 4]
    condfm_checks.py --checks [--reg REG]
    condfm_checks.py --gauss

--audit: AUDIT_SEED fixes --batches training-only batches (256 PYTHIA and
256 JEWEL jets each, drawn without replacement from the training pools), with
the trainer's cost: the squared L2 of the standardised jet-centred states.
For the entropic plan (fm_common.RowCoupling: balanced, uniform marginals,
log-domain Sinkhorn) the regularisation is set by the median over all rows of
the effective support exp(-sum_j q_ij log q_ij), q_i = P_i. / P_i.sum():
bisection in log reg to --support (4), rounded to two significant digits,
accepted if the median lies in [3, 5]. Reported at that value: the spread of
the row entropies, the marginal residuals, the solve's sweeps and time, and
the pairs drawn as the trainer draws them (one target per source row) next to
the exact plan's assignments and to random pairs: energy and shape
displacement, cost, and how far a soft target lies from the hard one.
    OUT/coupling_audit_grid.csv, coupling_audit_pairs.csv, coupling_audit.json

--checks: the noise-to-target path through cond_fm_loss itself (endpoints,
padding, the velocity as the derivative of the path), the condition/target
indexing after both plans (a shuffled copy must be paired back exactly; row
draws must follow P_i. / P_i.sum(); Method.pair must keep the conditions in
order), and fixed-noise reproducibility of the sampler (sample_cond) with the
translation network.  OUT/checks.json

--gauss: y = c + 0.2 xi in 16 dimensions (as 1 x 4 x 4 states, 4 of them
padding), trained with cond_fm_loss and sampled with sample_cond, the
trainer's and the evaluation's own functions, with a small MLP. For fixed c
the samples must recover the mean c and the sd 0.2.  OUT/gauss_check.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import fm_common as fc
import translation_eval as te
from closure_data import OFFSET
from jet_fidelity import EMD
from substructure import Geometry

AUDIT_SEED = 20261005
N = 256

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'condjet: audit and checks')
    parser.add_argument('--audit', action = 'store_true')
    parser.add_argument('--checks', action = 'store_true')
    parser.add_argument('--gauss', action = 'store_true')
    parser.add_argument('--partners', action = 'store_true',
        help = 'the couplings\' own conditional distribution of the first'
               ' 1000 test inputs: their targets over many training-like batches')
    parser.add_argument('--partner-batches', type = int, default = 2000)
    parser.add_argument('--pair-corr', action = 'store_true',
        help = 'descriptive: condition-target correlations of the audit pairs'
               ' at the frozen regularisation (no new choice)')
    parser.add_argument('--batches', type = int, default = 32)
    parser.add_argument('--support', type = float, default = 4.0)
    parser.add_argument('--tol', type = float, default = 1e-4)
    parser.add_argument('--reg', type = float, default = None,
        help = '--checks: the frozen regularisation (default: the audit\'s)')
    parser.add_argument('--out', default = 'docs/flow/translation/condfm')
    return parser.parse_args()

def norm():
    with open(fc.translation_norm_path(), encoding = 'utf-8') as f:
        return fc.Norm(json.load(f))

def audit_batches(n_batches, device):
    """Fixed training-only batches: (PYTHIA, JEWEL) canvases, GeV, (B, N, 16,
    16), and their jet-axis rows."""
    rng = np.random.default_rng(AUDIT_SEED)
    out = {}
    for d in [ 'pythia', 'jewel' ]:
        pool = np.load(fc.cache_path(f'tr_{d}'), mmap_mode = 'r')
        rows = te.train_meta(d)['row']
        idx  = rng.choice(len(pool), n_batches * N, replace = False)
        rank = torch.as_tensor(np.argsort(np.argsort(idx)))
        imgs = torch.from_numpy(np.asarray(pool[np.sort(idx)]).astype(np.float32))[rank]
        out[d] = (imgs.to(device).view(n_batches, N, 16, 16),
                  rows[idx].reshape(n_batches, N))
    return out

def median_support(states, reg, tol):
    coupling = fc.RowCoupling('entropic', reg, tol)
    supp = [ fc.effective_support(coupling.log_plan(c, y)) for (c, y) in states ]
    return (float(torch.cat(supp).median()), coupling)

def shape_emd(a, b, emd, geo):
    """Normalised-shape EMD between canvases (translation_report.preference)."""
    def unit(x):
        w = x[:, OFFSET:OFFSET + 9, OFFSET:OFFSET + 9].float().clamp(min = 0).cpu() \
            * geo.mask.cpu()
        return w / w.sum((1, 2), keepdim = True).clamp(min = 1e-12)
    return emd(unit(a), unit(b))[0]

def audit(cmdargs, device):
    # pylint: disable=too-many-locals,too-many-statements
    nm = norm()
    data = audit_batches(cmdargs.batches, device)
    (cp, cy) = (data['pythia'][0], data['jewel'][0])
    states = [ (nm.z(cp[b], 'jet').unsqueeze(1), nm.z(cy[b], 'jet').unsqueeze(1))
               for b in range(cmdargs.batches) ]
    costs = torch.stack([ torch.cdist(c.flatten(1), y.flatten(1))**2 for (c, y) in states ])

    # a coarse grid, for the record, then bisection in log reg
    grid = []
    for reg in [ 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256 ]:
        (med, _) = median_support(states, reg, cmdargs.tol)
        grid.append({ 'reg' : reg, 'median_support' : med })
        print(f'reg {reg:6.2f}: median support {med:.2f}', flush = True)
    lo = max(g['reg'] for g in grid if g['median_support'] < cmdargs.support)
    hi = min(g['reg'] for g in grid if g['median_support'] > cmdargs.support)
    for _ in range(30):
        mid = math.sqrt(lo * hi)
        (med, _) = median_support(states, mid, cmdargs.tol)
        (lo, hi) = (mid, hi) if med < cmdargs.support else (lo, mid)
        if hi / lo < 1.001:
            break
    reg = float(f'{math.sqrt(lo * hi):.2g}')

    # everything at the frozen value
    coupling = fc.RowCoupling('entropic', reg, cmdargs.tol)
    exact = fc.RowCoupling('exact')
    gen = torch.Generator(device = device).manual_seed(AUDIT_SEED)
    supp, ent, res, times, blocks, j_soft, j_hard, uniq = [], [], [], [], [], [], [], []
    for (c, y) in states:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lp = coupling.log_plan(c, y)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        blocks.append(coupling.solver.blocks)
        p = lp.double().exp()
        res.append([ float((p.sum(1) * N - 1).abs().max()), float((p.sum(0) * N - 1).abs().max()),
                     float((p.sum(1) - 1 / N).abs().sum()), float((p.sum(0) - 1 / N).abs().sum()),
                     float(p.sum() - 1) ])
        s = fc.effective_support(lp)
        supp.append(s)
        ent.append(torch.log(s))
        js = fc.row_targets(lp, gen)
        j_soft.append(js)
        uniq.append(float(torch.unique(js).numel() / N))
        j_hard.append(fc.row_targets(exact.log_plan(c, y), gen))
    supp = torch.cat(supp).cpu().numpy()
    ent = torch.cat(ent).cpu().numpy()
    res = np.array(res)

    # the pairs: soft draws, hard assignments, random pairs
    geo = Geometry(torch.device('cpu'))
    emd = EMD(geo)
    rng = torch.Generator(device = device).manual_seed(AUDIT_SEED + 1)
    j_rand = [ torch.randperm(N, device = device, generator = rng) for _ in states ]
    o_c = te.jet_observables(cp.view(-1, 16, 16), data['pythia'][1].reshape(-1), device)
    o_y = te.jet_observables(cy.view(-1, 16, 16), data['jewel'][1].reshape(-1), device)
    base = torch.arange(cmdargs.batches, device = device)[:, None] * N
    flat = lambda js: (torch.stack(js) + base).flatten().cpu().numpy()  # pylint: disable=unnecessary-lambda-assignment
    pairs = []
    picks = { 'hard (exact plan)' : flat(j_hard), 'soft (entropic plan)' : flat(j_soft),
              'random pairs' : flat(j_rand) }
    src = cp.view(-1, 16, 16)
    tgt = cy.view(-1, 16, 16)
    for (kind, j) in picks.items():
        cost = costs.view(-1, N).cpu()[torch.arange(len(j)), torch.as_tensor(j) % N]
        jt = torch.as_tensor(j, device = device)
        d_e  = o_y['E'][j] - o_c['E']
        row = { 'pairs' : kind, 'n' : len(j),
                'cost_mean' : float(cost.mean()),
                'dE_mean_gev' : float(d_e.mean()), 'dE_sd_gev' : float(d_e.std()),
                'dE_rms_gev' : float(np.sqrt(np.mean(d_e**2))),
                'abs_dlogE_mean' : float(np.mean(np.abs(np.log(o_y['E'][j] / o_c['E'])))),
                'shape_emd_mean' : float(np.nanmean(shape_emd(src, tgt[jt], emd, geo))) }
        for q in [ 'girth', 'zlead', 'ptd', 'mass' ]:
            d = o_y[q][j] - o_c[q]
            row[f'd{q}_mean'] = float(np.nanmean(d))
            row[f'd{q}_rms'] = float(np.sqrt(np.nanmean(d**2)))
        pairs.append(row)
        print(row, flush = True)
    (jh, js) = (picks['hard (exact plan)'], picks['soft (entropic plan)'])
    same = float(np.mean(jh == js))
    (th, ts) = (torch.as_tensor(jh, device = device), torch.as_tensor(js, device = device))
    soft_vs_hard = {
        'share_soft_equals_hard' : same,
        'shape_emd_soft_to_hard_target' : float(np.nanmean(
            shape_emd(tgt[ts], tgt[th], emd, geo)[js != jh])),
        'dE_rms_soft_to_hard_target_gev' : float(np.sqrt(np.mean(
            (o_y['E'][js] - o_y['E'][jh])**2))),
        'unique_targets_per_batch_soft' : float(np.mean(uniq)),
    }

    q = np.quantile(supp, [ 0.05, 0.25, 0.5, 0.75, 0.95 ])
    summary = {
        'audit_seed' : AUDIT_SEED, 'batches' : cmdargs.batches, 'batch' : N,
        'cost' : 'squared L2 of the standardised states (16 x 16, translation norm.json)',
        'cost_median' : float(costs.median()),
        'cost_row_min_median' : float(costs.min(dim = 2).values.median()),
        'target_support' : cmdargs.support, 'reg' : reg,
        'reg_over_median_cost' : reg / float(costs.median()),
        'accepted' : bool(3 <= q[2] <= 5),
        'support_quantiles_5_25_50_75_95' : q.tolist(),
        'support_mean' : float(supp.mean()), 'support_min' : float(supp.min()),
        'support_max' : float(supp.max()),
        'row_entropy_mean' : float(ent.mean()), 'row_entropy_sd' : float(ent.std()),
        'sinkhorn_tol_row_l1' : cmdargs.tol,
        'row_marginal_max_rel_dev' : float(res[:, 0].max()),
        'col_marginal_max_rel_dev' : float(res[:, 1].max()),
        'row_marginal_l1_max' : float(res[:, 2].max()),
        'col_marginal_l1_max' : float(res[:, 3].max()),
        'mass_minus_one_max' : float(np.abs(res[:, 4]).max()),
        'sweeps_median' : float(np.median(blocks)) * coupling.solver.block,
        'sweeps_max' : int(max(blocks)) * coupling.solver.block,
        'solve_ms_median' : 1000 * float(np.median(times)),
        **soft_vs_hard,
    }
    os.makedirs(cmdargs.out, exist_ok = True)
    import pandas as pd       # pylint: disable=import-outside-toplevel
    pd.DataFrame(grid).to_csv(os.path.join(cmdargs.out, 'coupling_audit_grid.csv'),
                              index = False)
    pd.DataFrame(pairs).to_csv(os.path.join(cmdargs.out, 'coupling_audit_pairs.csv'),
                               index = False)
    with open(os.path.join(cmdargs.out, 'coupling_audit.json'), 'w', encoding = 'utf-8') as f:
        json.dump(summary, f, indent = 4)
    print(json.dumps(summary, indent = 4))

def pair_corr(cmdargs, device):
    """How much of a condition's energy and shape its sampled targets carry:
    Pearson correlations between condition and target observables of the
    audit pairs (hard, soft at the frozen reg, random). Descriptive; the
    regularisation is read, not chosen, here."""
    import pandas as pd       # pylint: disable=import-outside-toplevel
    nm  = norm()
    reg = reg_of(cmdargs)
    data = audit_batches(cmdargs.batches, device)
    (cp, cy) = (data['pythia'][0], data['jewel'][0])
    o_c = te.jet_observables(cp.view(-1, 16, 16), data['pythia'][1].reshape(-1), device)
    o_y = te.jet_observables(cy.view(-1, 16, 16), data['jewel'][1].reshape(-1), device)
    gen = torch.Generator(device = device).manual_seed(AUDIT_SEED)
    rng = torch.Generator(device = device).manual_seed(AUDIT_SEED + 1)
    picks = { 'hard (exact plan)' : [], 'soft (entropic plan)' : [], 'random pairs' : [] }
    (exact, soft) = (fc.RowCoupling('exact'), fc.RowCoupling('entropic', reg, cmdargs.tol))
    for b in range(cmdargs.batches):
        (c, y) = (nm.z(cp[b], 'jet').unsqueeze(1), nm.z(cy[b], 'jet').unsqueeze(1))
        picks['soft (entropic plan)'].append(fc.row_targets(soft.log_plan(c, y), gen) + b * N)
        picks['hard (exact plan)'].append(fc.row_targets(exact.log_plan(c, y), gen) + b * N)
        picks['random pairs'].append(torch.randperm(N, device = device, generator = rng) + b * N)
    rows = []
    for (kind, js) in picks.items():
        j = torch.cat(js).cpu().numpy()
        row = { 'pairs' : kind, 'reg' : reg if 'soft' in kind else np.nan }
        for q in [ 'E', 'mass', 'girth', 'ptd', 'zlead', 'core', 'lead' ]:
            (a, b) = (np.asarray(o_c[q], float), np.asarray(o_y[q], float)[j])
            ok = np.isfinite(a) & np.isfinite(b)
            row[f'corr_{q}'] = float(np.corrcoef(a[ok], b[ok])[0, 1])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cmdargs.out, 'coupling_audit_corr.csv'), index = False)
    print(df.round(3).to_string(index = False))

def partners(cmdargs, device):
    """What each coupling assigns to a jet: the first 1000 PYTHIA test
    inputs, 256 at a time (random subsets) in --partner-batches batches with
    256 random JEWEL training jets each, as in training; the same batches for
    both plans, one target drawn per row. Each input collects about 500
    targets per plan: the conditional distribution the noise-to-target model
    is trained to reproduce (not a physical one). Nothing is fitted.
    OUTDIR/.../translation/condfm/outputs/partners.npz: (input, pool index)
    pairs of each plan."""
    nm  = norm()
    reg = reg_of(cmdargs)
    n   = 1000
    src = torch.from_numpy(te.load_set('test', 'pythia')['canvas'][:n]).float().to(device)
    pool = torch.from_numpy(np.load(fc.cache_path('tr_jewel'))).to(device)
    g = torch.Generator(device = device).manual_seed(AUDIT_SEED + 2)
    plans = { 'hard' : fc.RowCoupling('exact'),
              'soft' : fc.RowCoupling('entropic', reg, cmdargs.tol) }
    out = { k : [] for k in plans }
    for _ in range(cmdargs.partner_batches):
        i = torch.randperm(n, device = device, generator = g)[:N]
        k = torch.randint(len(pool), (N,), device = device, generator = g)
        c = nm.z(src[i], 'jet').unsqueeze(1)
        y = nm.z(pool[k].float(), 'jet').unsqueeze(1)
        for (name, cpl) in plans.items():
            j = fc.row_targets(cpl.log_plan(c, y), g)
            out[name].append(torch.stack([ i, k[j] ]).cpu())
    path = os.path.join(fc.translation_root(), 'condfm', 'outputs', 'partners.npz')
    os.makedirs(os.path.dirname(path), exist_ok = True)
    np.savez_compressed(path, reg = reg, batches = cmdargs.partner_batches,
                        **{ k : torch.cat(v, 1).numpy() for (k, v) in out.items() })
    print(f'wrote {path}', flush = True)

# --- checks

class Probe(torch.nn.Module):
    """A 'network' that records its input and returns a fixed velocity."""

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.seen = None

    def forward(self, t, inp):
        self.seen = (t, inp)
        return self.value

class Counter(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.calls = 0

    def forward(self, t, inp):
        self.calls += 1
        return self.net(t, inp)

def reg_of(cmdargs):
    if cmdargs.reg is not None:
        return cmdargs.reg
    with open(os.path.join(cmdargs.out, 'coupling_audit.json'), encoding = 'utf-8') as f:
        return json.load(f)['reg']

def checks(cmdargs, device):
    # pylint: disable=too-many-locals,too-many-statements
    torch.backends.cudnn.benchmark = False
    nm  = norm()
    reg = reg_of(cmdargs)
    data = audit_batches(2, device)
    c = nm.z(data['pythia'][0][0], 'jet').unsqueeze(1).double()
    y = nm.z(data['jewel'][0][0], 'jet').unsqueeze(1).double()
    meth = fc.Method('condjet', nm, source = 'tr_pythia', target = 'tr_jewel')
    (mask, pad) = meth.padding(device)
    gen = torch.Generator(device = device).manual_seed(1)
    eps = torch.randn(y.shape, device = device, generator = gen, dtype = torch.float64)
    x0  = fc.noise_start(eps, mask, pad)
    out = {}

    # 1. the path inside cond_fm_loss: endpoints, padding, derivative
    probe = Probe(torch.zeros_like(y))
    path = {}
    for tv in [ 0.0, 1.0, 0.37 ]:
        t = torch.full((N,), tv, device = device, dtype = torch.float64)
        loss = fc.cond_fm_loss(probe, c, y, x0, t)
        (xt, cond_in) = (probe.seen[1][:, :1], probe.seen[1][:, 1:])
        path[tv] = xt
        out[f'path_t{tv:g}'] = {
            'x_t_minus_expected' : float((xt - ((1 - tv) * x0 + tv * y)).abs().max()),
            'cond_unchanged' : float((cond_in - c).abs().max()),
            'loss_minus_mean_u2' : float(loss - torch.mean((y - x0)**2)),
            'padding_dev' : float((xt - pad)[~mask.expand_as(xt)].abs().max()) }
    out['x_t0_equals_noise_start'] = float((path[0.0] - x0).abs().max())
    out['x_t1_equals_target'] = float((path[1.0] - y).abs().max())
    out['noise_on_active_only'] = bool(torch.all(x0[~mask.expand_as(x0)] == pad))
    out['target_padding_is_pad'] = float((y - pad)[~mask.expand_as(y)].abs().max())
    h = 1e-4
    fd = []
    for tv in [ 0.1, 0.5, 0.9 ]:
        xs = []
        for s in (tv - h, tv + h):
            t = torch.full((N,), s, device = device, dtype = torch.float64)
            fc.cond_fm_loss(probe, c, y, x0, t)
            xs.append(probe.seen[1][:, :1])
        fd.append(float(((xs[1] - xs[0]) / (2 * h) - (y - x0)).abs().max()))
    out['velocity_minus_finite_difference_max'] = max(fd)
    out['velocity_on_padding_max'] = float((y - x0)[~mask.expand_as(y)].abs().max())

    # 2. indexing after the plans
    cf = c.float()
    perm = torch.randperm(N, device = device, generator = torch.Generator(device = device)
                          .manual_seed(2))
    shuffled = cf[perm]
    for (kind, r) in [ ('exact', None), ('entropic', 0.05) ]:
        cpl = fc.RowCoupling(kind, r, cmdargs.tol)
        (src, tgt, j) = cpl(cf, shuffled, torch.Generator(device = device).manual_seed(3))
        out[f'shuffled_copy_{kind}'] = {
            'conditions_in_order' : float((src - cf).abs().max()),
            'share_paired_to_own_copy' : float((perm[j] == torch.arange(N, device = device))
                                               .float().mean()),
            'max_condition_target_diff' : float((src - tgt).abs().max()),
            'target_is_shuffled_row_j' : float((tgt - shuffled[j]).abs().max()) }
    cpl = fc.RowCoupling('entropic', reg, cmdargs.tol)
    lp = cpl.log_plan(cf, y.float())
    q = torch.softmax(lp.double(), dim = 1)
    draws = 20000
    g = torch.Generator(device = device).manual_seed(4)
    counts = torch.zeros_like(q)
    for _ in range(draws // 1000):
        js = torch.stack([ fc.row_targets(lp, g) for _ in range(1000) ])     # (1000, N)
        counts.scatter_add_(1, js.T, torch.ones_like(js.T, dtype = counts.dtype))
    freq = counts / draws
    z = (freq - q).abs() / torch.sqrt(q * (1 - q) / draws).clamp(min = 1e-12)
    big = q > 0.01
    out['row_draw_frequencies'] = {
        'reg' : reg, 'draws_per_row' : draws,
        'max_abs_freq_minus_q' : float((freq - q).abs().max()),
        'max_z_where_q_gt_0p01' : float(z[big].max()),
        'share_z_gt_3_where_q_gt_0p01' : float((z[big] > 3).float().mean()),
        'mass_where_q_lt_1e-6' : float(freq[q < 1e-6].sum() / N) }
    meth_s = fc.Method('condjet', nm, source = 'tr_pythia', target = 'tr_jewel',
                       coupling = 'entropic', ot_reg = reg)
    meth_s.pair_gen = torch.Generator(device = device).manual_seed(5)
    (c2, y2) = meth_s.pair(cf, y.float())
    yf = y.float().flatten(1)
    match = (y2.flatten(1)[:, None, :] == yf[None, :, :]).all(-1)       # (N, N)
    out['method_pair'] = {
        'conditions_in_order' : float((c2 - cf).abs().max()),
        'every_target_is_a_batch_target' : bool(match.any(1).all()),
        'unique_targets' : meth_s.last_parts['unique_targets'],
        'plan_err' : meth_s.last_parts['plan_err'] }

    # 3. fixed-noise reproducibility, with the translation network
    torch.manual_seed(0)
    net = fc.construct_net('condjet', backbone = 'uvcgan').to(device).eval()
    cnt = Counter(net)
    src = data['pythia'][0][1][:64]
    e1 = torch.randn((64, 1, 16, 16), device = device,
                     generator = torch.Generator(device = device).manual_seed(6))
    e2 = torch.randn((64, 1, 16, 16), device = device,
                     generator = torch.Generator(device = device).manual_seed(7))
    smp = fc.CondJetSampler(meth, cnt, 128, 'midpoint', clip = False)
    (a, b, d) = (smp(src, e1), smp(src, e1.clone()), smp(src, e2))
    zc = fc.sample_cond(net, meth.source(src), e1, mask, pad, 16)
    out['reproducibility'] = {
        'network' : 'condjet UVCGAN velocity, random init (seed 0)',
        'same_c_same_eps_max_abs_gev' : float((a - b).abs().max()),
        'same_c_other_eps_rms_gev' : float(((a - d)**2).mean().sqrt()),
        'nfe_per_sample_call' : cnt.calls // 3,
        'padding_dev_after_solve' : float((zc - pad)[~mask.expand_as(zc)].abs().max()) }
    with open(os.path.join(cmdargs.out, 'checks.json'), 'w', encoding = 'utf-8') as f:
        json.dump(out, f, indent = 4)
    print(json.dumps(out, indent = 4))

# --- Gaussian positive control

class GaussMLP(torch.nn.Module):
    """v(t, [x, c]) for (N, 2, 4, 4) inputs: time embedding, x and c through
    a three-layer SiLU MLP."""

    def __init__(self, width = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(32 + 32, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 16))

    def forward(self, t, inp):
        # pylint: disable=import-outside-toplevel
        from torchcfm.models.unet.nn import timestep_embedding
        t = t.reshape(-1).expand(inp.shape[0]) if t.numel() == 1 else t.reshape(-1)
        h = torch.cat((timestep_embedding(t, 32), inp.flatten(1)), dim = 1)
        return self.net(h).view(-1, 1, 4, 4)

def gauss(cmdargs, device):
    # pylint: disable=too-many-locals
    torch.manual_seed(0)
    s = 0.2
    pad = 0.3
    mask = torch.ones((1, 1, 4, 4), dtype = torch.bool, device = device)
    mask[..., [ 0, 0, 3, 3 ], [ 0, 3, 0, 3 ]] = False          # 4 padding corners
    def draw_c(n, g):
        return fc.noise_start(torch.randn((n, 1, 4, 4), device = device, generator = g),
                              mask, pad)
    net = GaussMLP().to(device)
    opt = torch.optim.Adam(net.parameters(), lr = 1e-3)
    steps, batch = 15000, 1024
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device = device).manual_seed(10)
    t0 = time.perf_counter()
    for _ in range(steps):
        c = draw_c(batch, g)
        y = fc.noise_start(c + s * torch.randn(c.shape, device = device, generator = g),
                           mask, pad)
        eps = torch.randn(c.shape, device = device, generator = g)
        t = torch.rand(batch, device = device, generator = g)
        loss = fc.cond_fm_loss(net, c, y, fc.noise_start(eps, mask, pad), t)
        opt.zero_grad(set_to_none = True)
        loss.backward()
        opt.step()
        sched.step()
    train_s = time.perf_counter() - t0
    net.eval()
    k, m = 8, 20000
    conds = draw_c(k, torch.Generator(device = device).manual_seed(11))
    rows = []
    ge = torch.Generator(device = device).manual_seed(12)
    act = mask[0, 0]
    for i in range(k):
        cc = conds[i:i + 1].expand(m, -1, -1, -1)
        eps = torch.randn((m, 1, 4, 4), device = device, generator = ge)
        x = fc.sample_cond(net, cc, eps, mask, pad, 128)
        x2 = fc.sample_cond(net, cc, eps, mask, pad, 256)
        xr = fc.sample_cond(net, cc, eps, mask, pad, 128)
        v = x[:, 0][:, act]                                     # (m, 12)
        dev = v - conds[i, 0][act]
        cov = torch.cov(v.T)
        corr = cov / torch.sqrt(torch.diag(cov)[:, None] * torch.diag(cov)[None, :])
        rows.append({
            'mean_minus_c_max' : float(dev.mean(0).abs().max()),
            'sd_min' : float(v.std(0).min()), 'sd_max' : float(v.std(0).max()),
            'offdiag_corr_max' : float((corr - torch.eye(12, device = device)).abs().max()),
            'padding_dev' : float((x - pad)[:, 0][:, ~act].abs().max()),
            'nfe128_vs_256_rms' : float(((x - x2)**2).mean().sqrt()),
            'same_eps_repeat_max' : float((x - xr).abs().max()) })
    worst = { key : (max if key not in ('sd_min',) else min)(r[key] for r in rows)
              for key in rows[0] }
    worst['sd_max'] = max(r['sd_max'] for r in rows)
    res = {
        'relation' : 'y = c + 0.2 xi on 12 active coordinates, 4 padding at 0.3;'
                     ' c ~ N(0, 1); MLP, 15k updates of 1024, Adam 1e-3 cosine',
        'train_seconds' : train_s, 'final_loss' : float(loss),
        'conditions' : k, 'samples_per_condition' : m, 'solver' : 'midpoint 128',
        'per_condition' : rows, 'worst' : worst,
        'mc_sd_of_mean' : s / math.sqrt(m),
        'pass_mean' : worst['mean_minus_c_max'] < 0.02,
        'pass_sd' : (worst['sd_min'] > 0.95 * s) and (worst['sd_max'] < 1.05 * s),
        'pass_padding' : worst['padding_dev'] == 0.0,
        'references' : { 'noise ignored: sd' : 0.0,
                         'condition ignored: mean - c, sd' : 'c, sqrt(1.04) = 1.02' } }
    res['pass'] = bool(res['pass_mean'] and res['pass_sd'] and res['pass_padding'])
    with open(os.path.join(cmdargs.out, 'gauss_check.json'), 'w', encoding = 'utf-8') as f:
        json.dump(res, f, indent = 4)
    print(json.dumps({ k : v for (k, v) in res.items() if k != 'per_condition' }, indent = 4))

def main():
    cmdargs = parse_cmdargs()
    device = torch.device('cuda')
    os.makedirs(cmdargs.out, exist_ok = True)
    if cmdargs.audit:
        audit(cmdargs, device)
    if cmdargs.checks:
        checks(cmdargs, device)
    if cmdargs.gauss:
        gauss(cmdargs, device)
    if cmdargs.pair_corr:
        pair_corr(cmdargs, device)
    if cmdargs.partners:
        partners(cmdargs, device)

if __name__ == '__main__':
    main()
