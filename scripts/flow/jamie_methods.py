"""Training pairs and losses of the toy study (FLOW_NOTES.md, "Jamie's toy
exercises with OT flow matching"), with fm_train.py's Method interface
(draw, endpoints, couple / pair, loss, last_parts), on full 24 x 64 canvases
of the toy pools (jamie_data.py).

Models (--method):
    toyflow  D, deterministic OT-CFM: x_t = (1 - t) a + t b, u = b - a,
             mean((v(t, x_t) - u)^2); solved from the source image
    toycond  C, conditional FM: (c, y) from the pairing, eps ~ N(0, I) drawn
             after it, x_t = (1 - t) eps + t y, mean((v(t, [x_t, c]) -
             (y - eps))^2) (fm_common.cond_fm_loss, every tower active);
             sampled from fresh noise with c held fixed

Pairings (--toy-pairing), always stated with the run:
    unpaired   pure unpaired: a batch of the source pool and an independent
               batch of the target pool, paired by the minibatch plan
               (fm_common.RowCoupling, exact or entropic; one target per
               source row) on the squared L2 of the standardised states
    synthetic  synthetic paired (D only): M = J + B from the prior jet pool
               and the UE pool, trained M -> B; the pairs are exact because
               we made the sum
    paired     true-paired positive control (C only): c a vacuum jet, y one
               of its own 8 independent quenchings
    hybrid     the synthetic term plus lambda_U times the unpaired term
               (mixture data -> UE pool), each averaged on its own batch

Extra terms of the continuations (--toy-extra, weights --toy-lambda):
    abs, bal, ring   Exercise 1, on the synthetic batch, through the local
               endpoint surrogate b* = x_t + (1 - t) v(t, x_t) at t ~ U(0,
               0.25) (not a solved endpoint): J* = M - B*(GeV); abs = sum|J* -
               J| / E_J; bal = the mean |J* - J| within occupied, near-empty
               (R < 0.4) and far-empty towers, averaged over the three, over
               E_J / N_occ; ring = sum over Jamie's rings of ((sum J* - sum
               J) / E_J)^2
    energy     Exercise 3 (D): ((E_out - E_in) / E_in)^2, total GeV of an
               actual differentiable solve of --roll-n source images
    inv_global, inv_far   Exercise 4: mean |E_out - E_in| per tower over all
               towers or over R >= 1.0 from the source axis, in units of
               0.6 GeV (the toy's mean UE per tower), on solved endpoints
               (D from the source; C from noise with c held fixed)
    profile    Exercise 3 (C): the batch-mean radial profile (GeV) of solved
               samples around their condition's axis against that of
               independent target jets around their own axes
    div        Exercise 3 (C): mode seeking, -min(mean|x1 - x2| /
               mean|eps1 - eps2|, cap), two solves of the same conditions;
               cap = the same ratio for two unrelated target images
    ueprof     the 2 x 2 (hybrid): the batch-mean radial profile of the
               surrogate background B* of the mixture data around their axes
               against that of independent UE images around the same axes

Axes (eta0, phi0) are the generator's, supplied as controlled toy
information to every method that uses a region or a profile.
"""

import json
import math
import os

import numpy as np
import torch

import fm_common as fc
import toycalo as tc

RING_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1e9)
PROFILE_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2)
RHO0 = 0.6

def jamie_root():
    return os.path.join(fc.out_root(), 'jamie')

def load_norm():
    with open(os.path.join(jamie_root(), 'norm.json'), encoding = 'utf-8') as f:
        return fc.Norm(json.load(f))

class Pool:
    """A training pool on the GPU (float16 images) with its axes."""

    def __init__(self, name, device):
        d = os.path.join(jamie_root(), 'cache')
        self.name = name
        self.x = torch.from_numpy(np.load(os.path.join(d, f'{name}.npy'))).to(device)
        with np.load(os.path.join(d, f'{name}_meta.npz')) as f:
            self.axis = torch.tensor(np.stack([ f['eta0'], f['phi0'] ], 1),
                                     dtype = torch.float32, device = device)
        self.reps = None
        path = os.path.join(d, f'{name}_reps.npy')
        if os.path.exists(path):
            self.reps = torch.from_numpy(np.load(path)).to(device)

    def __len__(self):
        return len(self.x)

def offsets(axis):
    """(deta, dphi, dr), each (N, 24, 64), of the tower centres to the axes."""
    dev = axis.device
    eta_c = torch.tensor(tc.ETA_C, dtype = torch.float32, device = dev)
    phi_c = torch.tensor(tc.PHI_C, dtype = torch.float32, device = dev)
    deta = (eta_c[None, :, None] - axis[:, 0, None, None]).expand(-1, -1, tc.NPHI)
    dphi = torch.remainder(phi_c[None, None, :] - axis[:, 1, None, None] + math.pi,
                           2 * math.pi) - math.pi
    dphi = dphi.expand(-1, tc.NETA, -1)
    return (deta, dphi, torch.sqrt(deta**2 + dphi**2))

def rollout(net, x0, cond, nfe):
    """A differentiable midpoint solve from t = 0 to 1 in nfe evaluations."""
    x = x0
    steps = nfe // 2
    h = 1.0 / steps
    def vel(t, x):
        inp = x if cond is None else torch.cat((x, cond), dim = 1)
        return net(torch.full((x.shape[0],), t, device = x.device), inp)
    for k in range(steps):
        t = k * h
        xm = x + 0.5 * h * vel(t, x)
        x = x + h * vel(t + 0.5 * h, xm)
    return x

def radial_profile(e, dr, edges = PROFILE_EDGES):
    """(N, n_bins) energy in radial bins (GeV)."""
    return torch.stack([ (e * ((dr >= lo) & (dr < hi))).sum((1, 2))
                         for (lo, hi) in zip(edges[:-1], edges[1:]) ], 1)

class ToyMethod:
    """One toy run's pairs and loss (module docstring)."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, a, norm, device):
        assert a.method in ('toyflow', 'toycond'), a.method
        self.name = a.method
        self.norm = norm
        self.kind = a.toy_kind
        self.pairing = a.toy_pairing
        self.device = device
        self.sigma = 0.0
        self.n_paired = 0
        self.matcher = None
        self.extra = [ e for e in (a.toy_extra or '').split(',') if e ]
        self.lam = dict(zip(self.extra, [ float(x) for x in (a.toy_lambda or '').split(',') if x ]))
        assert set(self.lam) == set(self.extra), (self.extra, self.lam)
        self.lam_u = a.toy_lambda_u
        self.roll_n = a.roll_n
        self.roll_nfe = a.roll_nfe
        self.coupling = fc.RowCoupling(a.coupling, a.ot_reg, a.sinkhorn_tol)
        self.cost = a.toy_cost
        assert self.cost in ('full', 'nearfar'), self.cost
        self.last_parts = {}
        self.cur = {}
        names = { 'source' : a.toy_source, 'target' : a.toy_target,
                  'prior' : a.toy_prior, 'ue' : a.toy_ue, 'data' : a.toy_data }
        self.pools = { k : Pool(v, device) for (k, v) in names.items() if v }
        if self.pairing in ('synthetic', 'hybrid'):
            assert 'prior' in self.pools and 'ue' in self.pools
        if self.pairing == 'hybrid':
            assert 'data' in self.pools
        if self.pairing == 'paired':
            assert self.name == 'toycond' and self.pools['source'].reps is not None
        if self.pairing == 'unpaired':
            assert 'source' in self.pools and 'target' in self.pools
        self.gens = {}
        self.div_cap = None
        # independent target images, for the profile and diversity terms
        self.need_ref = any(e in ('profile', 'div') for e in self.extra)

    # --- fm_train interface

    domains = ()

    @staticmethod
    def pop_recovery():
        return None

    @property
    def coupled(self):
        return self.pairing in ('unpaired', 'hybrid')

    def set_streams(self, seed, step):
        """Batches, times, noise, row draws, rollout subsets: one stream each,
        so arms that differ only in their pairing or extra terms see the same
        batches, times and noise, step by step."""
        for (k, name) in enumerate([ 'data', 'time', 'noise', 'pair', 'roll', 'data_u',
                                     'time_u', 'time_x' ]):
            g = torch.Generator(device = self.device)
            g.manual_seed((7_000_003 + k) * seed + 101 + 13 * k + step)
            self.gens[name] = g

    def z(self, e):
        return self.norm.z(e, self.kind)

    def energy(self, x):
        return self.norm.energy(x, self.kind)

    def idx(self, pool, n, stream = 'data'):
        return torch.randint(len(pool), (n,), device = self.device, generator = self.gens[stream])

    def draw(self, data, batch, pool = None):
        # pylint: disable=unused-argument
        out = {}
        if self.pairing in ('synthetic', 'hybrid'):
            (ip, iu) = (self.idx(self.pools['prior'], batch), self.idx(self.pools['ue'], batch))
            jet = self.pools['prior'].x[ip].float()
            ue = self.pools['ue'].x[iu].float()
            out['syn'] = { 'jet' : jet, 'ue' : ue, 'mix' : jet + ue,
                           'axis' : self.pools['prior'].axis[ip] }
        if self.pairing == 'hybrid':
            # their own stream: the synthetic batches are those of lambda_U = 0
            (idm, iu) = (self.idx(self.pools['data'], batch, 'data_u'),
                         self.idx(self.pools['ue'], batch, 'data_u'))
            out['unp'] = { 'src' : self.pools['data'].x[idm].float(),
                           'tgt' : self.pools['ue'].x[iu].float(),
                           'axis' : self.pools['data'].axis[idm] }
        if self.pairing == 'unpaired':
            (i, j) = (self.idx(self.pools['source'], batch), self.idx(self.pools['target'], batch))
            out['unp'] = { 'src' : self.pools['source'].x[i].float(),
                           'tgt' : self.pools['target'].x[j].float(),
                           'axis' : self.pools['source'].axis[i],
                           'tgt_axis' : self.pools['target'].axis[j] }
        if self.pairing == 'paired':
            src = self.pools['source']
            i = self.idx(src, batch)
            r = torch.randint(src.reps.shape[1], (batch,), device = self.device,
                              generator = self.gens['data'])
            out['pair'] = { 'src' : src.x[i].float(), 'tgt' : src.reps[i, r].float(),
                            'axis' : src.axis[i] }
        if self.need_ref:
            t = self.pools['target']
            j = self.idx(t, batch)
            out['ref'] = { 'tgt' : t.x[j].float(), 'axis' : t.axis[j] }
        return out

    def endpoints(self, batch):
        self.cur = batch
        if 'syn' in batch:
            s = batch['syn']
            return (self.z(s['mix']).unsqueeze(1), self.z(s['ue']).unsqueeze(1), None)
        part = batch.get('unp') or batch['pair']
        (a, b) = (self.z(part['src']).unsqueeze(1), self.z(part['tgt']).unsqueeze(1))
        if self.name == 'toycond':
            return (None, b, a)
        return (a, b, None)

    def couple(self, x0, x1, n_pairs = None):
        """D: the unpaired batch's plan (the hybrid's second term is paired
        here and kept for loss()); the synthetic term passes through."""
        # pylint: disable=unused-argument
        u = self.cur['unp']
        (a, b) = (self.z(u['src']).unsqueeze(1), self.z(u['tgt']).unsqueeze(1))
        if self.cost == 'nearfar':
            (_, bj, j) = self.coupling_nearfar(a, b, u)
        else:
            (_, bj, j) = self.coupling(a, b, self.gens['pair'])
        u['x0'] = a
        u['x1'] = bj
        u['j'] = j
        self.last_parts = dict(self.coupling.last)
        if self.pairing == 'hybrid':
            return (x0, x1)
        return (a, bj)

    def coupling_nearfar(self, a, b, u):
        """The pre-specified alternative cost (Exercise 4, used only because the
        audit found full-image OT dominated by the UE): squared L2 over the
        towers within R < 1.0 of either jet axis and over the rest, each
        divided by its median over the batch's pairs, then added; the same
        plan and row draws."""
        (za, zb) = (a.flatten(1).double(), b.flatten(1).double())
        (_, _, dra) = offsets(u['axis'])
        (_, _, drb) = offsets(u['tgt_axis'])
        (na, nb) = ((dra < 1.0).flatten(1).double(), (drb < 1.0).flatten(1).double())
        def masked(ma, mb):
            # sum over towers of ma_i mb_j (za_i - zb_j)^2, for every (i, j)
            return ((ma * za**2) @ mb.T + ma @ (mb * zb**2).T - 2 * (ma * za) @ (mb * zb).T)
        ones_a = torch.ones_like(na)
        ones_b = torch.ones_like(nb)
        full = masked(ones_a, ones_b)
        near = masked(na, ones_b) + masked(ones_a, nb) - masked(na, nb)
        far = full - near
        cost = near / near.median() + far / far.median()
        plan = torch.from_numpy(self.coupling.exact.ot_fn(
            np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b)),
            cost.cpu().numpy())).to(a.device)
        j = fc.row_targets(torch.log(plan), self.gens['pair'])
        self.coupling.last = { 'plan_err' : 0.0,
                               'unique_targets' : float(torch.unique(j).numel() / len(j)) }
        return (a, b[j], j)

    def pair(self, cond, x1):
        """C: every condition row with one target drawn from the plan."""
        (c, y, j) = self.coupling(cond, x1, self.gens['pair'])
        self.cur['unp']['j'] = j
        self.last_parts = dict(self.coupling.last)
        return (c, y)

    # --- losses

    def times(self, n, lo = 0.0, hi = 1.0, stream = 'time'):
        return lo + (hi - lo) * torch.rand(n, device = self.device, generator = self.gens[stream])

    def fm(self, net, x0, x1, stream = 'time'):
        t = self.times(len(x0), stream = stream)
        tt = t.view(-1, 1, 1, 1)
        xt = (1 - tt) * x0 + tt * x1
        return torch.mean((net(t, xt) - (x1 - x0))**2)

    def loss(self, net, x0, x1, cond):
        parts = dict(self.last_parts)
        if self.name == 'toycond':
            eps = torch.randn(x1.shape, device = self.device, generator = self.gens['noise'])
            t = self.times(len(x1))
            total = fc.cond_fm_loss(net, cond, x1, eps, t)
            parts['loss_fm'] = total.detach()
            self.cur['cond'] = cond
        else:
            total = self.fm(net, x0, x1)
            parts['loss_fm'] = total.detach()
            if self.pairing == 'hybrid':
                u = self.cur['unp']
                lu = self.fm(net, u['x0'], u['x1'], 'time_u')
                parts['loss_unp'] = lu.detach()
                self.calls = getattr(self, 'calls', 0) + 1
                if (self.calls % 500 == 1) and (self.lam_u > 0):
                    # both pools reach the gradient: the two terms' gradient norms
                    params = [ p for p in net.parameters() if p.requires_grad ]
                    for (k, v) in [ ('gnorm_syn', total), ('gnorm_unp', self.lam_u * lu) ]:
                        g = torch.autograd.grad(v, params, retain_graph = True)
                        parts[k] = torch.sqrt(sum((x**2).sum() for x in g)).detach()
                total = total + self.lam_u * lu
        for name in self.extra:
            term = getattr(self, f'term_{name}')(net, x0, x1, cond)
            parts[f'loss_{name}'] = term.detach()
            total = total + self.lam[name] * term
        self.last_parts = parts
        return total

    # Exercise 1: the local endpoint surrogate on the synthetic batch

    def surrogate(self, net, s):
        """J* (GeV) of the synthetic batch at early times, the true J, the
        energy scale, and the masks; computed once per step."""
        if 'sur' in self.cur:
            return self.cur['sur']
        x0 = self.z(s['mix']).unsqueeze(1)
        x1 = self.z(s['ue']).unsqueeze(1)
        t = self.times(len(x0), 0.0, 0.25, 'time_x')
        tt = t.view(-1, 1, 1, 1)
        xt = (1 - tt) * x0 + tt * x1
        b_star = xt + (1 - tt) * net(t, xt)
        jstar = s['mix'] - self.energy(b_star[:, 0])
        (_, _, dr) = offsets(s['axis'])
        jet = s['jet']
        e_j = jet.sum((1, 2)).clamp(min = 1e-3)
        occ = jet > 0
        sur = { 'jstar' : jstar, 'jet' : jet, 'e_j' : e_j, 'dr' : dr, 'occ' : occ,
                'near' : (~occ) & (dr < 0.4), 'far' : (~occ) & (dr >= 0.4) }
        self.cur['sur'] = sur
        return sur

    def term_abs(self, net, x0, x1, cond):
        # pylint: disable=unused-argument
        s = self.surrogate(net, self.cur['syn'])
        return ((s['jstar'] - s['jet']).abs().sum((1, 2)) / s['e_j']).mean()

    def term_bal(self, net, x0, x1, cond):
        # pylint: disable=unused-argument
        s = self.surrogate(net, self.cur['syn'])
        err = (s['jstar'] - s['jet']).abs()
        scale = s['e_j'] / s['occ'].sum((1, 2)).clamp(min = 1)
        groups = []
        for m in (s['occ'], s['near'], s['far']):
            groups.append((err * m).sum((1, 2)) / m.sum((1, 2)).clamp(min = 1))
        return (sum(groups) / 3 / scale).mean()

    def term_ring(self, net, x0, x1, cond):
        # pylint: disable=unused-argument
        s = self.surrogate(net, self.cur['syn'])
        d = s['jstar'] - s['jet']
        tot = 0
        for (lo, hi) in zip(RING_EDGES[:-1], RING_EDGES[1:]):
            m = (s['dr'] >= lo) & (s['dr'] < hi)
            tot = tot + ((d * m).sum((1, 2)) / s['e_j'])**2
        return tot.mean()

    def term_ueprof(self, net, x0, x1, cond):
        """The hybrid's mixture data: surrogate B* around their axes against
        independent UE images around the same axes, batch means."""
        # pylint: disable=unused-argument
        u = self.cur['unp']
        n = min(len(u['src']), 128)
        x0u = self.z(u['src'][:n]).unsqueeze(1)
        x1u = u['x1'][:n]
        t = self.times(n, 0.0, 0.25, 'time_x')
        tt = t.view(-1, 1, 1, 1)
        xt = (1 - tt) * x0u + tt * x1u
        b_star = self.energy((xt + (1 - tt) * net(t, xt))[:, 0])
        (_, _, dr) = offsets(u['axis'][:n])
        ue = self.cur['syn']['ue'][:n]          # independent UE images
        p_gen = radial_profile(b_star, dr).mean(0)
        p_ref = radial_profile(ue, dr).mean(0)
        return ((p_gen - p_ref)**2).sum() / (p_ref**2).sum()

    # rollouts: actual solves on a subset

    def roll_idx(self, n):
        return torch.randperm(n, device = self.device, generator = self.gens['roll'])[:self.roll_n]

    def solved(self, net, part):
        """GeV images of an actual differentiable solve of a subset: D from
        the source images; C from fresh noise with the conditions held
        fixed. Returns (output, input, axis) in GeV."""
        k = self.roll_idx(len(part['src']))
        src = part['src'][k]
        if self.name == 'toycond':
            eps = torch.randn((len(k), 1, tc.NETA, tc.NPHI), device = self.device,
                              generator = self.gens['roll'])
            x = rollout(net, eps, self.z(src).unsqueeze(1), self.roll_nfe)
        else:
            x = rollout(net, self.z(src).unsqueeze(1), None, self.roll_nfe)
        return (self.energy(x[:, 0]), src, part['axis'][k])

    def part(self):
        return self.cur.get('unp') or self.cur.get('pair')

    def term_energy(self, net, x0, x1, cond):
        # pylint: disable=unused-argument
        (out, src, _) = self.solved(net, self.part())
        e_in = src.sum((1, 2)).clamp(min = 1e-3)
        return (((out.sum((1, 2)) - e_in) / e_in)**2).mean()

    def term_inv_global(self, net, x0, x1, cond):
        # pylint: disable=unused-argument
        (out, src, _) = self.solved(net, self.part())
        return (out - src).abs().mean() / RHO0

    def term_inv_far(self, net, x0, x1, cond):
        # pylint: disable=unused-argument
        (out, src, axis) = self.solved(net, self.part())
        (_, _, dr) = offsets(axis)
        far = dr >= 1.0
        return ((out - src).abs() * far).sum() / far.sum().clamp(min = 1) / RHO0

    def term_profile(self, net, x0, x1, cond):
        # pylint: disable=unused-argument
        (out, _, axis) = self.solved(net, self.part())
        (_, _, dr) = offsets(axis)
        ref = self.cur['ref']
        (_, _, dr_ref) = offsets(ref['axis'])
        p_gen = radial_profile(out, dr).mean(0)
        p_ref = radial_profile(ref['tgt'], dr_ref).mean(0)
        return ((p_gen - p_ref)**2).sum() / (p_ref**2).sum()

    def term_div(self, net, x0, x1, cond):
        # pylint: disable=unused-argument
        part = self.part()
        k = self.roll_idx(len(part['src']))
        c = self.z(part['src'][k]).unsqueeze(1)
        shape = (len(k), 1, tc.NETA, tc.NPHI)
        e1 = torch.randn(shape, device = self.device, generator = self.gens['roll'])
        e2 = torch.randn(shape, device = self.device, generator = self.gens['roll'])
        y1 = rollout(net, e1, c, self.roll_nfe)
        y2 = rollout(net, e2, c, self.roll_nfe)
        ratio = (y1 - y2).abs().mean((1, 2, 3)) / (e1 - e2).abs().mean((1, 2, 3))
        if self.div_cap is None:
            # two unrelated target images, standardised, over the same noise scale
            ref = self.z(self.cur['ref']['tgt']).unsqueeze(1)
            half = len(ref) // 2
            d_ref = (ref[:half] - ref[half:2 * half]).abs().mean()
            self.div_cap = float(d_ref / (2 / math.sqrt(math.pi)) )
        return -torch.clamp(ratio, max = self.div_cap).mean()
