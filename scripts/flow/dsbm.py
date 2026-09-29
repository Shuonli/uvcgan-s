#!/usr/bin/env python
"""Online alpha-DSBM (De Bortoli, Korshunova, Mnih, Doucet, "Schrodinger
Bridge Flow for Unpaired Data Translation", NeurIPS 2024, arXiv:2409.09347,
Algorithm 1) for the closure test of the flow study (FLOW_NOTES.md,
"alpha-DSBM closure test"), and the shared code of its Gaussian check
(dsbm_gauss.py).

Reference process: Brownian motion with variance eps per unit time,
dX = sqrt(eps) dB. Its bridge between x0 (t = 0) and x1 (t = 1):

    Interp_t(x0, x1, z) = (1 - t) x0 + t x1 + sqrt(eps t (1 - t)) z

One network, two drifts (Algorithm 1, Appendix K): a direction input s (1 =
forward, 0 = backward) through the same sinusoidal embedding and 2-layer MLP
as the time, the two outputs concatenated. Each direction runs on its own
clock u in [0, 1] from its start to its end: forward u = t, backward u = 1 -
t (the reverse time of Algorithm 1's backward loss).

Endpoint parameterisation with Appendix J's preconditioning: the model's
endpoint of the current direction, E[X_end | X_u = x], is

    endpoint(s, u, x) = c_skip(u) x + c_out(u) nn(s, u, c_in(u) x)
    drift(s, u, x)    = (endpoint(s, u, x) - x) / (1 - u)

(coefficients(): c_in = D^-1/2, c_skip = u / D, c_out = (1 - u^2/D)^1/2,
D = (1 - u)^2 + u^2 + eps u (1 - u); Appendix J's c_i, c_s, c_o rewritten for
the endpoint, as the reference DSBM's `mean_match` predicts the endpoint).
So forward, (E[X1 | X_t] - X_t) / (1 - t), the Markovian projection of
Definition 2.2, and backward, (E[X0 | X_t] - X_t) / t at reverse time 1 - t:
the two targets of Algorithm 1's losses (13). The network regresses (X_end
- c_skip x) / c_out, of unit variance at every u, with unit weight (Karras
et al.'s choice in Appendix J): the same per-time minimiser as loss (13),
without its singular weight near u = 1 (the drift target (X1 - X_t)/(1 - t)
has variance eps t / (1 - t) there). As u -> 1, c_skip -> 1 and c_out -> 0,
so the endpoint tends to x exactly. Training times t ~ U[1e-4, 1 - 1e-4] as
in the reference. (A first version predicted the endpoint without the skip;
the Gaussian check failed with it: 8% slope errors of the network near u = 1
inflated the generated variance, and the online loop amplified them.)

Sampling (Euler-Maruyama on N equal steps, h = 1 / N):
    x <- x + h drift(s, u_k, x) + sqrt(eps h) z_k,   u_k = k h,  k < N - 1
and the last step returns the endpoint prediction, i.e. x + h drift at u =
1 - h without the final increment's noise: the reference implementation's
endpoint handling (`mean_match`, `sample = True`, used both for its cached
training samples of transfer tasks and at test time). Diffusion is kept at
every other step. Brownian increments can be passed in, to couple N and 2N
steps (dW_coarse_k = dW_fine_2k + dW_fine_2k+1).

Stages (all with Adam, gradient clip 1, EMA of the parameters):
- pretrain: bridge matching on independent pairs (pi0 x pi1), half of each
  batch trains the forward drift, half the backward one (Algorithm 1, 2-7);
- refine: alpha-DSBM (Algorithm 1, 8-16). Per update, b = B/2 real sources X0
  and b real targets X1; X1_hat from the forward SDE started at X0, X0_hat
  from the backward SDE started at X1, both with the EMA parameters and no
  gradient; the forward drift is trained on (X0_hat, X1), the backward drift
  on (X0, X1_hat); loss = (l_fwd + l_bwd) / 2;
- continue: the pretrain loss (independent pairs), continued from a
  pretrained checkpoint: the control for refine at matched GPU-hours.
refine and continue start from a pretrained checkpoint (its raw weights, and
its EMA as their EMA), with a fresh optimiser (Appendix K) and warm-up.

No true pairs, OT assignment, adversarial or shape losses, or teacher enter
training. The generated endpoints stay in model coordinates (standardised
log(E + 0.1)); nothing is decoded, clipped or re-encoded during training.

    dsbm.py --label NAME --stage pretrain --minutes 30 --eps 1.0
    dsbm.py --label NAME --stage refine --init RUN --minutes 90
    dsbm.py --label NAME --stage continue --init RUN --minutes 90

The PYTHIA -> JEWEL translation pilot (FLOW_NOTES.md, "PYTHIA -> JEWEL
translation pilot") runs the same code on its own pools, normalisation and
run area, with a population monitor (there is no per-jet truth):

    dsbm.py ... --source-domain tr_pythia --target-domain tr_jewel \
        --norm-path OUTDIR/sphenix/flow/translation/norm.json \
        --outdir OUTDIR/sphenix/flow/translation/runs --monitor translation
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

import fm_common as fc

T_EPS = 1e-4          # training times t ~ U[T_EPS, 1 - T_EPS] (reference)
EMB   = 128           # sinusoidal features of t and of s

def embed(values, dim = EMB):
    # pylint: disable=import-outside-toplevel
    from torchcfm.models.unet.nn import timestep_embedding
    return timestep_embedding(values, dim)

def expand(v, x):
    return v.view(-1, *[ 1 ] * (x.dim() - 1))

def coefficients(u, eps, x, var_start = 1.0, var_end = 1.0):
    """Appendix J's preconditioning, written for the endpoint (derived there
    for unit-variance, independent endpoints; any choice keeps the
    minimiser): with D = (1 - u)^2 + u^2 + eps u (1 - u) = E|X_u|^2 / d,

        endpoint = c_skip x + c_out nn(c_in x),
        c_in = D^(-1/2),  c_skip = u / D,  c_out = (1 - u^2 / D)^(1/2)

    i.e. Appendix J's 1 + c_s (1 - t) = t / D and (1 - t) c_o = c_out. At
    u = 0 the network predicts the endpoint (c_skip = 0, c_out = 1); as u ->
    1, c_skip -> 1 and c_out -> 0, so the endpoint tends to x exactly and no
    network error is amplified by the drift's 1 / (1 - u).

    With endpoint variances var_start, var_end per element (Appendix J's
    generalisation to E|X0|^2 = s0^2 d, E|X1|^2 = s1^2 d): D = (1 - u)^2
    var_start + u^2 var_end + eps u (1 - u), c_skip = u var_end / D, c_out =
    (var_end (1 - u^2 var_end / D))^1/2."""
    uu = expand(u, x)
    vs = expand(var_start, x) if torch.is_tensor(var_start) else var_start
    ve = expand(var_end, x) if torch.is_tensor(var_end) else var_end
    d  = (1 - uu)**2 * vs + uu**2 * ve + eps * uu * (1 - uu)
    return (d.rsqrt(), uu * ve / d, (ve * (1 - uu**2 * ve / d)).clamp(min = 0).sqrt())

class Preconditioned(torch.nn.Module):
    """endpoint(s, u, x) = c_skip x + c_out raw(s, u, c_in x); subclasses
    define the network raw(s, u, x) and set `eps` (and `var0`, `var1`, the
    per-element second moments of pi0 and pi1 in model coordinates)."""
    eps  = None
    var0 = 1.0
    var1 = 1.0

    def scales(self, s):
        """(var_start, var_end) of each direction: forward pi0 -> pi1."""
        fwd = s > 0.5
        v0 = torch.full_like(s, self.var0)
        v1 = torch.full_like(s, self.var1)
        return (torch.where(fwd, v0, v1), torch.where(fwd, v1, v0))

    def endpoint(self, s, u, x):
        (c_in, c_skip, c_out) = coefficients(u, self.eps, x, *self.scales(s))
        return c_skip * x + c_out * self.raw(s, u, c_in * x)

    def regression(self, s, u, x, target):
        """Per-element loss of the network's own regression, target (X_end -
        c_skip x) / c_out of unit variance: Karras et al.'s unit weight
        (Appendix J), the endpoint error divided by c_out^2."""
        (c_in, c_skip, c_out) = coefficients(u, self.eps, x, *self.scales(s))
        return (self.raw(s, u, c_in * x) - (target - c_skip * x) / c_out)**2

# --- networks: raw(s, u, x)

class BridgeUVCGAN(Preconditioned, fc.UVCGANVelocity):
    """The UVCGAN-S velocity backbone (fm_common.UVCGANVelocity: the
    generator, time added to its extra/style token), with the direction:
    sinusoidal(s) -> 2-layer MLP, concatenated with the time MLP's output
    and projected to the token width, then added to the extra token."""

    def __init__(self, c_in, c_out, shape, eps):
        fc.UVCGANVelocity.__init__(self, c_in, c_out, shape)
        self.eps = eps
        width = fc.UVCGAN_GENERATOR['features'] * fc.UVCGAN_GENERATOR['n_ext']
        self.dir_embed = torch.nn.Sequential(
            torch.nn.Linear(EMB, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.cond = torch.nn.Linear(2 * width, width)

    def raw(self, s, u, x):
        c = torch.cat([ self.time_embed(embed(u)), self.dir_embed(embed(s)) ], dim = 1)
        self.gen.net.get_bottleneck().time = self.cond(c)
        return self.gen(x)

class BridgeMLP(Preconditioned):
    """The Gaussian check's network (paper Appendix K.2): time and direction
    each through sinusoidal features and a 2-layer MLP (256 hidden, 50
    out), concatenated with x, then a 2-layer MLP with 256 hidden units."""

    def __init__(self, dim, eps, hidden = 256, emb_out = 50):
        super().__init__()
        self.eps = eps
        def mlp(n_in, n_out):
            return torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.SiLU(),
                                       torch.nn.Linear(hidden, n_out))
        self.t_embed = mlp(EMB, emb_out)
        self.s_embed = mlp(EMB, emb_out)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 2 * emb_out, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, dim))

    def raw(self, s, u, x):
        return self.net(torch.cat([ x, self.t_embed(embed(u)), self.s_embed(embed(s)) ],
                                  dim = 1))

# --- the bridge, losses and sampler

def interp(x0, x1, t, eps, z):
    tt = expand(t, x0)
    return (1 - tt) * x0 + tt * x1 + torch.sqrt(eps * tt * (1 - tt)) * z

def bridge_loss(model, x0, x1, direction, eps, gen):
    """Algorithm 1's l_fwd (direction 1: target x1 at time t) or l_bwd
    (direction 0: target x0 at reverse time 1 - t), endpoint form."""
    n = len(x0)
    t = torch.rand(n, device = x0.device, generator = gen) * (1 - 2 * T_EPS) + T_EPS
    z = torch.randn(x0.shape, device = x0.device, generator = gen)
    xt = interp(x0, x1, t, eps, z)
    s  = torch.full((n,), float(direction), device = x0.device)
    if direction == 1:
        (u, target) = (t, x1)
    else:
        (u, target) = (1 - t, x0)
    return torch.mean(model.regression(s, u, xt, target))

@torch.no_grad()
def sample_sde(model, x, direction, eps, n_steps, gen = None, increments = None):
    """Euler-Maruyama from x along `direction` (1: forward from pi0, 0:
    backward from pi1), n_steps equal steps; `increments` (n_steps, *x.shape)
    Brownian increments with variance h (else drawn from `gen`). The last step
    returns the endpoint prediction (no final noise), c.f. the module notes."""
    h = 1.0 / n_steps
    s = torch.full((len(x),), float(direction), device = x.device)
    for k in range(n_steps):
        u = torch.full((len(x),), k * h, device = x.device)
        end = model.endpoint(s, u, x)
        if k == n_steps - 1:
            return end
        if increments is None:
            dw = math.sqrt(h) * torch.randn(x.shape, device = x.device, generator = gen)
        else:
            dw = increments[k]
        x = x + h * (end - x) / (1 - k * h) + math.sqrt(eps) * dw
    return x

def coarsen(increments):
    """Brownian increments of 2N steps -> those of N steps (sums of pairs)."""
    return increments[0::2] + increments[1::2]

# --- one update of each stage (shared by the closure and the Gaussian check)

def pretrain_loss(model, x0, x1, eps, gen):
    """Algorithm 1, lines 3-5: independent (x0, x1); the first half trains
    the forward drift, the second half the backward one."""
    b = len(x0) // 2
    l_f = bridge_loss(model, x0[:b], x1[:b], 1, eps, gen)
    l_b = bridge_loss(model, x0[b:], x1[b:], 0, eps, gen)
    return (0.5 * (l_f + l_b), l_f, l_b)

@torch.no_grad()
def rollout(sampler, x0, x1, eps, n_steps, gen):
    """Algorithm 1, lines 10-11: X1_hat from the forward SDE started at the
    real X0, X0_hat from the backward SDE started at the real X1 (no
    gradient)."""
    x1_hat = sample_sde(sampler, x0, 1, eps, n_steps, gen)
    x0_hat = sample_sde(sampler, x1, 0, eps, n_steps, gen)
    return (x0_hat, x1_hat)

def refine_loss(model, x0, x1, x0_hat, x1_hat, eps, gen):
    """Algorithm 1, lines 12-14: the forward drift on (X0_hat, X1), the
    backward drift on (X0, X1_hat)."""
    l_f = bridge_loss(model, x0_hat, x1, 1, eps, gen)
    l_b = bridge_loss(model, x0, x1_hat, 0, eps, gen)
    return (0.5 * (l_f + l_b), l_f, l_b)

def optimizer_step(model, ema, opt, loss, grad_clip, ema_decay):
    """Adam step with the gradient norm clipped, then the EMA update
    theta_ema = gamma theta_ema + (1 - gamma) theta (Algorithm 1, lines 6, 15)."""
    opt.zero_grad(set_to_none = True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    with torch.no_grad():
        torch._foreach_lerp_(       # pylint: disable=protected-access
            list(ema.parameters()), list(model.parameters()), 1 - ema_decay)
    return gnorm

# --- data

class ClosureData:
    """The closure pools in model coordinates: z = (log(E + 0.1) - mu) / sd
    of the canvases (closure_data.py, norm_closure.json), on the GPU; or any
    other pair of jet pools with their own normalisation (the translation
    pilot: tr_pythia, tr_jewel and translation/norm.json)."""

    def __init__(self, device, target_domain = 'closure_tgt',
                 source_domain = 'closure_src', norm_path = None):
        # the closure normalisation, fitted by closure_data.py (as fm_train)
        with open(norm_path or fc.Norm.path(method = 'jetflow'),
                  encoding = 'utf-8') as f:
            self.method = fc.Method('jetflow', fc.Norm(json.load(f)))
        self.src = torch.from_numpy(np.load(fc.cache_path(source_domain))).to(device)
        self.tgt = torch.from_numpy(np.load(fc.cache_path(target_domain))).to(device)
        self.device = device

    def z(self, energy):
        return self.method.source(energy.float())

    def source(self, n, gen):
        idx = torch.randint(len(self.src), (n,), device = self.device, generator = gen)
        return self.z(self.src[idx])

    def target(self, n, gen):
        idx = torch.randint(len(self.tgt), (n,), device = self.device, generator = gen)
        return self.z(self.tgt[idx])

    def second_moments(self, n = 20000):
        """Per-element E[z^2] of the two pools (the preconditioning's
        endpoint variances), on the first n canvases of each."""
        return tuple(float(self.z(pool[:n]).pow(2).mean())
                     for pool in (self.src, self.tgt))

# --- training

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'alpha-DSBM on the closure test')
    parser.add_argument('--label', required = True)
    parser.add_argument('--stage', required = True,
                        choices = [ 'pretrain', 'refine', 'continue' ])
    parser.add_argument('--init', default = None,
        help = 'pretrained run (refine, continue): its last checkpoint')
    parser.add_argument('--eps', type = float, default = 1.0)
    parser.add_argument('--steps', type = int, default = 30,
        help = 'Euler-Maruyama steps of the online rollouts')
    parser.add_argument('--batch', type = int, default = 256)
    parser.add_argument('--lr', type = float, default = 2e-4)
    parser.add_argument('--warmup', type = int, default = 1000)
    parser.add_argument('--ema', type = float, default = 0.999)
    parser.add_argument('--grad-clip', type = float, default = 1.0)
    parser.add_argument('--minutes', type = float, required = True,
        help = 'training time, rollouts included')
    parser.add_argument('--ckpt-minutes', type = float, default = 10)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--monitor-n', type = int, default = 1000)
    parser.add_argument('--target-domain', default = 'closure_tgt')
    parser.add_argument('--source-domain', default = 'closure_src')
    parser.add_argument('--norm-path', default = None,
        help = 'normalisation JSON (default: the closure\'s norm_closure.json)')
    parser.add_argument('--outdir', default = None,
        help = 'directory holding the runs (default: OUTDIR/sphenix/flow);'
               ' --init is looked up there too')
    parser.add_argument('--monitor', default = 'closure',
        choices = [ 'closure', 'translation' ],
        help = 'closure: shape EMD to T(J) on validation pairs; translation:'
               ' population scores of validation outputs against held-out'
               ' JEWEL (translation_eval.monitor)')
    return parser.parse_args()

def build_model(config, device):
    shape = tuple(config['shape'])
    model = BridgeUVCGAN(1, 1, shape, config['eps']).to(device)
    (model.var0, model.var1) = (config.get('var0', 1.0), config.get('var1', 1.0))
    return model

def load_bridge(run_dir, ckpt, device, which = 'ema'):
    with open(os.path.join(run_dir, 'config.json'), encoding = 'utf-8') as f:
        config = json.load(f)
    state = torch.load(ckpt, map_location = device, weights_only = False)
    model = build_model(config, device)
    model.load_state_dict(state[which])
    model.eval()
    return (model, state, config)

def last_checkpoint(run_dir):
    ckpts = fc.list_checkpoints(run_dir)
    assert ckpts, f'no checkpoint in {run_dir}'
    return ckpts[-1]

def monitor(model, data, pairs, jets, eps, n_steps, seed):
    """Shape EMD and energy response of single SDE samples on validation
    pairs (held-out truth: monitoring only)."""
    model.eval()
    gen = torch.Generator(device = data.device).manual_seed(seed)
    src = data.z(torch.as_tensor(pairs['src'], device = data.device))
    out = sample_sde(model, src, 1, eps, n_steps, gen)
    energy = data.method.norm.energy(out[:, 0], 'jet').clamp(min = 0).cpu().numpy()
    (full, shape) = jets.emds(energy, pairs['tgt'])
    e_out = energy.sum((1, 2))
    e_in  = pairs['src'].sum((1, 2))
    return { 'val_emd_gev' : float(full.mean()), 'val_shape_emd' : float(np.nanmean(shape)),
             'val_response' : float(np.mean(e_out / e_in)) }

def main():
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    # pylint: disable=import-outside-toplevel
    from closure_eval import load_pairs, Jets

    cmdargs = parse_cmdargs()
    device  = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    runs    = cmdargs.outdir or fc.out_root()
    run_dir = os.path.join(runs, cmdargs.label)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok = True)
    translation = cmdargs.monitor == 'translation'
    assert translation == (cmdargs.source_domain == 'tr_pythia') \
        == (cmdargs.target_domain == 'tr_jewel') == (cmdargs.norm_path is not None), \
        'the translation pools go with their own normalisation and monitor'
    data = ClosureData(device, cmdargs.target_domain, cmdargs.source_domain,
                       cmdargs.norm_path)

    config = { 'label' : cmdargs.label, 'stage' : cmdargs.stage, 'eps' : cmdargs.eps,
               'midpoint_noise_sd' : 0.5 * math.sqrt(cmdargs.eps),
               'rollout_steps' : cmdargs.steps, 'batch' : cmdargs.batch,
               'lr' : cmdargs.lr, 'warmup' : cmdargs.warmup, 'ema' : cmdargs.ema,
               'grad_clip' : cmdargs.grad_clip, 'minutes' : cmdargs.minutes,
               'seed' : cmdargs.seed, 'backbone' : 'uvcgan', 'shape' : list(fc.JET_SHAPE),
               'target_domain' : cmdargs.target_domain,
               'source_domain' : cmdargs.source_domain,
               'norm_path' : cmdargs.norm_path, 'monitor' : cmdargs.monitor,
               'norm' : data.method.norm.to_dict(),
               'parameterisation' : 'endpoint = c_skip x + c_out nn(c_in x) (Appendix J,'
                                    ' per-direction endpoint variances), drift ='
                                    ' (endpoint - x)/(1 - u), unit-weight regression',
               'gpu' : torch.cuda.get_device_name() }
    (config['var0'], config['var1']) = data.second_moments()
    step0 = 0
    if cmdargs.stage == 'pretrain':
        torch.manual_seed(cmdargs.seed)
        model = build_model(config, device)
        ema   = copy.deepcopy(model)
    else:
        assert cmdargs.init, '--init: the pretrained run'
        init_dir  = os.path.join(runs, cmdargs.init)
        init_ckpt = last_checkpoint(init_dir)
        with open(os.path.join(init_dir, 'config.json'), encoding = 'utf-8') as f:
            init_cfg = json.load(f)
        assert abs(init_cfg['eps'] - cmdargs.eps) < 1e-12, 'eps must match the pretraining'
        # the pools and normalisation of the pretraining (older closure
        # configs record only the target)
        for (key, default) in (('source_domain', 'closure_src'),
                               ('target_domain', 'closure_tgt'), ('norm_path', None)):
            assert init_cfg.get(key, default) == getattr(cmdargs, key), \
                f'{key} must match the pretraining'
        state = torch.load(init_ckpt, map_location = device, weights_only = False)
        model = build_model(config, device)
        model.load_state_dict(state['raw'])
        ema = build_model(config, device)
        ema.load_state_dict(state['ema'])
        step0 = int(state['stats']['step'])
        config.update({ 'init' : cmdargs.init, 'init_checkpoint' : os.path.basename(init_ckpt),
                        'init_step' : step0,
                        'init_train_time' : state['stats']['train_time'] })
    model.train()
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    config['n_params'] = fc.count_params(model)
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding = 'utf-8') as f:
        json.dump(config, f, indent = 4)

    # fresh optimiser in every stage (Appendix K: reset when finetuning starts)
    opt = torch.optim.Adam(model.parameters(), lr = cmdargs.lr, betas = (0.9, 0.999))
    stage_seed = { 'pretrain' : 0, 'refine' : 1, 'continue' : 2 }[cmdargs.stage]
    g_data = torch.Generator(device = device).manual_seed(1000 * cmdargs.seed + 10 * stage_seed + 1)
    g_loss = torch.Generator(device = device).manual_seed(1000 * cmdargs.seed + 10 * stage_seed + 2)
    g_roll = torch.Generator(device = device).manual_seed(1000 * cmdargs.seed + 10 * stage_seed + 3)

    if translation:
        import translation_eval
        val  = translation_eval.MonitorSets(cmdargs.monitor_n, device)
        jets = None
    else:
        val  = load_pairs('val', cmdargs.monitor_n)
        jets = Jets(val['row'], device)
    b    = cmdargs.batch // 2
    hist = []
    timing = { 'rollout' : 0.0, 'update' : 0.0 }
    train_time = 0.0
    next_ckpt  = cmdargs.ckpt_minutes * 60
    update = 0
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    while train_time < cmdargs.minutes * 60:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        update += 1
        lr = cmdargs.lr * min(1.0, update / cmdargs.warmup)
        for group in opt.param_groups:
            group['lr'] = lr

        if cmdargs.stage in ('pretrain', 'continue'):
            x0 = data.source(cmdargs.batch, g_data)
            x1 = data.target(cmdargs.batch, g_data)
            t1 = time.perf_counter()
            (loss, l_f, l_b) = pretrain_loss(model, x0, x1, cmdargs.eps, g_loss)
        else:
            # the rollouts use the EMA parameters (the paper's default)
            x0 = data.source(b, g_data)
            x1 = data.target(b, g_data)
            (x0_hat, x1_hat) = rollout(ema, x0, x1, cmdargs.eps, cmdargs.steps, g_roll)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            timing['rollout'] += t1 - t0
            (loss, l_f, l_b) = refine_loss(model, x0, x1, x0_hat, x1_hat, cmdargs.eps,
                                           g_loss)
        gnorm = optimizer_step(model, ema, opt, loss, cmdargs.grad_clip, cmdargs.ema)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        timing['update'] += t2 - t1
        train_time += t2 - t0

        if update % 100 == 0:
            hist.append({ 'update' : update, 'step' : step0 + update,
                          'train_time' : train_time, 'loss' : float(loss),
                          'loss_fwd' : float(l_f), 'loss_bwd' : float(l_b),
                          'gnorm' : float(gnorm), 'lr' : lr,
                          'time_rollout' : timing['rollout'],
                          'time_update' : timing['update'] })

        if train_time >= next_ckpt or train_time >= cmdargs.minutes * 60:
            next_ckpt += cmdargs.ckpt_minutes * 60
            stats = { 'step' : step0 + update, 'updates' : update,
                      'train_time' : train_time, 'stage' : cmdargs.stage,
                      'total_train_time' : train_time + config.get('init_train_time', 0.0),
                      **{ f'time_{k}' : v for (k, v) in timing.items() },
                      'peak_mem_gb' : torch.cuda.max_memory_allocated() / 2**30 }
            fc.save_checkpoint(
                os.path.join(run_dir, 'checkpoints', f'step_{step0 + update:08d}.pt'),
                raw = model.state_dict(), ema = ema.state_dict(),
                norm = data.method.norm.to_dict(), stats = stats, config = config)
            if translation:
                mon = val.monitor(ema, data, cmdargs.eps, cmdargs.steps, 12345)
                text = '  '.join(f'{k} {v:.3f}' for (k, v) in mon.items())
            else:
                mon = monitor(ema, data, val, jets, cmdargs.eps, cmdargs.steps, 12345)
                text = (f"val shape EMD {mon['val_shape_emd']:.4f}  EMD "
                        f"{mon['val_emd_gev']:.3f} GeV  response {mon['val_response']:.3f}")
            model.train()
            print(f"{cmdargs.stage} update {update:6d} ({step0 + update}) "
                  f"{train_time / 60:6.1f} min  loss {float(loss):.4f} "
                  f"(fwd {float(l_f):.4f}, bwd {float(l_b):.4f})  {text}", flush = True)
            hist.append({ 'update' : update, 'step' : step0 + update,
                          'train_time' : train_time, **mon })
            pd.DataFrame(hist).to_csv(os.path.join(run_dir, 'history.csv'), index = False)

    summary = { 'updates' : update, 'train_time' : train_time,
                'end_to_end' : time.perf_counter() - t_start,
                'updates_per_s' : update / train_time,
                **{ f'time_{k}' : v for (k, v) in timing.items() },
                'rollout_share' : timing['rollout'] / train_time,
                'rollout_nfe_per_update' : (2 * b * cmdargs.steps
                                            if cmdargs.stage == 'refine' else 0),
                'peak_mem_gb' : torch.cuda.max_memory_allocated() / 2**30,
                'n_params' : config['n_params'] }
    with open(os.path.join(run_dir, 'summary.json'), 'w', encoding = 'utf-8') as f:
        json.dump(summary, f, indent = 4)
    print(json.dumps(summary, indent = 2))

if __name__ == '__main__':
    main()
