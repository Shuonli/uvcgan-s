"""Flow matching for the sPHENIX decomposition: what fm_train.py,
fm_eval.py and coupling_diag.py share.

State. Tower energies E (GeV, 24 x 64) are mapped by the baseline's data
norm psi(E) = log(E + 0.1) and standardised per channel with a mean and
deviation fitted on training events only (`Norm`). The flow state is
x = (background, signal), 2 x 24 x 64.

Methods (c.f. FLOW_NOTES.md):

    otcfm    x0 = (psi(m), psi(0)) of a real mixture m, x1 = (psi(b), psi(s))
             of an independently drawn background and signal; exact
             minibatch OT (TorchCFM ExactOptimalTransportConditionalFlowMatcher)
    sbcfm    same endpoints; entropic OT plan, reg = 2 sigma^2, Brownian
             bridge (TorchCFM SchrodingerBridgeConditionalFlowMatcher);
             velocity only, sampled with the probability-flow ODE
    condcfm  x1 = (psi(b), psi(s)), x0 ~ N(0, 1), conditioned on the synthetic
             mixture psi(b + s); independent coupling (TorchCFM
             ConditionalFlowMatcher, the pairs are exact by construction)
    regress  the same network mapping psi(b + s) to (psi(b), psi(s)) by mean
             squared error, no flow

Nothing here reads the index files: the domains are drawn independently.
"""

import json
import os
import sys

import numpy as np
import torch

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
)
from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.models.unet.unet import UNetModel

import ot as pot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'slurm'))
import eval_val_truth as ev   # pylint: disable=wrong-import-position

DATA_PATH = 'sphenix/2025-06-05_jet_bkg_sub'
BIAS      = 0.1
SHAPE     = (24, 64)
METHODS   = [ 'otcfm', 'sbcfm', 'condcfm', 'regress' ]

# domains each method draws from; condcfm and regress build their mixtures
DOMAINS = {
    'otcfm'   : ('embed', 'background', 'signal'),
    'sbcfm'   : ('embed', 'background', 'signal'),
    'condcfm' : ('background', 'signal'),
    'regress' : ('background', 'signal'),
}

def data_root():
    return os.path.join(os.environ.get('UVCGAN_S_DATA', 'data'), DATA_PATH)

def out_root():
    return os.path.join(
        os.environ.get('UVCGAN_S_OUTDIR', 'outdir'), 'sphenix', 'flow'
    )

def cache_path(domain):
    """Flat float16 copy of a training domain, c.f. make_cache.py."""
    return os.path.join(out_root(), 'cache', f'train_{domain}.npy')

def draw_indices(rng, n, k):
    """k distinct random indices below n, increasing."""
    idx = np.unique(rng.integers(0, n, size = k))

    while len(idx) < k:
        idx = np.unique(np.concatenate(
            (idx, rng.integers(0, n, size = k - len(idx)))
        ))

    return idx

def read_random(domain, n, rng):
    """n random training events of a domain from the cache, in random order."""
    data = np.load(cache_path(domain), mmap_mode = 'r')
    idx  = draw_indices(rng, len(data), n)
    return data[idx].astype(np.float32)[rng.permutation(n)]

class GPUData:
    """The training domains in GPU memory (float16), drawn independently.

    Every batch takes fresh i.i.d. random indices per domain, so there is no
    correspondence between the domains -- as with the baseline's unpaired
    loader, which shuffles each domain on its own. `stream` (the step a run
    resumes from) keeps a resumed run from replaying its first batches.
    """

    def __init__(self, domains, device, seed, stream = 0):
        self.data = {}

        for d in domains:
            self.data[d] = torch.from_numpy(np.load(cache_path(d))).to(device)

        self.gen = torch.Generator(device = device)
        self.gen.manual_seed(1_000_003 * seed + stream)

    def sizes(self):
        return { d : len(x) for (d, x) in self.data.items() }

    def batch(self, n):
        return {
            d : x[torch.randint(
                len(x), (n,), device = x.device, generator = self.gen
            )].float() for (d, x) in self.data.items()
        }

class Norm:
    """psi(E) = log(E + 0.1), standardised per kind of image.

    Kinds: `bkg` (background domain), `sig` (signal domain), `syn`
    (synthetic mixtures b + s). Fitted on training events only.
    """

    KINDS = ('bkg', 'sig', 'syn')

    def __init__(self, stats):
        self.stats = {
            k : (float(v[0]), float(v[1])) for (k, v) in stats.items()
        }

    @staticmethod
    def psi(energy):
        return torch.log(energy + BIAS)

    def z(self, energy, kind):
        (mean, std) = self.stats[kind]
        return (self.psi(energy) - mean) / std

    def energy(self, z, kind):
        (mean, std) = self.stats[kind]
        return torch.exp(z * std + mean) - BIAS

    @staticmethod
    def fit(n = 20000, seed = 0):
        rng  = np.random.default_rng(seed)
        imgs = {
            'bkg' : read_random('background', n, rng),
            'sig' : read_random('signal', n, rng),
        }

        # the two draws are independent: synthetic mixtures
        imgs['syn'] = imgs['bkg'] + imgs['sig']

        stats = {}
        for (kind, x) in imgs.items():
            p = np.log(x.astype(np.float64) + BIAS)
            stats[kind] = (p.mean(), p.std())

        return Norm(stats)

    @staticmethod
    def load_or_fit(path, n = 20000, seed = 0):
        if os.path.exists(path):
            with open(path, 'r', encoding = 'utf-8') as f:
                return Norm(json.load(f))

        norm = Norm.fit(n, seed)
        os.makedirs(os.path.dirname(path), exist_ok = True)
        with open(path, 'w', encoding = 'utf-8') as f:
            json.dump(norm.stats, f, indent = 4)

        return norm

    def state(self, bkg, sig):
        """(N, H, W) energies -> (N, 2, H, W) standardised state."""
        return torch.stack((self.z(bkg, 'bkg'), self.z(sig, 'sig')), dim = 1)

    def components(self, x):
        """(N, 2, H, W) state -> background, signal energies."""
        return (self.energy(x[:, 0], 'bkg'), self.energy(x[:, 1], 'sig'))

def construct_net(method, channels = 96, res_blocks = 2, attn = (4,)):
    """The velocity (or regression) network, a compact ADM U-Net.

    24 x 64 is downsampled three times, to 3 x 8; attention runs at the 4x
    downsampled 6 x 16. condcfm and regress see the mixture as an extra
    input channel.
    """
    cond = 1 if method in ('condcfm', 'regress') else 0

    return UNetModel(
        image_size            = SHAPE[1],
        in_channels           = 2 + cond if method != 'regress' else 1,
        model_channels        = channels,
        out_channels          = 2,
        num_res_blocks        = res_blocks,
        attention_resolutions = tuple(attn),
        dropout               = 0.0,
        channel_mult          = (1, 2, 2, 2),
        num_heads             = 4,
        use_scale_shift_norm  = True,
    )

class GraphedSinkhorn:
    """Entropic OT plan between uniform marginals, min <P, C> - reg H(P).

    The log-domain Sinkhorn fixed point of POT's `sinkhorn_log`, in dual
    potentials, float32 on a cost shifted to start at zero. At the small
    regularisation of the SB-CFM convention it needs thousands of sweeps of
    tiny kernels, so blocks of sweeps are captured once as a CUDA graph and
    replayed until the L1 error of the row marginals is below `tol` (the
    column marginals are exact after every sweep). c.f. smoke_checks.py for
    the comparison with POT.
    """

    def __init__(self, n, reg, device, tol = 1e-3, block = 50,
                 max_blocks = 200):
        # pylint: disable=too-many-arguments
        self.n    = n
        self.reg  = reg
        self.tol  = tol
        self.max_blocks = max_blocks
        self.log_w = float(-np.log(n))

        self.kmat = torch.zeros((n, n), device = device)   # -C / reg
        self.f    = torch.zeros(n, device = device)         # phi / reg
        self.g    = torch.zeros(n, device = device)         # psi / reg

        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):
                self._sweeps(block)
        torch.cuda.current_stream(device).wait_stream(side)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._sweeps(block)

    def _sweeps(self, k):
        for _ in range(k):
            self.f.copy_(-torch.logsumexp(
                self.kmat + (self.g + self.log_w)[None, :], dim = 1
            ))
            self.g.copy_(-torch.logsumexp(
                self.kmat + (self.f + self.log_w)[:, None], dim = 0
            ))

    def log_plan(self):
        return self.kmat + self.f[:, None] + self.g[None, :] + 2 * self.log_w

    def __call__(self, cost):
        cost = cost.float()
        self.kmat.copy_(-(cost - cost.min()) / self.reg)
        self.f.zero_()
        self.g.zero_()

        err = None
        for _ in range(self.max_blocks):
            self.graph.replay()
            err = float((torch.logsumexp(self.log_plan(), dim = 1).exp()
                         - 1 / self.n).abs().sum())
            if err < self.tol:
                break

        return (self.log_plan().exp(), err)

class GPUSinkhornPlanSampler(OTPlanSampler):
    """TorchCFM's plan sampler with the entropic plan solved on the GPU.

    The library's numpy `ot.sinkhorn` underflows at the small regularisation
    of the SB-CFM convention (reg = 2 sigma^2 against costs of thousands)
    and silently falls back to the independent plan; POT's log-domain
    solver is correct but spends ~0.35 s a batch in kernel launches.
    `GraphedSinkhorn` solves the same problem (c.f. smoke_checks.py); pairs
    are drawn from the plan by the library's `sample_map`, as for the exact
    plan.
    """

    def __init__(self, reg, tol = 1e-3):
        super().__init__(method = 'sinkhorn', reg = reg)
        self.tol      = tol
        self.solver   = None
        self.last_err = None

    def get_map(self, x0, x1):
        x0 = x0.reshape(x0.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        assert x0.shape[0] == x1.shape[0]

        if (self.solver is None) or (self.solver.n != x0.shape[0]):
            self.solver = GraphedSinkhorn(
                x0.shape[0], self.reg, x0.device, tol = self.tol
            )

        (plan, self.last_err) = self.solver(torch.cdist(x0, x1) ** 2)
        return plan.double().cpu().numpy()

class Method:
    """Training pairs and loss of one method, and its decoding."""

    def __init__(self, name, norm, sigma = None):
        assert name in METHODS, name

        self.name = name
        self.norm = norm

        if name == 'otcfm':
            self.sigma   = 0.0 if sigma is None else sigma
            self.matcher = ExactOptimalTransportConditionalFlowMatcher(
                sigma = self.sigma
            )
        elif name == 'sbcfm':
            self.sigma   = 1.0 if sigma is None else sigma
            self.matcher = SchrodingerBridgeConditionalFlowMatcher(
                sigma = self.sigma, ot_method = 'sinkhorn'
            )
            self.matcher.ot_sampler = GPUSinkhornPlanSampler(
                reg = 2 * self.sigma**2
            )
        elif name == 'condcfm':
            self.sigma   = 0.0 if sigma is None else sigma
            self.matcher = ConditionalFlowMatcher(sigma = self.sigma)
        else:
            self.sigma   = None
            self.matcher = None

    @property
    def domains(self):
        return DOMAINS[self.name]

    @property
    def coupled(self):
        """Whether a minibatch plan is solved for every batch."""
        return self.name in ('otcfm', 'sbcfm')

    def source(self, mixture):
        """x0 of the augmented state: the mixture, all of it background."""
        return self.norm.state(mixture, torch.zeros_like(mixture))

    def condition(self, mixture):
        return self.norm.z(mixture, 'syn').unsqueeze(1)

    def endpoints(self, batch):
        """(x0, x1, cond) of a batch of energies, before any coupling."""
        (bkg, sig) = (batch['background'], batch['signal'])
        x1 = self.norm.state(bkg, sig)

        if self.name in ('otcfm', 'sbcfm'):
            return (self.source(batch['embed']), x1, None)

        cond = self.condition(bkg + sig)

        if self.name == 'condcfm':
            return (torch.randn_like(x1), x1, cond)

        return (None, x1, cond)

    def couple(self, x0, x1):
        """Minibatch plan, drawn as TorchCFM draws it (with replacement)."""
        return self.matcher.ot_sampler.sample_plan(x0, x1)

    def loss(self, net, x0, x1, cond):
        if self.name == 'regress':
            t = torch.zeros(x1.shape[0], device = x1.device)
            return torch.mean((net(t, cond) - x1)**2)

        # the plan is solved by the caller (so that it can be timed), the
        # matcher's own call only samples t and the interpolant
        (t, xt, ut) = ConditionalFlowMatcher.sample_location_and_conditional_flow(
            self.matcher, x0, x1
        )

        inp = xt if cond is None else torch.cat((xt, cond), dim = 1)
        return torch.mean((net(t, inp) - ut)**2)

@torch.no_grad()
def integrate(net, x0, cond, nfe, solver = 'midpoint'):
    """Fixed-step solve of dx/dt = v(t, x) from t = 0 to 1 in exactly `nfe`
    network evaluations."""
    x = x0

    def velocity(t, x):
        inp = x if cond is None else torch.cat((x, cond), dim = 1)
        return net(torch.full((x.shape[0],), t, device = x.device), inp)

    if solver == 'euler':
        h = 1.0 / nfe
        for k in range(nfe):
            x = x + h * velocity(k * h, x)

    elif solver == 'midpoint':
        assert nfe % 2 == 0, 'the midpoint rule takes two evaluations a step'
        steps = nfe // 2
        h     = 1.0 / steps

        for k in range(steps):
            t    = k * h
            xmid = x + 0.5 * h * velocity(t, x)
            x    = x + h * velocity(t + 0.5 * h, xmid)

    else:
        raise ValueError(f"unknown solver '{solver}'")

    return x

class Decomposer(torch.nn.Module):
    """Mixture energies (N, 1, H, W) -> (N, 2, H, W) (background, signal)
    energies, the interface `eval_val_truth.score_generator` expects of a
    generator when it is given no data norm.

    decode = 'direct'  : both channels as the flow ends them
             'mixture' : signal = mixture - background channel, i.e. the
                         additive mixing imposed at decoding
    samples > 1 (condcfm): the mean energy of that many samples.
    """

    def __init__(
        self, method, net, nfe = 16, solver = 'midpoint', decode = 'direct',
        samples = 1, seed = 0
    ):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.method  = method
        self.net     = net
        self.nfe     = nfe
        self.solver  = solver
        self.decode  = decode
        self.samples = samples
        self.seed    = seed
        self.gen     = None

    @torch.no_grad()
    def forward(self, mixture):
        m    = mixture[:, 0]
        norm = self.method.norm
        name = self.method.name

        if (self.gen is None) or (self.gen.device != m.device):
            self.gen = torch.Generator(device = m.device)
            self.gen.manual_seed(self.seed)

        if name == 'regress':
            t = torch.zeros(m.shape[0], device = m.device)
            x = self.net(t, self.method.condition(m))
            (bkg, sig) = norm.components(x)

        elif name == 'condcfm':
            cond = self.method.condition(m)
            (bkg, sig) = (0, 0)

            for _ in range(self.samples):
                x0 = torch.randn(
                    (m.shape[0], 2, *m.shape[1:]), device = m.device,
                    generator = self.gen
                )
                x  = integrate(self.net, x0, cond, self.nfe, self.solver)
                (b, s) = norm.components(x)
                (bkg, sig) = (bkg + b / self.samples, sig + s / self.samples)

        else:
            x = integrate(
                self.net, self.method.source(m), None, self.nfe, self.solver
            )
            (bkg, sig) = norm.components(x)

        if self.decode == 'mixture':
            sig = m - bkg

        return torch.stack((bkg, sig), dim = 1)

def count_params(net):
    return sum(p.numel() for p in net.parameters())

def save_checkpoint(path, **state):
    tmp = f'{path}.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)

def list_checkpoints(run_dir):
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return []

    return sorted(
        os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
            if f.startswith('step_') and f.endswith('.pt')
    )

def load_run(run_dir, ckpt, device, which = 'ema'):
    """(method, network) of a checkpoint, `which` of 'ema' and 'raw'."""
    with open(os.path.join(run_dir, 'config.json'), 'r', encoding = 'utf-8') as f:
        config = json.load(f)

    state = torch.load(ckpt, map_location = device, weights_only = False)
    norm  = Norm(state['norm'])
    meth  = Method(config['method'], norm, config.get('sigma'))

    net = construct_net(
        config['method'], config['channels'], config['res_blocks'],
        config['attn']
    ).to(device)
    net.load_state_dict(state[which])
    net.eval()

    return (meth, net, state, config)
