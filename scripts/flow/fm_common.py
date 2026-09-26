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
    regress_l1  the same, trained with the baseline's own `idt-aa` loss: L1 of
             the energies in GeV, background and signal weighted 1 : 10
             (a follow-up, c.f. FLOW_NOTES.md)
    otcfm1   one panel: real mixtures -> independently drawn HIJING events,
             exact minibatch OT; the jet is read as mixture - background
    otcfm_pieces  the otcfm set-up (mixture, empty) -> (HIJING, PYTHIA), but
             every batch holds the true pieces of its mixtures: the mixtures
             are made by adding the batch's HIJING and PYTHIA events, whose
             order is then shuffled, and the minibatch OT pairs them back
             (the share it gets right is logged). Synthetic mixtures, so
             the pairing is known information, as in UVCGAN-S's idt-aa.
    postflow the posterior sampler of conditional CFM with the physics built
             in: x1 = psi(s) alone, x0 ~ N(0, 1), conditioned on the synthetic
             mixture psi(b + s); every sample is clipped to 0 <= s <= m and
             the background is m - s, so the two always add up to the
             mixture. Optionally trained on randomised jet shapes
             (`JetShapes`, fm_train.py --augment jets).
    regress_mse  one network for the posterior mean: the signal share of
             every tower, s = sigmoid(net) m, trained by mean squared error
             in GeV -- the estimator a K-sample mean of a sampler converges
             to, in one evaluation; background m - s as for postflow
    jetflow  jet -> jet OT-CFM between two unpaired pools of jet-centred
             jets (16 x 16 canvases in GeV, closure_data.py): x0 = psi of a
             source jet, x1 = psi of an independently drawn target jet,
             exact minibatch OT. The matching cost is the squared L2 of the
             standardised states (`cost = 'l2'`, TorchCFM's own, the
             jet-centred cost of vac_med_coupling.py) or `ShapeEnergyCost`
             (`cost = 'shape_energy'`); both go through the same solver
             and pair sampling. `pairing = 'paired'` is the positive
             control: the target of each source jet J is its own T(J)
             (closure_data.Modification), no matching.

The log bias of psi can be changed (`Norm(bias = ...)`, fm_train.py
--log-bias); 0.1 is the baseline's.

Backbones (`construct_net(backbone = ...)`, fm_train.py --backbone): `unet`,
the TorchCFM ADM U-Net used throughout; `uvcgan`, the published UVCGAN-S
generator (ViT-ModNet, sPHENIX configuration) turned into a velocity
network by adding the time to its style token (`UVCGANVelocity`).

Nothing here reads the index files: the domains are drawn independently.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

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
JET_SHAPE = (16, 16)
METHODS   = [ 'otcfm', 'sbcfm', 'condcfm', 'regress', 'regress_l1',
              'otcfm1', 'otcfm_pieces', 'postflow', 'regress_mse',
              'jetflow' ]

# methods whose flow starts from the mixture and uses a minibatch OT plan
OT_METHODS = ('otcfm', 'sbcfm', 'otcfm1', 'otcfm_pieces', 'jetflow')

# posterior samplers: noise start, conditioned on the mixture, read as the
# mean of `samples` draws
SAMPLERS = ('condcfm', 'postflow')

# (nfe, solver, decode, samples) that checkpoints are selected, and time
# curves drawn, with -- all chosen on val (FLOW_NOTES.md): the regression
# takes one evaluation and no solver; the unpaired flows are read as
# mixture - background channel, their signal channel carries no event
# information, and OT-CFM with 4 Euler steps (3.68 GeV against 4.08 for the
# converged midpoint solve at the same checkpoint); conditional CFM is read
# as the mean energy of 4 posterior samples at 8 NFE -- one sample carries
# the posterior's spread (5.0 GeV against 3.9 for four), and 16 NFE are no
# better than 8
SELECTION = {
    'regress'    : (1, 'none', 'direct', 1),
    'regress_l1' : (1, 'none', 'direct', 1),
    'condcfm'    : (8, 'midpoint', 'direct', 4),
    'otcfm'      : (4, 'euler', 'mixture', 1),
    'sbcfm'      : (16, 'midpoint', 'mixture', 1),
    # set before training, as for otcfm; revisited on val afterwards
    'otcfm1'       : (4, 'euler', 'mixture', 1),
    'otcfm_pieces' : (16, 'midpoint', 'direct', 1),
    # as condcfm, set before training
    'postflow'     : (8, 'midpoint', 'direct', 4),
    'regress_mse'  : (1, 'none', 'direct', 1),
    # the OT-CFM inference setting, kept for the jet -> jet flow
    'jetflow'      : (4, 'euler', 'direct', 1),
}

# domains each method draws from; condcfm and regress build their mixtures
DOMAINS = {
    'otcfm'   : ('embed', 'background', 'signal'),
    'sbcfm'   : ('embed', 'background', 'signal'),
    'condcfm' : ('background', 'signal'),
    'regress' : ('background', 'signal'),
    'regress_l1' : ('background', 'signal'),
    'otcfm1'       : ('embed', 'background'),
    'otcfm_pieces' : ('background', 'signal'),
    'postflow'     : ('background', 'signal'),
    'regress_mse'  : ('background', 'signal'),
    'jetflow'      : ('closure_src', 'closure_tgt'),
}

# weights of the background and signal L1 terms of regress_l1: those of the
# baseline's idt-aa loss (lambda_cyc_a0 10, lambda_cyc_a1 100)
L1_WEIGHTS = (1.0, 10.0)

def is_regression(name):
    return name.startswith('regress')

def image_shape(method):
    return JET_SHAPE if method == 'jetflow' else SHAPE

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
    """psi(E) = log(E + bias), standardised per kind of image.

    Kinds: `bkg` (background domain), `sig` (signal domain), `syn`
    (synthetic mixtures b + s). Fitted on training events only. The bias
    travels with the statistics (key `_bias`; absent means 0.1).
    """

    KINDS = ('bkg', 'sig', 'syn')

    def __init__(self, stats, bias = BIAS):
        stats = dict(stats)
        self.bias  = float(stats.pop('_bias', bias))
        self.stats = {
            k : (float(v[0]), float(v[1])) for (k, v) in stats.items()
        }

    def to_dict(self):
        return { **self.stats, '_bias' : self.bias }

    def psi(self, energy):
        return torch.log(energy + self.bias)

    def z(self, energy, kind):
        (mean, std) = self.stats[kind]
        return (self.psi(energy) - mean) / std

    def energy(self, z, kind):
        (mean, std) = self.stats[kind]
        return torch.exp(z * std + mean) - self.bias

    @staticmethod
    def fit(n = 20000, seed = 0, bias = BIAS):
        rng  = np.random.default_rng(seed)
        imgs = {
            'bkg' : read_random('background', n, rng),
            'sig' : read_random('signal', n, rng),
        }

        # the two draws are independent: synthetic mixtures
        imgs['syn'] = imgs['bkg'] + imgs['sig']

        stats = {}
        for (kind, x) in imgs.items():
            p = np.log(x.astype(np.float64) + bias)
            stats[kind] = (p.mean(), p.std())

        return Norm(stats, bias)

    @staticmethod
    def load_or_fit(path, n = 20000, seed = 0, bias = BIAS):
        if os.path.exists(path):
            with open(path, 'r', encoding = 'utf-8') as f:
                return Norm(json.load(f), bias)

        norm = Norm.fit(n, seed, bias)
        os.makedirs(os.path.dirname(path), exist_ok = True)
        with open(path, 'w', encoding = 'utf-8') as f:
            json.dump(norm.to_dict(), f, indent = 4)

        return norm

    @staticmethod
    def path(bias = BIAS, method = None):
        if method == 'jetflow':
            # fitted by closure_data.py on its training canvases
            return os.path.join(out_root(), 'norm_closure.json')
        name = 'norm_n20000_seed0.json' if bias == BIAS \
            else f'norm_n20000_seed0_bias{bias:g}.json'
        return os.path.join(out_root(), name)

    def state(self, bkg, sig):
        """(N, H, W) energies -> (N, 2, H, W) standardised state."""
        return torch.stack((self.z(bkg, 'bkg'), self.z(sig, 'sig')), dim = 1)

    def components(self, x):
        """(N, 2, H, W) state -> background, signal energies."""
        return (self.energy(x[:, 0], 'bkg'), self.energy(x[:, 1], 'sig'))

# generator of the published UVCGAN-S model (its config.json, `model_args`)
UVCGAN_GENERATOR = {
    'features' : 384, 'n_heads' : 6, 'n_blocks' : 12, 'ffn_features' : 1536,
    'embed_features' : 384, 'activ' : 'gelu', 'norm' : 'layer',
    'modnet_features_list' : [ 96, 192, 384 ], 'modnet_activ' : 'leakyrelu',
    'modnet_norm' : None, 'modnet_downsample' : 'conv',
    'modnet_upsample' : 'upsample-conv', 'modnet_rezero' : False,
    'rezero' : True, 'activ_output' : None, 'style_rezero' : True,
    'style_bias' : True, 'n_ext' : 1,
}

class UVCGANVelocity(torch.nn.Module):
    """The UVCGAN-S generator as a velocity network v(t, x).

    ViT-ModNet: a convolutional encoder (96, 192, 384 features, no
    normalisation) down to 3 x 8, a pixel-wise transformer over those 24
    positions plus one extra token (12 blocks, 384 features), and a decoder
    of modulated, demodulated convolutions whose style is the extra token's
    output, with skip connections. The only change: a sinusoidal embedding of
    t, through a two-layer MLP, is added to the extra token's input, so the
    style that already modulates every decoder layer (and, through
    attention, the bottleneck) knows the time. Weights initialised as
    UVCGAN-S initialises them (Kaiming); the time MLP by PyTorch's default.
    """

    TIME_FEATURES = 128

    def __init__(self, c_in, c_out, shape = SHAPE, **kwargs):
        # pylint: disable=import-outside-toplevel
        super().__init__()
        from uvcgan_s.base.weight_init import init_weights
        from uvcgan_s.models.generator.vitmodnet import ViTModNetGenerator
        from uvcgan_s.torch.layers.transformer import (
            ExtendedPixelwiseViT, img_to_pixelwise_tokens,
            img_from_pixelwise_tokens
        )

        args = { **UVCGAN_GENERATOR, **kwargs }
        self.gen = ViTModNetGenerator(
            input_shape = (c_in, *shape), output_shape = (c_out, *shape),
            **args
        )

        class TimedPixelwiseViT(ExtendedPixelwiseViT):
            """ExtendedPixelwiseViT with `time`, (N, n_ext * features), added
            to its extra tokens."""
            time = None

            def forward(self, x):
                itokens = img_to_pixelwise_tokens(x)
                (n, length, _) = itokens.shape
                extra = self.extra_tokens.tile(n, 1, 1) \
                    + self.time.view(n, *self.extra_tokens.shape[1:])
                y = self.trans_input(itokens)
                y = self.encoder(torch.cat([ y, extra ], dim = 1))
                otokens = self.trans_output(y[:, :length, :])
                return (img_from_pixelwise_tokens(otokens, self.image_shape),
                        y[:, length:, :].reshape(n, -1))

        # replaces the generator's own (time-blind) bottleneck
        self.gen.net.set_bottleneck(TimedPixelwiseViT(
            args['features'], args['n_heads'], args['n_blocks'],
            args['ffn_features'], args['embed_features'], args['activ'],
            args['norm'], image_shape = self.gen.net.get_inner_shape(),
            rezero = args['rezero'], n_ext = args['n_ext'],
        ))
        init_weights(self.gen, { 'name' : 'kaiming' })

        width = args['features'] * args['n_ext']
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.TIME_FEATURES, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )

    def forward(self, t, x):
        # pylint: disable=import-outside-toplevel
        from torchcfm.models.unet.nn import timestep_embedding

        t = t.reshape(-1).expand(x.shape[0]) if t.numel() == 1 \
            else t.reshape(-1)
        self.gen.net.get_bottleneck().time = self.time_embed(
            timestep_embedding(t, self.TIME_FEATURES)
        )
        return self.gen(x)

def construct_net(method, channels = 96, res_blocks = 2, attn = (4,),
                  backbone = 'unet'):
    """The velocity (or regression) network: a compact ADM U-Net, or the
    UVCGAN-S generator (`backbone = 'uvcgan'`, c.f. `UVCGANVelocity`).

    24 x 64 is downsampled three times, to 3 x 8; attention runs at the 4x
    downsampled 6 x 16. condcfm and regress see the mixture as an extra
    input channel.
    """
    cond = 1 if (method == 'condcfm') or is_regression(method) else 0
    if method == 'regress_mse':
        (c_in, c_out) = (1, 1)
    elif is_regression(method):
        (c_in, c_out) = (1, 2)
    elif method in ('otcfm1', 'jetflow'):
        (c_in, c_out) = (1, 1)
    elif method == 'postflow':
        (c_in, c_out) = (2, 1)          # signal state and the mixture
    else:
        (c_in, c_out) = (2 + cond, 2)

    shape = image_shape(method)
    if backbone == 'uvcgan':
        return UVCGANVelocity(c_in, c_out, shape)
    assert backbone == 'unet', backbone

    return UNetModel(
        image_size            = shape[1],
        in_channels           = c_in,
        model_channels        = channels,
        out_channels          = c_out,
        num_res_blocks        = res_blocks,
        attention_resolutions = tuple(attn),
        dropout               = 0.0,
        channel_mult          = (1, 2, 2, 2),
        num_heads             = 4,
        use_scale_shift_norm  = True,
    )

class JetShapes:
    """Randomised jet shapes: a broader signal prior than PYTHIA's.

    With probability `p` a signal image (GeV, (N, H, W)) is replaced by a
    transformed one, the strengths drawn per event:

        hardness  s -> s^g sum(s) / sum(s^g), log g ~ U(hardness): harder
                  (g > 1) or softer fragmentation at the same energy
        spread    s -> (1 - l) s + l (K * s), l ~ U(0, spread), K the mean of
                  the 8 neighbouring towers (periodic in phi; the eta edges
                  lose their share): each tower gives a share l of its
                  energy to its neighbours, wider jets
        jitter    s_i -> s_i exp(sigma e_i - sigma^2 / 2), e_i ~ N(0, 1),
                  sigma ~ U(0, jitter): tower-level fluctuations
        scale     s -> a s, log a ~ U(-scale, scale): energy lost or gained

    The synthetic mixture is made from the transformed signal, so the
    training pairs stay exact. The hardness range is centred so that the
    transformed PYTHIA signals keep, on average, the leading-tower share
    and p_T^D of the originals (spreading alone softens them); set on
    training signals before any training, not tuned on JEWEL
    (FLOW_NOTES.md).
    """

    def __init__(self, p = 0.5, hardness = (-0.38, 0.52), spread = 0.4,
                 jitter = 0.3, scale = 0.25):
        # pylint: disable=too-many-arguments
        self.p        = p
        self.hardness = hardness
        self.spread   = spread
        self.jitter   = jitter
        self.scale    = scale
        self.kernel   = None

    def neighbours(self, x):
        if (self.kernel is None) or (self.kernel.device != x.device):
            self.kernel = torch.full((1, 1, 3, 3), 1 / 8, device = x.device)
            self.kernel[0, 0, 1, 1] = 0

        x = F.pad(x.unsqueeze(1), (1, 1, 0, 0), mode = 'circular')
        x = F.pad(x, (0, 0, 1, 1))
        return F.conv2d(x, self.kernel)[:, 0]

    @torch.no_grad()
    def __call__(self, sig):
        n = sig.shape[0]

        def uniform(lo, hi):
            return torch.empty((n, 1, 1), device = sig.device).uniform_(lo, hi)

        x = sig.clamp(min = 0)

        g     = torch.exp(uniform(*self.hardness))
        total = x.sum((1, 2), keepdim = True)
        xg    = x ** g
        x     = xg * total / xg.sum((1, 2), keepdim = True).clamp(min = 1e-12)

        share = uniform(0, self.spread)
        x     = (1 - share) * x + share * self.neighbours(x)

        sigma = uniform(0, self.jitter)
        x     = x * torch.exp(sigma * torch.randn_like(x) - sigma**2 / 2)

        x     = x * torch.exp(uniform(-self.scale, self.scale))

        keep  = torch.rand((n, 1, 1), device = sig.device) >= self.p
        return torch.where(keep, sig, x)

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

class ShapeEnergyCost:
    """Matching cost of jet-centred jets that separates shape and energy.

    For a jet J (canvas, GeV, >= 0): E = sum J, Q = J / E. Then

        D_s(i, j) = |P_s Q_i - P_s Q'_j|^2   P_s: sum pooling over s x s
                                             blocks of the 16 x 16 canvas
        D_E(i, j) = [log((E_i + eps) / (E'_j + eps))]^2
        C(i, j)   = mean_s D_s / a_s + lambda D_E / a_E,   s in 1, 2, 4

    a_s, a_E: medians of the positive pairwise distances between a fixed
    training-only subset of source and target jets (`calibrate`), frozen in a
    JSON file; a scale whose median is not positive is dropped and reported.
    Only the shape features are normalised to unit energy; the flow's
    endpoints keep their physical amplitudes.
    """

    def __init__(self, scales = (1, 2, 4), lam = 1.0, eps = 0.1,
                 a_s = None, a_e = None):
        # pylint: disable=too-many-arguments
        self.scales = tuple(scales)
        self.lam    = float(lam)
        self.eps    = float(eps)
        self.a_s    = a_s
        self.a_e    = a_e

    def features(self, jets):
        jets   = jets.clamp(min = 0).float()
        energy = jets.sum((-2, -1))
        shape  = jets / energy.clamp(min = 1e-12)[:, None, None]
        pooled = [
            F.avg_pool2d(shape.unsqueeze(1), s).flatten(1) * s * s
                for s in self.scales
        ]
        return (pooled, torch.log(energy + self.eps))

    def terms(self, src, tgt):
        """[D_s for each scale], D_E, (N_src, N_tgt) each."""
        (ps, ls) = self.features(src)
        (pt, lt) = self.features(tgt)
        d_s = [ torch.cdist(a, b)**2 for (a, b) in zip(ps, pt) ]
        d_e = (ls[:, None] - lt[None, :])**2
        return (d_s, d_e)

    def __call__(self, src, tgt):
        (d_s, d_e) = self.terms(src, tgt)
        used  = [ (d, a) for (d, a) in zip(d_s, self.a_s) if a is not None ]
        shape = sum(d / a for (d, a) in used) / len(used)
        return shape + self.lam * d_e / self.a_e

    def calibrate(self, src, tgt):
        (d_s, d_e) = self.terms(src, tgt)

        def median_positive(d):
            d = d.flatten()
            d = d[d > 0]
            return float(d.median()) if len(d) else None

        self.a_s = [ median_positive(d) for d in d_s ]
        self.a_s = [ a if (a is not None) and (a > 1e-12) else None
                     for a in self.a_s ]
        self.a_e = median_positive(d_e)
        return { 'scales' : list(self.scales), 'a_s' : self.a_s,
                 'a_E' : self.a_e, 'eps' : self.eps,
                 'dropped_scales' : [ s for (s, a) in zip(self.scales, self.a_s)
                                      if a is None ] }

    @staticmethod
    def calib_path():
        return os.path.join(out_root(), 'closure_cost_calib.json')

    @staticmethod
    def load(lam, path = None):
        with open(path or ShapeEnergyCost.calib_path(), 'r',
                  encoding = 'utf-8') as f:
            c = json.load(f)
        return ShapeEnergyCost(c['scales'], lam, c['eps'], c['a_s'], c['a_E'])

class Method:
    """Training pairs and loss of one method, and its decoding."""

    def __init__(self, name, norm, sigma = None, augment = 'none',
                 cost = 'l2', cost_lambda = 1.0, pairing = 'unpaired',
                 target = 'closure_tgt'):
        # pylint: disable=too-many-arguments
        assert name in METHODS, name
        assert augment in ('none', 'jets'), augment
        assert cost in ('l2', 'shape_energy'), cost
        assert pairing in ('unpaired', 'paired'), pairing
        self.pairing = pairing
        self.modify  = None
        # jetflow's target pool: closure_tgt = T(B); closure_null_tgt = B
        # itself (the null test, closure_data.py --null)
        self.target  = target

        self.name    = name
        self.norm    = norm
        self.augment = JetShapes() if augment == 'jets' else None
        self.cost    = cost
        self.cost_fn = ShapeEnergyCost.load(cost_lambda) \
            if cost == 'shape_energy' else None
        self.raw     = None

        # otcfm_pieces: share of mixtures the plan pairs with their own pieces
        self.recovery = []

        if name in ('otcfm', 'otcfm1', 'otcfm_pieces', 'jetflow'):
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
        elif name in SAMPLERS:
            self.sigma   = 0.0 if sigma is None else sigma
            self.matcher = ConditionalFlowMatcher(sigma = self.sigma)
        else:
            self.sigma   = None
            self.matcher = None

    @property
    def domains(self):
        if self.pairing == 'paired':
            return DOMAINS[self.name][:1]      # targets are made from them
        if self.name == 'jetflow':
            return (DOMAINS[self.name][0], self.target)
        return DOMAINS[self.name]

    @property
    def coupled(self):
        """Whether a minibatch plan is solved for every batch."""
        return (self.name in OT_METHODS) and (self.pairing == 'unpaired')

    def source(self, mixture):
        """x0 of the flow: the mixture, all of it background (one panel for
        otcfm1, with an empty signal panel otherwise)."""
        if self.name == 'otcfm1':
            return self.norm.z(mixture, 'bkg').unsqueeze(1)
        if self.name == 'jetflow':
            return self.norm.z(mixture, 'jet').unsqueeze(1)
        return self.norm.state(mixture, torch.zeros_like(mixture))

    def condition(self, mixture):
        return self.norm.z(mixture, 'syn').unsqueeze(1)

    def endpoints(self, batch):
        """(x0, x1, cond) of a batch of energies, before any coupling."""
        if self.name == 'jetflow':
            src = batch['closure_src']
            if self.pairing == 'paired':
                if self.modify is None:
                    # pylint: disable=import-outside-toplevel
                    from closure_data import Modification
                    self.modify = Modification(src.device)
                return (self.source(src), self.source(self.modify(src)), None)
            # the energies are kept for a cost computed on them
            self.raw = (src, batch[self.target])
            return (self.source(src), self.source(batch[self.target]), None)

        bkg = batch['background']

        if self.name == 'otcfm1':
            return (self.source(batch['embed']),
                    self.norm.z(bkg, 'bkg').unsqueeze(1), None)

        sig = batch['signal']
        if self.augment is not None:
            sig = self.augment(sig)

        if self.name == 'postflow':
            x1 = self.norm.z(sig, 'sig').unsqueeze(1)
            return (torch.randn_like(x1), x1, self.condition(bkg + sig))

        if self.name == 'regress_mse':
            # x0 carries the mixture energy, x1 the signal energy
            return (bkg + sig, sig, self.condition(bkg + sig))

        x1  = self.norm.state(bkg, sig)

        if self.name in ('otcfm', 'sbcfm'):
            return (self.source(batch['embed']), x1, None)

        if self.name == 'otcfm_pieces':
            # the mixtures of the batch are made of its own pieces, whose
            # order is then hidden
            self.perm = torch.randperm(len(bkg), device = bkg.device)
            return (self.source(bkg + sig), x1[self.perm], None)

        cond = self.condition(bkg + sig)

        if self.name == 'condcfm':
            return (torch.randn_like(x1), x1, cond)

        return (None, x1, cond)

    def couple(self, x0, x1):
        """Minibatch plan, drawn as TorchCFM draws it (with replacement)."""
        sampler = self.matcher.ot_sampler

        if self.cost_fn is not None:
            # the same solver and pair sampling as sample_plan, with this
            # cost matrix in place of the squared L2 of the states
            cost = self.cost_fn(*self.raw).double().cpu().numpy()
            plan = sampler.ot_fn(pot.unif(len(x0)), pot.unif(len(x1)), cost)
            if (not np.all(np.isfinite(plan))) or (abs(plan.sum()) < 1e-8):
                raise RuntimeError('the OT plan of the shape-energy cost failed')
            (i, j) = sampler.sample_map(plan, x0.shape[0])
            return (x0[i], x1[j])

        if self.name != 'otcfm_pieces':
            return sampler.sample_plan(x0, x1)

        # as sample_plan, keeping the plan to see how often it pairs a
        # mixture with its own pieces (x1[k] is the piece of mixture perm[k])
        plan = sampler.get_map(x0, x1)
        best = torch.from_numpy(plan.argmax(axis = 1)).to(self.perm.device)
        self.recovery.append(float(
            (self.perm[best] == torch.arange(len(best), device = best.device))
            .float().mean()
        ))
        (i, j) = sampler.sample_map(plan, x0.shape[0])
        return (x0[i], x1[j])

    def pop_recovery(self):
        """Mean share of correct pairs since the last call (None if none)."""
        if not self.recovery:
            return None
        value = float(np.mean(self.recovery))
        self.recovery = []
        return value

    def loss(self, net, x0, x1, cond):
        if self.name == 'regress_mse':
            t     = torch.zeros(x1.shape[0], device = x1.device)
            share = torch.sigmoid(net(t, cond)[:, 0])
            return torch.mean((share * x0 - x1)**2)

        if self.name == 'regress':
            t = torch.zeros(x1.shape[0], device = x1.device)
            return torch.mean((net(t, cond) - x1)**2)

        if self.name == 'regress_l1':
            t = torch.zeros(x1.shape[0], device = x1.device)
            (bkg, sig)     = self.norm.components(net(t, cond))
            (bkg_t, sig_t) = self.norm.components(x1)
            return (
                  L1_WEIGHTS[0] * torch.mean((bkg - bkg_t).abs())
                + L1_WEIGHTS[1] * torch.mean((sig - sig_t).abs())
            )

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
    samples > 1 (condcfm, postflow): the mean energy of that many samples.
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

        if name == 'regress_mse':
            t   = torch.zeros(m.shape[0], device = m.device)
            sig = torch.sigmoid(
                self.net(t, self.method.condition(m))[:, 0]
            ) * m.clamp(min = 0)
            bkg = m - sig

        elif is_regression(name):
            t = torch.zeros(m.shape[0], device = m.device)
            x = self.net(t, self.method.condition(m))
            (bkg, sig) = norm.components(x)

        elif name == 'postflow':
            # every sample is a decomposition that adds up: 0 <= s <= m
            cond = self.method.condition(m)
            top  = m.clamp(min = 0)
            sig  = 0

            for _ in range(self.samples):
                x0 = torch.randn(
                    (m.shape[0], 1, *m.shape[1:]), device = m.device,
                    generator = self.gen
                )
                x  = integrate(self.net, x0, cond, self.nfe, self.solver)
                s  = torch.minimum(norm.energy(x[:, 0], 'sig').clamp(min = 0), top)
                sig = sig + s / self.samples

            bkg = m - sig

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

        elif name == 'otcfm1':
            x   = integrate(
                self.net, self.method.source(m), None, self.nfe, self.solver
            )
            bkg = norm.energy(x[:, 0], 'bkg')
            sig = m - bkg           # the only reading of a one-panel flow

        else:
            x = integrate(
                self.net, self.method.source(m), None, self.nfe, self.solver
            )
            (bkg, sig) = norm.components(x)

        if self.decode == 'mixture':
            sig = m - bkg

        return torch.stack((bkg, sig), dim = 1)

class JetMapper(torch.nn.Module):
    """Source jets (N, H, W), GeV -> the flow's jets F(J), GeV; clipped at 0
    unless `clip = False` (the raw inverse of the normalisation, >= -bias)."""

    def __init__(self, method, net, nfe = 4, solver = 'euler', clip = True):
        # pylint: disable=too-many-arguments
        super().__init__()
        self.method = method
        self.net    = net
        self.nfe    = nfe
        self.solver = solver
        self.clip   = clip

    @torch.no_grad()
    def forward(self, jets):
        x = integrate(self.net, self.method.source(jets), None, self.nfe,
                      self.solver)
        out = self.method.norm.energy(x[:, 0], 'jet')
        return out.clamp(min = 0) if self.clip else out

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
    meth.cost = config.get('cost', 'l2')      # training only; recorded

    net = construct_net(
        config['method'], config['channels'], config['res_blocks'],
        config['attn'], config.get('backbone', 'unet')
    ).to(device)
    net.load_state_dict(state[which])
    net.eval()

    return (meth, net, state, config)
