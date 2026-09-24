#!/usr/bin/env python
"""Score the signal extraction of sPHENIX models against truth.

The val split holds only mixed (`embed`) events, but each of them is an
event of the PYTHIA production in `train/signal.h5` embedded into HIJING:
the index files name the same (file, event), and the mixed image contains
the signal image tower by tower (no noise). So every val event comes with
its truth, which the training losses never see:

    truth signal      S = signal.h5[matching event]
    truth background  B = embed - S

For each checkpoint the generator `gen_ba` (its moving average `ema`, the
network used for inference, and/or the `raw` one) decomposes the val
events, and the result is scored on

    l1_sig, l1_bkg  mean |extracted - truth| per tower, GeV: the held-out
                    counterparts of `idt_aa_a1` and `idt_aa_a0`
    jet energy      energy in a R = 0.4 cone around the leading truth jet,
                    extracted against truth: scale `jes` (mean ratio),
                    `bias` (mean difference, GeV) and resolution `jer`
                    (standard deviation of the difference, GeV; `jer_iqr`
                    is the same from the interquartile range, which
                    ignores the tails; `jer_cal` is the spread around a
                    linear fit of the extracted against the true energy,
                    divided by its slope: the resolution after an offset
                    and scale calibration). Only events whose leading jet
                    lies inside |eta| < 0.7 and holds at least --min-jet
                    enter these.

A median-rho subtraction (per eta row, the median over phi of the mixed
event) is scored alongside as a reference point for the jet numbers. It is
a naive baseline, not the iterative subtraction sPHENIX uses.

NOTE: the truth signals of the val events are also training samples of the
signal domain, shown to the model unpaired among 2.6M others: the mixed
events and the pairing are held out, the signal images are not.

With `--truth jewel` the same scores are computed on the test split, JEWEL
jets embedded into HIJING, against the JEWEL jets without background of
Zenodo record 17594612 (`jewel_jet30.tar.gz`, unpacked under
DATA/sphenix/jewel_jet30/), matched by (file, event) the same way. These
jets are quenched and never seen in training: an out-of-distribution test.

    eval_val_truth.py MODEL_DIR [MODEL_DIR ...] [--epochs 50,100] [--nets ema]
                      [--truth val|jewel]

Results are appended to MODEL_DIR/evals/{val,jewel}_truth.csv, one row per
(epoch, net); rows already present are skipped, so rerunning the script
only evaluates new checkpoints.
"""

import argparse
import glob
import os
import re
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from uvcgan_s.config            import Args
from uvcgan_s.cgan              import construct_model
from uvcgan_s.torch.distributed import unwrap_model

DATA_PATH  = 'sphenix/2025-06-05_jet_bkg_sub'
PYTHIA_IDX = 'type11_run19_jet30_pythia_noNoise_allCentrality.h5_index.csv'
EMBED_IDX  = 'type11plus4_run19_jet30_hijing_noNoise_cent0.h5_index.csv'
TEST_IDX   = 'type4_hijing_plus_jewel_jet30_40_50_noNoise_cent0.h5_index.csv'
JEWEL_DIR  = 'sphenix/jewel_jet30/jewel_jet30'

# split and index file of the mixed events of each truth set
TRUTH_SETS = {
    'val'   : ('val',  EMBED_IDX),
    'jewel' : ('test', TEST_IDX),
}

# 24 x 64 towers over |eta| < 1.1 and the full azimuth
DETA  = 2.2 / 24
DPHI  = 2 * np.pi / 64
R_JET = 0.4

# networks that decompose the mixed events
NETS = {
    'ema' : 'ema_gen_ba',
    'raw' : 'gen_ba',
}

COLUMNS = [
    'label', 'epoch', 'updates', 'net', 'n_events', 'n_jets',
    'l1_sig', 'l1_bkg', 'jes', 'bias', 'jer', 'jer_iqr', 'jer_cal',
]

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Score against val truth')
    parser.add_argument('models', nargs = '*', help = 'model directories')
    parser.add_argument('--epochs', default = 'all',
        help = "comma separated checkpoint epochs, 'all' or 'final'")
    parser.add_argument('--nets', default = 'ema,raw',
        help = "comma separated subset of 'ema' and 'raw'")
    parser.add_argument('--n-events', type = int, default = 20000)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--batch', type = int, default = 500)
    parser.add_argument('--min-jet', type = float, default = 10.0,
        help = 'minimal truth cone energy (GeV) of the jet metrics')
    parser.add_argument('--truth', default = 'val',
        choices = list(TRUTH_SETS),
        help = 'PYTHIA val events, or JEWEL test events')
    parser.add_argument('--data', default = os.environ.get(
        'UVCGAN_S_DATA', 'data'))
    parser.add_argument('--cache', default = None,
        help = 'paired events, built on first use')
    parser.add_argument('--force', action = 'store_true',
        help = 'evaluate rows that are already present')
    return parser.parse_args()

def read_keys(path):
    """(file, event) of every sample of an index file, as one integer."""
    df = pd.read_csv(path, usecols = [ 'sample', 'hist' ])
    m  = df['hist'].str.extract(r'_file(\d+)_evt(\d+)').astype(np.int64)

    assert (df['sample'].values == np.arange(len(df))).all()
    return m[0].values * 1_000_000 + m[1].values

def read_pythia_truth(root, keys):
    """PYTHIA signals of the (file, event) keys, from train/signal.h5."""
    pythia_keys = read_keys(os.path.join(root, 'train', PYTHIA_IDX))

    order = np.argsort(pythia_keys)
    pos   = np.searchsorted(pythia_keys, keys, sorter = order)
    match = order[np.minimum(pos, len(order) - 1)]

    if not (pythia_keys[match] == keys).all():
        raise RuntimeError('some events have no PYTHIA counterpart')

    # h5py wants increasing indices
    sig_order = np.argsort(match)

    with h5py.File(os.path.join(root, 'train', 'signal.h5'), 'r') as f:
        signal = np.empty((len(keys), *f['data'].shape[1:]), np.float32)
        signal[sig_order] = f['data'][match[sig_order]]

    return signal

def read_jewel_truth(jewel_dir, keys, shape):
    """JEWEL jets of the (file, event) keys, from their ROOT files."""
    # pylint: disable=import-outside-toplevel
    import uproot

    signal = np.empty((len(keys), *shape), np.float32)
    files  = keys // 1_000_000
    events = keys %  1_000_000

    for fileno in np.unique(files):
        path = os.path.join(jewel_dir, f'jewel_jet30_file{fileno}.root')

        with uproot.open(path) as f:
            for i in np.nonzero(files == fileno)[0]:
                name = f'h_eta_phi_cent0_file{fileno}_evt{events[i]}'
                signal[i] = f[name].values()

    return signal

def build_pairs(truth, data, n_events, seed, path):
    """Pick mixed events and read them with their signal truth."""
    root = os.path.join(data, DATA_PATH)
    (split, index) = TRUTH_SETS[truth]

    embed_keys = read_keys(os.path.join(root, split, index))

    rng  = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(embed_keys), n_events, replace = False))

    with h5py.File(os.path.join(root, split, 'embed.h5'), 'r') as f:
        embed = f['data'][pick]

    if truth == 'val':
        signal = read_pythia_truth(root, embed_keys[pick])
    else:
        signal = read_jewel_truth(
            os.path.join(data, JEWEL_DIR), embed_keys[pick], embed.shape[1:]
        )

    # a true pair has embed >= signal in every tower; a wrong match
    # misses the signal's jet
    deficit = (signal - embed.astype(np.float32)).max(axis = (1, 2))
    n_bad   = int((deficit > 0.05).sum())

    if n_bad > 0:
        raise RuntimeError(
            f'{n_bad} {truth} events do not contain their signal'
        )

    os.makedirs(os.path.dirname(path), exist_ok = True)
    np.savez(
        path, index = pick, keys = embed_keys[pick],
        embed = embed, signal = signal
    )

def load_pairs(data, n_events, seed, cache = None, truth = 'val'):
    """Paired mixed events (embed, signal), built on first use."""
    name = f'pairs_n{n_events}_seed{seed}.npz'
    if truth != 'val':
        name = f'pairs_{truth}_n{n_events}_seed{seed}.npz'

    path = cache or os.path.join(
        os.environ.get('UVCGAN_S_OUTDIR', 'outdir'), 'sphenix', 'val_truth',
        name
    )

    if not os.path.exists(path):
        t0  = time.time()
        tmp = f'{path}.{os.getpid()}.npz'

        build_pairs(truth, data, n_events, seed, tmp)
        os.replace(tmp, path)

        print(f"built '{path}' in {time.time() - t0:.0f} s")

    with np.load(path) as f:
        return (f['embed'], f['signal'])

def cone_kernel(radius):
    ni = int(radius / DETA)
    nj = int(radius / DPHI)

    deta = np.arange(-ni, ni + 1)[:, None] * DETA
    dphi = np.arange(-nj, nj + 1)[None, :] * DPHI

    return torch.tensor(
        (deta**2 + dphi**2 <= radius**2 + 1e-9), dtype = torch.float32
    )

def cone_sums(x, kernel):
    """Energy in the cone around every tower: (N, H, W) -> (N, H, W).

    The azimuth wraps around, the pseudorapidity does not.
    """
    (kh, kw) = kernel.shape

    x = F.pad(x.unsqueeze(1), (kw // 2, kw // 2, 0, 0), mode = 'circular')
    x = F.pad(x, (0, 0, kh // 2, kh // 2))

    return F.conv2d(x, kernel[None, None].to(x.device)).squeeze(1)

class Truth:
    """What the scores need of the truth, computed once."""

    def __init__(self, embed, signal, kernel, min_jet, device):
        self.embed  = embed
        self.signal = signal
        self.kernel = kernel

        s = torch.from_numpy(signal).to(device, torch.float32)

        # leading jet: the cone of most truth energy; the jet scores only
        # use events where it lies entirely inside the acceptance
        sums = cone_sums(s, kernel)
        flat = sums.flatten(1).argmax(dim = 1)

        self.row = flat // sums.shape[2]
        self.col = flat %  sums.shape[2]

        margin = kernel.shape[0] // 2
        inside = (self.row >= margin) & (self.row < sums.shape[1] - margin)

        self.e_true = self.at_axis(s)
        self.jets   = inside & (self.e_true >= min_jet)

    def at_axis(self, x, start = 0):
        """Cone energy of the events `x` = start, start + 1, ... around
        their jet axes."""
        n    = x.shape[0]
        rows = self.row[start:start + n]
        cols = self.col[start:start + n]

        return cone_sums(x, self.kernel)[
            torch.arange(n, device = x.device), rows, cols
        ]

def jet_scores(e_fake, truth):
    """Scale and resolution of the cone energy of the selected jets."""
    e_fake = e_fake[truth.jets]
    e_true = truth.e_true[truth.jets]
    diff   = (e_fake - e_true).double()

    (q25, q75) = torch.quantile(diff, torch.tensor(
        [ 0.25, 0.75 ], dtype = diff.dtype, device = diff.device
    ))

    # calibrated resolution: the spread around the linear response
    # e_fake = a + b * e_true, in units of e_true
    x = e_true.double()
    y = e_fake.double()
    b = ((x - x.mean()) * (y - y.mean())).mean() / x.var(unbiased = False)
    a = y.mean() - b * x.mean()

    return {
        'n_jets'  : int(truth.jets.sum()),
        'jes'     : float((y / x).mean()),
        'bias'    : float(diff.mean()),
        'jer'     : float(diff.std()),
        'jer_iqr' : float((q75 - q25) / 1.349),
        'jer_cal' : float((y - a - b * x).std() / b),
    }

@torch.no_grad()
def score_generator(gen, data_norm, truth, batch, device):
    n      = len(truth.embed)
    l1_sig = 0.0
    l1_bkg = 0.0
    e_fake = []

    for start in range(0, n, batch):
        e = torch.from_numpy(truth.embed [start:start + batch])
        s = torch.from_numpy(truth.signal[start:start + batch])

        e = e.to(device, torch.float32)
        s = s.to(device, torch.float32)

        x = e.unsqueeze(1)
        if data_norm is not None:
            x = data_norm.normalize(x)

        y = gen(x)

        if data_norm is not None:
            y = data_norm.denormalize(y)

        (fake_bkg, fake_sig) = (y[:, 0], y[:, 1])

        l1_sig += float((fake_sig - s).abs().sum())
        l1_bkg += float((fake_bkg - (e - s)).abs().sum())

        e_fake.append(truth.at_axis(fake_sig, start))

    n_towers = truth.embed[0].size
    e_fake   = torch.cat(e_fake)

    result = {
        'n_events' : n,
        'l1_sig'   : l1_sig / (n * n_towers),
        'l1_bkg'   : l1_bkg / (n * n_towers),
        **jet_scores(e_fake, truth),
    }

    return (result, e_fake.cpu().numpy())

@torch.no_grad()
def reference_scores(truth, device):
    """The naive subtraction, in the same columns as a model.

    The jet energy is the cone sum of (embed - rho), as in an area based
    subtraction; the towers are clipped at zero for the per tower scores.
    """
    e = torch.from_numpy(truth.embed ).to(device, torch.float32)
    s = torch.from_numpy(truth.signal).to(device, torch.float32)

    rho  = e.median(dim = 2, keepdim = True).values
    fake = (e - rho).clip(min = 0)

    return {
        'label'    : 'median-rho',
        'epoch'    : -1,
        'updates'  : 0,
        'net'      : 'ref',
        'n_events' : len(e),
        'l1_sig'   : float((fake - s).abs().mean()),
        'l1_bkg'   : float(((e - fake) - (e - s)).abs().mean()),
        **jet_scores(truth.at_axis(e - rho), truth),
    }

def list_epochs(model_dir, spec):
    if spec == 'final':
        return [ None ]

    if spec != 'all':
        return [ int(x) for x in spec.split(',') ]

    files  = glob.glob(os.path.join(model_dir, 'checkpoints', '*_net_*.pth'))
    epochs = sorted({
        int(re.match(r'(\d+)_', os.path.basename(f)).group(1)) for f in files
    })

    if os.path.exists(os.path.join(model_dir, 'net_gen_ba.pth')):
        epochs.append(None)

    return epochs

def evaluate_model(model_dir, cmdargs, truth, device):
    args  = Args.load(model_dir)
    model = construct_model(
        args.savedir, args.config, is_train = False, device = device
    )
    model.eval()

    csv  = os.path.join(model_dir, 'evals', f'{cmdargs.truth}_truth.csv')
    done = set()

    if os.path.exists(csv) and not cmdargs.force:
        old  = pd.read_csv(csv)
        done = set(zip(old['epoch'], old['net']))

    wanted = cmdargs.nets.split(',')
    rows   = []

    for epoch in list_epochs(model_dir, cmdargs.epochs):
        tag = -1 if epoch is None else epoch
        todo = [ n for n in wanted if (tag, n) not in done ]

        if not todo:
            continue

        try:
            model.load(epoch)
        except (OSError, RuntimeError, EOFError) as e:
            # e.g. a checkpoint the training is still writing
            print(f'{args.label} epoch {epoch}: cannot load ({e})')
            continue

        for net in todo:
            if NETS[net] not in model.models:
                continue

            gen = unwrap_model(model.models[NETS[net]])
            (scores, e_fake) = score_generator(
                gen, model.data_norm, truth, cmdargs.batch, device
            )

            row = {
                'label'   : args.label,
                'epoch'   : tag,
                'updates' : (
                    -1 if epoch is None
                    else epoch * args.config.steps_per_epoch
                ),
                'net'     : net,
                **scores,
            }
            rows.append(row)

            outdir = os.path.join(
                model_dir, 'evals', f'{cmdargs.truth}_truth'
            )
            os.makedirs(outdir, exist_ok = True)
            np.save(os.path.join(outdir, f'e{tag:04d}_{net}.npy'), e_fake)

            print(format_row(row), flush = True)

    if rows:
        new = pd.DataFrame(rows, columns = COLUMNS)
        if os.path.exists(csv):
            new = pd.concat([ pd.read_csv(csv), new ])
            new = new.drop_duplicates([ 'epoch', 'net' ], keep = 'last')
        new.sort_values([ 'epoch', 'net' ]).to_csv(csv, index = False)

    return rows

class EpochScorer:
    """Score a model in training against the val truth.

    Passed as `train(args_dict, epoch_callback = EpochScorer())`, it scores
    the networks every `every` epochs and appends one row per network to
    MODEL_DIR/val_truth_history.csv, the held-out counterpart of
    history.csv. `eval_time` is the time the scoring took, which is not
    part of the `epoch_time` of history.csv.
    """

    def __init__(
        self, n_events = 20000, every = 1, nets = ('raw', 'ema'),
        min_jet = 10.0, batch = 500
    ):
        # pylint: disable=too-many-arguments
        (embed, signal) = load_pairs(
            os.environ.get('UVCGAN_S_DATA', 'data'), 20000, 0
        )

        self._embed   = embed [:n_events]
        self._signal  = signal[:n_events]
        self._every   = every
        self._nets    = nets
        self._min_jet = min_jet
        self._batch   = batch
        self._truth   = None

    def __call__(self, model, epoch):
        if epoch % self._every != 0:
            return

        if self._truth is None:
            self._truth = Truth(
                self._embed, self._signal, cone_kernel(R_JET),
                self._min_jet, model.device
            )

        rows = []

        for net in self._nets:
            if NETS[net] not in model.models:
                continue

            t0  = time.perf_counter()
            gen = unwrap_model(model.models[NETS[net]])

            (scores, _e_fake) = score_generator(
                gen, model.data_norm, self._truth, self._batch, model.device
            )

            rows.append({
                'epoch'     : epoch,
                'net'       : net,
                **scores,
                'eval_time' : time.perf_counter() - t0,
            })

        path = os.path.join(model.savedir, 'val_truth_history.csv')
        pd.DataFrame(rows).to_csv(
            path, mode = 'a', header = not os.path.exists(path),
            index = False
        )

def format_row(row):
    return (
        f"{row['label']:>28s} {row['epoch']:5d} {row['updates']:7d}"
        f" {row['net']:>3s}  l1_sig {row['l1_sig']:.4f}"
        f"  l1_bkg {row['l1_bkg']:.4f}  jes {row['jes']:.3f}"
        f"  bias {row['bias']:+6.2f}  jer {row['jer']:5.2f}"
        f"  jer_iqr {row['jer_iqr']:5.2f}  jer_cal {row['jer_cal']:5.2f}"
        f"  ({row['n_jets']} jets)"
    )

def main():
    cmdargs = parse_cmdargs()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    (embed, signal) = load_pairs(
        cmdargs.data, cmdargs.n_events, cmdargs.seed, cmdargs.cache,
        cmdargs.truth
    )
    truth = Truth(
        embed, signal, cone_kernel(R_JET), cmdargs.min_jet, device
    )

    print(format_row(reference_scores(truth, device)))

    for model_dir in cmdargs.models:
        evaluate_model(model_dir.rstrip('/'), cmdargs, truth, device)

if __name__ == '__main__':
    main()
