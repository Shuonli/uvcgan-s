"""Two-process DDP smoke test on CPU (gloo backend).

Trains a tiny UVCGAN-S model for two epochs on synthetic data with two
processes and checks that both processes end up with identical weights,
that the losses are finite, and that only the main process writes files.
"""

import os
import socket
import tempfile
import unittest

# Spawned workers import torch (libgomp) before numpy; with a conda MKL
# numpy that aborts unless the GNU threading layer is selected.
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from uvcgan_s import train
from uvcgan_s.torch.distributed import unwrap_model

SHAPE = (24, 64)

def make_dataset(root, n = 8):
    rng = np.random.default_rng(0)

    for domain in ('a0', 'a1', 'b'):
        path = os.path.join(root, 'train', domain)
        os.makedirs(path)

        for i in range(n):
            image = np.abs(rng.normal(size = SHAPE)).astype(np.float32)
            np.savez(os.path.join(path, f'{i:04d}.npz'), image)

def make_dataset_config(datadir, domain):
    # an absolute path bypasses ROOT_DATA, which is fixed at import time
    return {
        'dataset' : {
            'name'   : 'ndarray-domain-hierarchy',
            'path'   : datadir,
            'domain' : domain,
        },
        'shape'           : (1, *SHAPE),
        'transform_train' : None,
        'transform_test'  : None,
    }

def make_args(datadir, outdir, epochs = 2):
    return {
        'batch_size' : 2,
        'data' : {
            'datasets'   : [
                make_dataset_config(datadir, d) for d in ('a0', 'a1', 'b')
            ],
            'merge_type' : 'unpaired',
            'workers'    : 0,
        },
        'epochs'        : epochs,
        'discriminator' : {
            'model'      : 'resnet',
            'model_args' : {
                'block_specs' : [
                    ('stem', {
                        'kernel_size' : 3, 'padding' : 1, 'stride' : 1,
                        'features' : 8
                    }),
                    ('resnet', 1),
                    ('stem', {
                        'kernel_size' : 2, 'padding' : 0, 'stride' : 2,
                        'features' : 16
                    }),
                    ('resnet', 1),
                ],
                'norm'        : 'batch',
                'activ'       : 'leakyrelu',
                'rezero'      : True,
                'reduce_output_channels' : True,
            },
            'optimizer'   : { 'name' : 'Adam', 'lr' : 1e-4 },
            'weight_init' : { 'name' : 'normal', 'init_gain' : 0.02 },
            'spectr_norm' : True,
        },
        'generator' : {
            'model' : 'vit-modnet',
            'model_args' : {
                'features'             : 16,
                'n_heads'              : 2,
                'n_blocks'             : 1,
                'ffn_features'         : 32,
                'embed_features'       : 16,
                'activ'                : 'gelu',
                'norm'                 : 'layer',
                'modnet_features_list' : [8, 16],
                'modnet_activ'         : 'leakyrelu',
                'modnet_norm'          : None,
                'modnet_downsample'    : 'conv',
                'modnet_upsample'      : 'upsample-conv',
                'modnet_rezero'        : False,
                'rezero'               : True,
                'activ_output'         : None,
                'style_rezero'         : True,
                'style_bias'           : True,
                'n_ext'                : 1,
            },
            'optimizer'   : { 'name' : 'Adam', 'lr' : 1e-4 },
            'weight_init' : { 'name' : 'kaiming' },
        },
        'model'      : 'uvcgan-s',
        'model_args' : {
            'ema_momentum'    : 0.99,
            'data_norm'       : { 'name' : 'log', 'bias' : 0.1 },
            'grad_clip'       : { 'norm' : 0.5 },
            'head_queue_size' : 0,
            'head_config'     : 'idt',
        },
        'seed'             : 0,
        'scheduler'        : {
            'name' : 'linear-v2', 'start_factor' : 0.1, 'total_iters' : 4
        },
        'loss'             : 'hinge',
        'steps_per_epoch'  : 2,
        'transfer'         : None,
        'gradient_penalty' : {
            'center' : 0, 'lambda_gp' : 0.01, 'mix_type' : 'real-fake',
            'reduction' : 'mean',
        },
        'label'      : 'ddp',
        'outdir'     : outdir,
        'log_level'  : 'WARNING',
        'checkpoint' : 1,
    }

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def worker(rank, world_size, port, datadir, outdir, dumpdir):
    os.environ.update({
        'RANK'          : str(rank),
        'LOCAL_RANK'    : str(rank),
        'WORLD_SIZE'    : str(world_size),
        'MASTER_ADDR'   : '127.0.0.1',
        'MASTER_PORT'   : str(port),
    })
    torch.set_num_threads(1)

    model = train(make_args(datadir, outdir))

    state = {}
    for (name, module) in model.models.items():
        module = unwrap_model(module)
        state[name] = {
            'params'  : dict(module.named_parameters()),
            'buffers' : dict(module.named_buffers()),
        }

    torch.save(state, os.path.join(dumpdir, f'rank{rank}.pth'))

class TestDDP(unittest.TestCase):

    def setUp(self):
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
            os.environ.pop(key, None)

    def test_single_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            datadir = os.path.join(tmp, 'data')
            outdir  = os.path.join(tmp, 'outdir')
            make_dataset(datadir)

            model = train(make_args(datadir, outdir))

            history = pd.read_csv(os.path.join(model.savedir, 'history.csv'))
            self.assertEqual(len(history), 2)
            self.assertTrue(np.isfinite(history.drop(columns = 'time')).all().all())

    def test_two_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            datadir = os.path.join(tmp, 'data')
            outdir  = os.path.join(tmp, 'outdir')
            dumpdir = os.path.join(tmp, 'dump')
            make_dataset(datadir)
            os.makedirs(dumpdir)

            mp.spawn(
                worker, args = (2, free_port(), datadir, outdir, dumpdir),
                nprocs = 2, join = True
            )

            states = [
                torch.load(
                    os.path.join(dumpdir, f'rank{r}.pth'), weights_only = True
                ) for r in range(2)
            ]

            for name in states[0]:
                for kind in ('params', 'buffers'):
                    for (k, v0) in states[0][name][kind].items():
                        # wrappers must not leak into checkpoint keys
                        self.assertNotIn('module', k.split('.'))

                        # BatchNorm statistics are per process by design
                        if k.endswith(('running_mean', 'running_var')):
                            continue

                        v1 = states[1][name][kind][k]
                        self.assertTrue(
                            torch.allclose(v0, v1, rtol = 1e-6, atol = 1e-7),
                            f'{name}.{k} differs between processes'
                        )

            (savedir, ) = [
                os.path.join(outdir, d) for d in os.listdir(outdir)
            ]
            history = pd.read_csv(os.path.join(savedir, 'history.csv'))

            self.assertEqual(len(history), 2)
            self.assertTrue(np.isfinite(history.drop(columns = 'time')).all().all())
            self.assertIn('samples_per_sec', history.columns)

            checkpoints = os.listdir(os.path.join(savedir, 'checkpoints'))
            # 7 networks + 2 optimizers + 2 schedulers, written once per epoch
            self.assertEqual(len(checkpoints), 2 * 11)

if __name__ == '__main__':
    unittest.main()
