"""Training of the sPHENIX configuration at a given batch size and rate.

Used to find out whether progress on this problem tracks the number of
optimizer updates or the number of samples seen, which is what decides
whether a larger batch (or more GPUs) shortens training:

    DIAG_BATCH   samples per process per step   (default 32)
    DIAG_LR      learning rate of both optimizers (default 5e-5)
    DIAG_STEPS   updates per epoch, i.e. per row of history.csv (default 200)
    DIAG_EPOCHS  epochs; the run is normally stopped by its time limit
    DIAG_LABEL   model label
    DIAG_SUBDIR  directory under OUTDIR/sphenix (default 'diag')
    DIAG_CHECKPOINT  epochs between checkpoints (default: never)

Relative to `train_uvcgan-s.py` only the run length and the warm-up are
different: epochs are short so that `history.csv` is sampled finely, and
the warm-up lasts one epoch so that it does not consume the range of
samples that the batch sizes have in common.
"""

import os
import torch

from uvcgan_s import ROOT_OUTDIR, train

torch.backends.cudnn.benchmark = True

DATA_PATH = 'sphenix/2025-06-05_jet_bkg_sub'
GP_LAMBDA = 0.01

BATCH  = int(os.environ.get('DIAG_BATCH', 32))
LR     = float(os.environ.get('DIAG_LR', 5e-5))
STEPS  = int(os.environ.get('DIAG_STEPS', 200))
EPOCHS = int(os.environ.get('DIAG_EPOCHS', 2000))
LABEL  = os.environ.get('DIAG_LABEL', f'diag_b{BATCH}_lr{LR:g}')
SUBDIR = os.environ.get('DIAG_SUBDIR', 'diag')
CKPT   = int(os.environ.get('DIAG_CHECKPOINT', 1000000))

DISC_BLOCKS = [
    # (1, 24, 64)
    (
        'stem',
        { 'kernel_size' : 3, 'padding' : 1, 'stride' : 1, 'features' : 64 }
    ),
    # (64, 24, 64)
    ('resnet', 3),
    (
        'stem',
        { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 128 }
    ),
    # (128, 12, 32)
    ('resnet', 3),
    (
        'stem',
        { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 256 }
    ),
    # (256, 6, 16)
    ('resnet', 3),
    (
        'stem',
        { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 512 }
    ),
    # (512, 3, 8)
    ('resnet', 3),
]

def dataset(domain):
    return {
        'dataset' : {
            'name'   : 'h5array-domain-hierarchy',
            'path'   : DATA_PATH,
            'domain' : domain,
        },
        'shape'           : (1, 24, 64),
        'transform_train' : None,
        'transform_test'  : None,
    }

args_dict = {
    'batch_size' : BATCH,
    'data' : {
        'datasets'   : [
            dataset('background'), dataset('signal'), dataset('embed')
        ],
        'merge_type' : 'unpaired',
        'workers'    : int(os.environ.get('DIAG_WORKERS', 8)),
    },
    'epochs'        : EPOCHS,
    'discriminator' : {
        'model'      : 'resnet',
        'model_args' : {
            'block_specs' : DISC_BLOCKS,
            'norm'        : 'batch',
            'activ'       : 'leakyrelu',
            'rezero'      : True,
            'reduce_output_channels' : True,
        },
        'optimizer' : {
            'name'  : 'Adam',
            'lr'    : LR,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
        'spectr_norm' : True,
    },
    'generator' : {
        'model' : 'vit-modnet',
        'model_args' : {
            'features'             : 384,
            'n_heads'              : 6,
            'n_blocks'             : 12,
            'ffn_features'         : 1536,
            'embed_features'       : 384,
            'activ'                : 'gelu',
            'norm'                 : 'layer',
            'modnet_features_list' : [96, 192, 384],
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
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : LR,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name' : 'kaiming',
        },
    },
    'model' : 'uvcgan-s',
    'model_args' : {
        'lambda_adv_a0'   : 1,
        'lambda_adv_a1'   : 1,
        'lambda_adv_b'    : 1,
        'lambda_cyc_a0'   : 10,
        'lambda_cyc_a1'   : 100,
        'lambda_cyc_b'    : 10,
        'lambda_idt_aa'   : 0.5,
        'lambda_idt_bb'   : 0.5,
        'ema_momentum'    : 0.9999,
        'data_norm'       : {
            'name' : 'log',
            'bias' : 0.1,
        },
        'gp_cache_period' : 0,
        'grad_clip'       : { 'norm' : 0.5 },
        'norm_loss_a0'    : False,
        'norm_loss_a1'    : False,
        'norm_loss_b'     : False,
        'norm_disc_a0'    : False,
        'norm_disc_a1'    : False,
        'norm_disc_b'     : False,
        'head_queue_size' : 0,
        'head_config'     : 'idt'
    },
    'seed'  : 0,
    'scheduler' : {
        'name' : 'linear-v2',
        'start_factor' : 0.01,
        'end_factor'   : 1.0,
        'total_iters'  : 1,
    },
    'loss'             : 'hinge',
    'steps_per_epoch'  : STEPS,
    'transfer'         : None,
    'gradient_penalty' : {
        'center'    : 0,
        'lambda_gp' : GP_LAMBDA,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    },
# args
    'label'  : LABEL,
    'outdir' : os.path.join(ROOT_OUTDIR, 'sphenix', SUBDIR),
    'log_level'  : 'INFO',
    # the run is normally stopped by its time limit, and history.csv is
    # written every epoch; checkpoints are only needed by the base run that
    # the mid-training measurement branches off
    'checkpoint' : CKPT,
}

train(args_dict)
