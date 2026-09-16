"""Thin helpers around `torch.distributed` (DDP).

Process-group initialization is driven purely by environment variables, so a
training script runs unchanged as a single process, under `torchrun`, or
under SLURM `srun` with one task per GPU:

    torchrun : RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    srun     : SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS
               (+ MASTER_ADDR / MASTER_PORT exported by the batch script,
                c.f. scripts/slurm/bench_ddp.sbatch)
"""

import contextlib
import datetime
import logging
import os
import subprocess

import torch
import torch.distributed as dist

from torch import nn
from torch.nn.parallel import DistributedDataParallel

LOGGER = logging.getLogger('uvcgan_s.torch')

WRAPPER_TYPES = (DistributedDataParallel, nn.DataParallel)

def _env_int(names, default = None):
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)

    return default

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    if is_distributed():
        return dist.get_world_size()

    return _env_int([ 'WORLD_SIZE', 'SLURM_NTASKS' ], 1)

def get_rank():
    if is_distributed():
        return dist.get_rank()

    return _env_int([ 'RANK', 'SLURM_PROCID' ], 0)

def get_local_rank():
    return _env_int([ 'LOCAL_RANK', 'SLURM_LOCALID' ], 0)

def is_main_process():
    return get_rank() == 0

def _guess_master_addr():
    nodelist = os.environ.get('SLURM_JOB_NODELIST')

    if nodelist:
        try:
            hostnames = subprocess.check_output(
                [ 'scontrol', 'show', 'hostnames', nodelist ], text = True
            )
            return hostnames.split()[0]

        except (OSError, subprocess.CalledProcessError, IndexError):
            pass

    return 'localhost'

def init_distributed(backend = None, timeout_minutes = None):
    """Initialize the default process group if launched with > 1 process.

    Returns True when running distributed and False for a plain single
    process run.
    """
    if is_distributed():
        return True

    world_size = get_world_size()
    if world_size <= 1:
        return False

    rank       = get_rank()
    local_rank = get_local_rank()

    if timeout_minutes is None:
        # a shorter timeout turns a collective deadlock into a fast
        # failure, which is what benchmarks and CI want
        timeout_minutes = float(
            os.environ.get('UVCGAN_S_DDP_TIMEOUT_MIN', 30)
        )

    os.environ.setdefault('MASTER_ADDR', _guess_master_addr())
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ['RANK']       = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)

    device_id = None
    if torch.cuda.is_available():
        # SLURM may expose either all GPUs of a node or a single GPU per task
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device_id = torch.device('cuda', torch.cuda.current_device())

    if backend is None:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'

    # declaring the device binds this rank to its GPU for NCCL up front
    dist.init_process_group(
        backend     = backend,
        init_method = 'env://',
        timeout     = datetime.timedelta(minutes = timeout_minutes),
        device_id   = device_id,
    )

    LOGGER.info(
        "Distributed training: rank %d / %d (local rank %d), backend '%s'",
        rank, world_size, local_rank, backend
    )

    return True

def cleanup_distributed():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()

def barrier():
    if is_distributed():
        dist.barrier()

def unwrap_model(model):
    """Strip DDP / DataParallel wrappers."""
    while isinstance(model, WRAPPER_TYPES):
        model = model.module

    return model

def wrap_model(model):
    """Wrap `model` for multi-GPU training.

    In a distributed run each process drives one GPU and `model` is wrapped
    into DDP. Otherwise the legacy behavior of `DataParallel` over all visible
    GPUs is preserved.
    """
    if is_distributed():
        if not any(p.requires_grad for p in model.parameters()):
            # DDP refuses modules without trainable parameters (e.g. Identity)
            return model

        device_ids = None
        if torch.cuda.is_available():
            device_ids = [ torch.cuda.current_device(), ]

        # Tuning knobs (c.f. `torch.nn.parallel.DistributedDataParallel`):
        #   UVCGAN_S_DDP_FIND_UNUSED=1   find_unused_parameters
        #   UVCGAN_S_DDP_BUCKET_MB=25    bucket_cap_mb
        #   UVCGAN_S_DDP_BUCKET_VIEW=1   gradient_as_bucket_view
        #   UVCGAN_S_DDP_COMPRESS=bf16   fp16 / bf16 gradient compression
        #   UVCGAN_S_DDP_STATIC_GRAPH=1  static_graph
        env         = os.environ
        find_unused = bool(int(env.get('UVCGAN_S_DDP_FIND_UNUSED', 0)))
        bucket_mb   = float(env.get('UVCGAN_S_DDP_BUCKET_MB', 25))
        bucket_view = bool(int(env.get('UVCGAN_S_DDP_BUCKET_VIEW', 0)))
        compress    = env.get('UVCGAN_S_DDP_COMPRESS', None)
        static      = bool(int(env.get('UVCGAN_S_DDP_STATIC_GRAPH', 0)))

        # NOTE: buffers (BatchNorm running statistics, spectral norm
        #       vectors) are synchronized once at construction and then
        #       evolve per process. DDP's per-forward buffer broadcast
        #       writes them in place between the real / fake forwards of
        #       a discriminator, invalidating tensors saved for backward.
        ddp = DistributedDataParallel(
            model,
            device_ids              = device_ids,
            broadcast_buffers       = False,
            find_unused_parameters  = find_unused,
            bucket_cap_mb           = bucket_mb,
            gradient_as_bucket_view = bucket_view,
            static_graph            = static,
        )

        if compress:
            # pylint: disable=import-outside-toplevel
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default_hooks
            )
            hooks = {
                'fp16' : default_hooks.fp16_compress_hook,
                'bf16' : default_hooks.bf16_compress_hook,
            }
            ddp.register_comm_hook(state = None, hook = hooks[compress])

        return ddp

    if torch.cuda.device_count() > 1:
        LOGGER.warning(
            "Multiple (%d) GPUs found. Using Data Parallelism",
            torch.cuda.device_count()
        )
        return nn.DataParallel(model)

    return model

def _ddp_submodules(module):
    return [
        m for m in module.modules() if isinstance(m, DistributedDataParallel)
    ]

@contextlib.contextmanager
def no_sync(*modules):
    """Disable DDP gradient synchronization of `modules`.

    Gradients computed inside this context accumulate locally and are
    all-reduced by the next synchronized backward pass. Applies to DDP
    submodules of `modules` as well (c.f. `BatchHeadWrapper`). No-op outside
    of a distributed run.
    """
    with contextlib.ExitStack() as stack:
        for module in modules:
            for ddp in _ddp_submodules(module):
                stack.enter_context(ddp.no_sync())

        yield

def reduce_dict(values, average = True):
    """All-reduce a dict of scalars across processes."""
    if (not is_distributed()) or (not values):
        return values

    keys = sorted(values.keys())

    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
    else:
        device = torch.device('cpu')

    tensor = torch.tensor(
        [ float(values[k]) for k in keys ], dtype = torch.float64,
        device = device
    )

    dist.all_reduce(tensor)

    if average:
        tensor /= dist.get_world_size()

    return dict(zip(keys, tensor.tolist()))
