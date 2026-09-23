import contextlib
import logging
import os
import random
import torch
import numpy as np

from .distributed import wrap_model, unwrap_model

LOGGER = logging.getLogger('uvcgan_s.torch')

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

AMP_DTYPES = {
    'bf16' : torch.bfloat16,
}

def get_amp_dtype():
    """Precision of the forward passes: `UVCGAN_S_AMP=bf16`, or None.

    Weights, gradients and optimizer states stay float32; autocast runs the
    operations it considers safe (convolutions, matrix products) in the
    lower precision.
    """
    name = os.environ.get('UVCGAN_S_AMP', '')

    if not name:
        return None

    if name not in AMP_DTYPES:
        raise ValueError(
            f"Unknown UVCGAN_S_AMP '{name}'. Supported: {list(AMP_DTYPES)}"
        )

    return AMP_DTYPES[name]

def autocast(device, dtype):
    """`torch.autocast` to `dtype`, or no-op if `dtype` is None."""
    if dtype is None:
        return contextlib.nullcontext()

    return torch.autocast(device_type = device.type, dtype = dtype)

def lazy_metrics():
    """Whether the losses stay on the device during an epoch.

    With `UVCGAN_S_LAZY_METRICS=1` the losses (and gradient norms) of every
    step are summed on the device and read back once per epoch, instead of
    once per step, which makes the host wait for the device to finish.
    """
    return os.environ.get('UVCGAN_S_LAZY_METRICS', '0') == '1'

def get_torch_device_smart():
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())

    return torch.device('cpu')

def prepare_model(model, device):
    return wrap_model(model.to(device))

@torch.no_grad()
def update_average_model(average_model, model, momentum):
    # TODO: Maybe it is better to copy buffers, instead of
    #       averaging them.
    #       Think about this later.
    average_model = unwrap_model(average_model)
    model         = unwrap_model(model)

    online_params = dict(model.named_parameters())
    online_bufs   = dict(model.named_buffers())

    for (k, v) in average_model.named_parameters():
        if v.ndim == 0:
            v.copy_(momentum * v + (1 - momentum) * online_params[k])
        else:
            v.lerp_(online_params[k], (1 - momentum))

    for (k, v) in average_model.named_buffers():
        if v.ndim == 0:
            v.copy_(momentum * v + (1 - momentum) * online_bufs[k])
        else:
            v.lerp_(online_bufs[k], (1 - momentum))

def clip_gradients(optimizer, norm = None, value = None, as_tensor = False):
    """Clip gradients in place, returning the norm before clipping.

    The norm is a float, or with `as_tensor` a tensor on the device, which
    does not make the host wait for the device.
    """
    if (norm is None) and (value is None):
        return None

    params = [
        param
            for param_group in optimizer.param_groups
                for param in param_group['params']
    ]

    total_norm = None

    if norm is not None:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm = norm)
        total_norm = total_norm.detach() if as_tensor else float(total_norm)

    if value is not None:
        torch.nn.utils.clip_grad_value_(params, clip_value = value)

    return total_norm
