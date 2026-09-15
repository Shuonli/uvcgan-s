import logging
import random
import torch
import numpy as np

from .distributed import wrap_model, unwrap_model

LOGGER = logging.getLogger('uvcgan_s.torch')

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

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

def clip_gradients(optimizer, norm = None, value = None):
    if (norm is None) and (value is None):
        return

    params = [
        param
            for param_group in optimizer.param_groups
                for param in param_group['params']
    ]

    if norm is not None:
        torch.nn.utils.clip_grad_norm_(params, max_norm = norm)

    if value is not None:
        torch.nn.utils.clip_grad_value_(params, clip_value = value)
