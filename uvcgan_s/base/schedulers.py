from torch.optim         import lr_scheduler
from uvcgan_s.torch.select import extract_name_kwargs

def linear_scheduler(optimizer, epochs_warmup, epochs_anneal):

    def lambda_rule(epoch, epochs_warmup, epochs_anneal):
        if epoch < epochs_warmup:
            return 1.0

        return 1.0 - (epoch - epochs_warmup) / (epochs_anneal + 1)

    lr_fn = lambda epoch : lambda_rule(epoch, epochs_warmup, epochs_anneal)

    return lr_scheduler.LambdaLR(optimizer, lr_fn)

SCHED_DICT = {
    'step'            : lr_scheduler.StepLR,
    'plateau'         : lr_scheduler.ReduceLROnPlateau,
    'cosine'          : lr_scheduler.CosineAnnealingLR,
    'cosine-restarts' : lr_scheduler.CosineAnnealingWarmRestarts,
    'constant'        : lr_scheduler.ConstantLR,
    # lr scheds below are for backward compatibility
    'linear'    : linear_scheduler,
    'linear-v2' : lr_scheduler.LinearLR,
    'CosineAnnealingWarmRestarts' : lr_scheduler.CosineAnnealingWarmRestarts,
}

def select_single_scheduler(optimizer, scheduler):
    if scheduler is None:
        return None

    name, kwargs = extract_name_kwargs(scheduler)

    if name not in SCHED_DICT:
        raise ValueError(
            f"Unknown scheduler: '{name}'. Supported: {SCHED_DICT.keys()}"
        )

    return SCHED_DICT[name](optimizer, **kwargs)

def select_scheduler(optimizer, scheduler, compose = False):
    if scheduler is None:
        return None

    if not isinstance(scheduler, (list, tuple)):
        scheduler = [ scheduler, ]

    result = [ select_single_scheduler(optimizer, x) for x in scheduler ]

    if compose:
        if len(result) == 1:
            return result[0]
        else:
            return lr_scheduler.ChainedScheduler(result)
    else:
        return result

def get_scheduler(optimizer, scheduler):
    return select_scheduler(optimizer, scheduler, compose = True)

