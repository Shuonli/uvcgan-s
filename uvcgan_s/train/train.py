import logging
import time

from itertools import islice
import tqdm

from uvcgan_s.config            import Args
from uvcgan_s.data              import construct_data_loaders
from uvcgan_s.data.data         import set_loader_epoch
from uvcgan_s.torch.funcs       import (
    get_torch_device_smart, seed_everything, lazy_metrics
)
from uvcgan_s.torch.distributed import (
    init_distributed, cleanup_distributed, barrier, is_main_process,
    get_world_size, reduce_dict
)
from uvcgan_s.cgan              import construct_model
from uvcgan_s.utils.log         import setup_logging

from .metrics   import LossMetrics
from .callbacks import TrainingHistory
from .transfer  import transfer

LOGGER = logging.getLogger('uvcgan_s.train')

# steps between updates of the progress bar when the losses stay on the
# device, c.f. `lazy_metrics`
LAZY_DISPLAY_PERIOD = 50

def to_host(values):
    return { k : float(v) for (k, v) in values.items() }

def training_epoch(
    it_train, model, title, steps_per_epoch, show_progress = True
):
    model.train()

    steps = len(it_train)
    if steps_per_epoch is not None:
        steps = min(steps, steps_per_epoch)

    progbar = tqdm.tqdm(
        desc = title, total = steps, dynamic_ncols = True,
        disable = not show_progress
    )
    metrics = LossMetrics()
    lazy    = lazy_metrics()

    for (step, batch) in enumerate(islice(it_train, steps)):
        model.set_input(batch)
        model.optimization_step()

        metrics.update(model.get_current_losses(lazy = lazy))

        if not lazy:
            progbar.set_postfix(metrics.values, refresh = False)
        elif step % LAZY_DISPLAY_PERIOD == 0:
            progbar.set_postfix(to_host(metrics.values), refresh = False)

        progbar.update()

    progbar.close()

    if lazy and (metrics.values is not None):
        # the only read back of the epoch
        metrics = LossMetrics(to_host(metrics.values), n = 1)

    return (metrics, steps)

def try_continue_training(args, model):
    history = TrainingHistory(args.savedir)

    start_epoch = model.find_last_checkpoint_epoch()
    model.load(start_epoch)

    if start_epoch > 0:
        history.load()

    start_epoch = max(start_epoch, 0)

    return (start_epoch, history)

def construct_args(args_dict):
    # The main process writes the config and label, the others only read them
    if is_main_process():
        args = Args.from_args_dict(**args_dict)

    barrier()

    if not is_main_process():
        args = Args.from_args_dict(**args_dict, write = False)

    return args

def setup_rank_logging(log_level):
    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level)

    if not is_main_process():
        log_level = max(log_level, logging.WARNING)

    setup_logging(log_level)

def reduce_metrics(metrics, epoch_time, steps, batch_size):
    # losses are averaged over processes; timing is measured by this process
    values = reduce_dict(metrics.values or {})

    values['epoch_time']      = epoch_time
    values['samples_per_sec'] = steps * batch_size * get_world_size() / epoch_time

    return LossMetrics(values, n = 1)

def train(args_dict, epoch_callback = None):
    """Train a model described by `args_dict`.

    `epoch_callback(model, epoch)`, if given, is called by the main process
    at the end of every epoch, e.g. to score the model on held-out data.
    Its run time is not part of the recorded `epoch_time`.
    """
    # pylint: disable=too-many-locals
    distributed = init_distributed()

    args = construct_args(args_dict)

    setup_rank_logging(args.log_level)
    seed_everything(args.config.seed)

    device   = get_torch_device_smart()
    it_train = construct_data_loaders(
        args.config.data, args.config.batch_size, split = 'train'
    )

    if is_main_process():
        print("Starting training...")
        print(args.config.to_json(indent = 4))

        if distributed:
            LOGGER.info(
                "DDP: %d processes, %d samples per step",
                get_world_size(), get_world_size() * args.config.batch_size
            )

    model = construct_model(
        args.savedir, args.config, is_train = True, device = device
    )
    start_epoch, history = try_continue_training(args, model)

    if (start_epoch == 0) and (args.transfer is not None):
        transfer(model, args.transfer)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        set_loader_epoch(it_train, epoch)

        title = 'Epoch %d / %d' % (epoch, args.epochs)
        start = time.perf_counter()

        metrics, steps = training_epoch(
            it_train, model, title, args.config.steps_per_epoch,
            show_progress = is_main_process()
        )

        epoch_time = time.perf_counter() - start
        metrics    = reduce_metrics(
            metrics, epoch_time, steps, args.config.batch_size
        )

        if is_main_process():
            LOGGER.info(
                "Epoch %d: %d steps in %.1f s, %.1f samples/s",
                epoch, steps, epoch_time, metrics.values['samples_per_sec']
            )
            history.end_epoch(epoch, metrics)

        model.end_epoch(epoch)

        if (epoch_callback is not None) and is_main_process():
            epoch_callback(model, epoch)

        if epoch % args.checkpoint == 0:
            model.save(epoch)

    model.save(epoch = None)
    cleanup_distributed()

    return model
