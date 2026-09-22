import logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from uvcgan_s.base.schedulers   import get_scheduler
from uvcgan_s.torch.distributed import is_main_process
from .named_dict import NamedDict
from .checkpoint import find_last_checkpoint_epoch, save, load

PREFIX_MODEL = 'net'
PREFIX_OPT   = 'opt'
PREFIX_SCHED = 'sched'

LOGGER = logging.getLogger('uvcgan_s.cgan')

class ModelBase:
    # pylint: disable=too-many-instance-attributes

    def __init__(self, savedir, config, is_train, device):
        self.is_train = is_train
        self.device   = device
        self.savedir  = savedir
        self._config  = config

        self.models = self._setup_models(config)
        self.images = self._setup_images(config)
        self.losses = self._setup_losses(config)
        self.metric = 0
        self.epoch  = 0

        self.optimizers = NamedDict()
        self.schedulers = NamedDict()

        self._config_lrs = {}

        if is_train:
            self.optimizers = self._setup_optimizers(config)

            # NOTE: loading an optimizer restores the learning rate it
            #       was saved with. Remember the configured rates so
            #       that `load` can put them back: the configuration,
            #       not the checkpoint, decides the rate. This matters
            #       when a checkpoint is continued with a different
            #       configuration, and is a no-op for a plain resume.
            #       They are read before the schedulers are built, as a
            #       warm-up scales the optimizer's rate down on the spot.
            self._config_lrs = {
                name : [ g['lr'] for g in opt.param_groups ]
                    for (name, opt) in self.optimizers.items()
                        if opt is not None
            }

            self.schedulers = self._setup_schedulers(config)

    def set_input(self, inputs, domain = None):
        for key in self.images:
            self.images[key] = None

        self._set_input(inputs, domain)

    def forward(self):
        raise NotImplementedError

    def optimization_step(self):
        raise NotImplementedError

    def _set_input(self, inputs, domain):
        raise NotImplementedError

    def _setup_images(self, config):
        raise NotImplementedError

    def _setup_models(self, config):
        raise NotImplementedError

    def _setup_losses(self, config):
        raise NotImplementedError

    def _setup_optimizers(self, config):
        raise NotImplementedError

    def _setup_schedulers(self, config):
        schedulers = { }

        for (name, opt) in self.optimizers.items():
            schedulers[name] = get_scheduler(opt, config.scheduler)

        return NamedDict(**schedulers)

    def _save_model_state(self, epoch):
        pass

    def _load_model_state(self, epoch):
        pass

    def _handle_epoch_end(self):
        pass

    def eval(self):
        self.is_train = False

        for model in self.models.values():
            model.eval()

    def train(self):
        self.is_train = True

        for model in self.models.values():
            model.train()

    def forward_nograd(self):
        with torch.no_grad():
            self.forward()

    def find_last_checkpoint_epoch(self):
        return find_last_checkpoint_epoch(self.savedir, PREFIX_MODEL)

    def load(self, epoch):
        if (epoch is not None) and (epoch <= 0):
            return

        LOGGER.debug('Loading model from epoch %s', epoch)

        load(self.models,     self.savedir, PREFIX_MODEL, epoch, self.device)
        load(self.optimizers, self.savedir, PREFIX_OPT,   epoch, self.device)
        load(self.schedulers, self.savedir, PREFIX_SCHED, epoch, self.device)

        self._restore_configured_lr()

        self.epoch = epoch
        self._load_model_state(epoch)
        self._handle_epoch_end()

    def _restore_configured_lr(self):
        for (name, lrs) in self._config_lrs.items():
            optimizer = self.optimizers[name]

            for (group, lr) in zip(optimizer.param_groups, lrs):
                if group['lr'] != lr:
                    LOGGER.warning(
                        "Optimizer '%s': keeping the configured learning"
                        " rate %g instead of the checkpoint's %g",
                        name, lr, group['lr']
                    )
                group['lr']         = lr
                group['initial_lr'] = lr

            scheduler = self.schedulers.get(name, None)
            if scheduler is not None:
                scheduler.base_lrs = list(lrs)

    def save(self, epoch = None):
        if not is_main_process():
            # under DDP all processes hold identical weights
            return

        LOGGER.debug('Saving model at epoch %s', epoch)

        save(self.models,     self.savedir, PREFIX_MODEL, epoch)
        save(self.optimizers, self.savedir, PREFIX_OPT,   epoch)
        save(self.schedulers, self.savedir, PREFIX_SCHED, epoch)

        self._save_model_state(epoch)

    def end_epoch(self, epoch = None):
        for scheduler in self.schedulers.values():
            if scheduler is None:
                continue

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(self.metric)
            else:
                scheduler.step()

        self._handle_epoch_end()

        if epoch is None:
            self.epoch = self.epoch + 1
        else:
            self.epoch = epoch

    def pprint(self, verbose):
        for name,model in self.models.items():
            num_params = 0

            for param in model.parameters():
                num_params += param.numel()

            if verbose:
                print(model)

            print(
                '[Network %s] Total number of parameters : %.3f M' % (
                    name, num_params / 1e6
                )
            )

    def set_requires_grad(self, models, requires_grad = False):
        # pylint: disable=no-self-use
        if not isinstance(models, list):
            models = [models, ]

        for model in models:
            for param in model.parameters():
                param.requires_grad = requires_grad

    def get_current_losses(self):
        result = {}

        for (k,v) in self.losses.items():
            result[k] = float(v)

        return result

    def get_images(self):
        return self.images


