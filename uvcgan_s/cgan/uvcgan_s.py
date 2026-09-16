# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import itertools
import torch

from uvcgan_s.torch.data_norm         import select_data_normalization
from uvcgan_s.torch.gan_losses        import select_gan_loss
from uvcgan_s.torch.select            import select_optimizer
from uvcgan_s.torch.queue             import FastQueue
from uvcgan_s.torch.funcs             import (
    update_average_model, clip_gradients
)
from uvcgan_s.torch.distributed       import (
    all_reduce_gradients, is_distributed, no_sync, wrap_model
)
from uvcgan_s.torch.layers.batch_head import BatchHeadWrapper, get_batch_head
from uvcgan_s.torch.gradient_penalty  import GradientPenalty
from uvcgan_s.torch.gradient_cacher   import GradientCacher
from uvcgan_s.models.discriminator    import construct_discriminator
from uvcgan_s.models.generator        import construct_generator

from .model_base import ModelBase
from .named_dict import NamedDict
from .funcs import set_asym_two_domain_input

def queued_forward(
    batch_head_model, input_image, queue, data_norm, normalize,
    update_queue = True
):
    # pylint: disable=too-many-arguments
    if normalize:
        input_image = data_norm.normalize(input_image)

    output, pred_body = batch_head_model.forward(
        input_image, extra_bodies = queue.query(), return_body = True
    )

    if update_queue:
        queue.push(pred_body)

    return output

def eval_loss_with_norm(loss_fn, a, b, data_norm, normalize):
    if normalize:
        a = data_norm.normalize(a)
        b = data_norm.normalize(b)

    return loss_fn(a, b)

class UVCGAN_S(ModelBase):
    # pylint: disable=too-many-instance-attributes

    def _setup_images(self, _config):
        images = [
            'real_a0', 'real_a1', 'real_a', 'real_b',
            'fake_a0', 'fake_a1', 'fake_a', 'fake_b',
            'reco_a0', 'reco_a1', 'reco_a', 'reco_b',
        ]

        if self.lambda_idt_bb:
            images += [ 'idt_bb_input', 'idt_bb', ]

        if self.lambda_idt_aa:
            images += [
                'idt_aa_input', 'idt_aa', 'idt_aa_a0', 'idt_aa_a1',
            ]

        return NamedDict(*images)

    def _construct_batch_head_disc(self, model_config, input_shape):
        distributed = is_distributed()

        disc_body = construct_discriminator(
            model_config, input_shape, self.device, wrap = not distributed
        )
        disc_head = get_batch_head(self.head_config).to(self.device)

        if distributed:
            # a single DDP wrapper around the whole discriminator keeps the
            # checkpoint keys identical to those of a single process run
            return wrap_model(BatchHeadWrapper(disc_body, disc_head))

        # legacy: DataParallel over body and head separately, so that the
        # queue concatenation happens on the main device
        return BatchHeadWrapper(disc_body, wrap_model(disc_head))

    def _setup_models(self, config):
        models = {}

        shape_a0 = tuple(config.data.datasets[0].shape)
        shape_a1 = tuple(config.data.datasets[1].shape)

        shape_a  = (shape_a0[0] + shape_a1[0], *shape_a1[1:])
        shape_b  = tuple(config.data.datasets[2].shape)

        models['gen_ab'] = construct_generator(
            config.generator, shape_a, shape_b, self.device
        )
        models['gen_ba'] = construct_generator(
            config.generator, shape_b, shape_a, self.device
        )

        if self.ema_momentum is not None:
            models['ema_gen_ab'] = construct_generator(
                config.generator, shape_a, shape_b, self.device
            )
            models['ema_gen_ba'] = construct_generator(
                config.generator, shape_b, shape_a, self.device
            )

            models['ema_gen_ab'].load_state_dict(models['gen_ab'].state_dict())
            models['ema_gen_ba'].load_state_dict(models['gen_ba'].state_dict())

        if self.is_train:
            models['disc_a0'] = self._construct_batch_head_disc(
                config.discriminator, shape_a0
            )
            models['disc_a1'] = self._construct_batch_head_disc(
                config.discriminator, shape_a1
            )
            models['disc_b'] = self._construct_batch_head_disc(
                config.discriminator, shape_b
            )

        return NamedDict(**models)

    def _setup_losses(self, config):
        losses = [
            'gen_ab',   'gen_ba0',  'gen_ba1',
            'cycle_a0', 'cycle_a1', 'cycle_b',
            'disc_a0',  'disc_a1',  'disc_b'
        ]

        if self.lambda_idt_aa:
            losses += [
                'idt_aa_a0', 'idt_aa_a1',
            ]

        if self.lambda_idt_bb:
            losses += [ 'idt_bb' ]

        if config.gradient_penalty is not None:
            losses += [ 'gp_a0', 'gp_a1', 'gp_b' ]

        return NamedDict(*losses)

    def _setup_optimizers(self, config):
        optimizers = NamedDict('gen', 'disc')

        optimizers.gen = select_optimizer(
            itertools.chain(
                self.models.gen_ab.parameters(),
                self.models.gen_ba.parameters()
            ),
            config.generator.optimizer
        )

        optimizers.disc = select_optimizer(
            itertools.chain(
                self.models.disc_a0.parameters(),
                self.models.disc_a1.parameters(),
                self.models.disc_b.parameters()
            ),
            config.discriminator.optimizer
        )

        return optimizers

    def __init__(
        self, savedir, config, is_train, device, head_config = None,
        lambda_adv_a0   = 1.0,
        lambda_adv_a1   = 1.0,
        lambda_adv_b    = 1.0,
        lambda_cyc_a0   = 10.0,
        lambda_cyc_a1   = 10.0,
        lambda_cyc_b    = 10.0,
        lambda_idt_aa   = 0.5,
        lambda_idt_bb   = 0.5,
        head_queue_size = 3,
        ema_momentum    = None,
        data_norm       = None,
        gp_cache_period = 1,
        grad_clip       = None,
        norm_loss_a0    = False,
        norm_loss_a1    = False,
        norm_loss_b     = False,
        norm_disc_a0    = False,
        norm_disc_a1    = False,
        norm_disc_b     = False,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        self.lambda_adv_a0 = lambda_adv_a0
        self.lambda_adv_a1 = lambda_adv_a1
        self.lambda_adv_b  = lambda_adv_b

        self.lambda_cyc_a0 = lambda_cyc_a0
        self.lambda_cyc_a1 = lambda_cyc_a1
        self.lambda_cyc_b  = lambda_cyc_b

        self.lambda_idt_aa = lambda_idt_aa
        self.lambda_idt_bb = lambda_idt_bb

        self.ema_momentum  = ema_momentum
        self.data_norm     = select_data_normalization(data_norm)
        self.head_config   = head_config or {}

        assert len(config.data.datasets) == 3, \
            "Asymmetric CycleGAN expects a triplet of datasets"

        self._c_a0 = config.data.datasets[0].shape[0]
        self._c_a1 = config.data.datasets[1].shape[0]

        self._grad_clip = grad_clip or {}

        self._norm_loss_a0 = norm_loss_a0
        self._norm_loss_a1 = norm_loss_a1
        self._norm_loss_b  = norm_loss_b

        self._norm_disc_a0 = norm_disc_a0
        self._norm_disc_a1 = norm_disc_a1
        self._norm_disc_b  = norm_disc_b

        super().__init__(savedir, config, is_train, device)

        self.criterion_gan     = select_gan_loss(config.loss).to(self.device)
        self.criterion_idt     = torch.nn.L1Loss()
        self.criterion_cycle   = torch.nn.L1Loss()
        self.gradient_penalty  = config.gradient_penalty

        if self.is_train:
            self.queues = NamedDict(**{
                name : FastQueue(head_queue_size, device = device)
                    for name in [
                        'real_a0', 'real_a1', 'real_b',
                        'fake_a0', 'fake_a1', 'fake_b'
                    ]
            })

            self.gp = None
            self.gp_cacher_a0 = None
            self.gp_cacher_a1 = None
            self.gp_cacher_b  = None

            if config.gradient_penalty is not None:
                self.gp = GradientPenalty(**config.gradient_penalty)

                self.gp_cacher_a0 = GradientCacher(
                    self.models.disc_a0, self.gp, gp_cache_period
                )
                self.gp_cacher_a1 = GradientCacher(
                    self.models.disc_a1, self.gp, gp_cache_period
                )
                self.gp_cacher_b = GradientCacher(
                    self.models.disc_b, self.gp, gp_cache_period
                )

    def split_domain_a_image(self, image_a):
        image_a0 = image_a[:, :self._c_a0, ...]
        image_a1 = image_a[:, self._c_a0:, ...]

        return (image_a0, image_a1)

    def merge_domain_a_images(self, image_a0, image_a1):
        # pylint: disable=no-self-use
        return torch.cat((image_a0, image_a1), dim = 1)

    def _set_input(self, inputs, domain):
        set_asym_two_domain_input(self.images, inputs, domain, self.device)

        if (
                (self.images.real_a0 is not None)
            and (self.images.real_a1 is not None)
        ):
            self.images.real_a = self.merge_domain_a_images(
                self.images.real_a0, self.images.real_a1
            )

        if not self.is_train:
            return

        if self.lambda_idt_bb and (self.images.real_b is not None):
            self.images.idt_bb_input = self.merge_domain_a_images(
                self.images.real_b, torch.zeros_like(self.images.real_a1)
            )

        if self.lambda_idt_aa and (
                (self.images.real_a0 is not None)
            and (self.images.real_a1 is not None)
        ):
            self.images.idt_aa_input = (
                self.images.real_a0 + self.images.real_a1
            )

    def cycle_forward_image(self, real, gen_fwd, gen_bkw):
        # pylint: disable=no-self-use

        # (N, C, H, W)
        if self.data_norm is not None:
            real = self.data_norm.normalize(real)

        fake = gen_fwd(real)
        reco = gen_bkw(fake)

        if self.data_norm is not None:
            fake = self.data_norm.denormalize(fake)
            reco = self.data_norm.denormalize(reco)

        return (fake, reco)

    def idt_forward_image(self, idt_input, gen):
        # pylint: disable=no-self-use
        if self.data_norm is not None:
            idt_input = self.data_norm.normalize(idt_input)

        idt = gen(idt_input)

        if self.data_norm is not None:
            idt = self.data_norm.denormalize(idt)

        return idt

    def forward_dispatch(self, direction):
        if direction == 'cyc-aba':
            (self.images.fake_b, self.images.reco_a) \
                = self.cycle_forward_image(
                    self.images.real_a, self.models.gen_ab, self.models.gen_ba
                )

            (self.images.reco_a0, self.images.reco_a1) \
                = self.split_domain_a_image(self.images.reco_a)

        elif direction == 'cyc-bab':
            (self.images.fake_a, self.images.reco_b) \
                = self.cycle_forward_image(
                    self.images.real_b, self.models.gen_ba, self.models.gen_ab
                )

            (self.images.fake_a0, self.images.fake_a1) \
                = self.split_domain_a_image(self.images.fake_a)

        elif direction == 'idt-bb':
            self.images.idt_bb = self.idt_forward_image(
                self.images.idt_bb_input, self.models.gen_ab
            )

        elif direction == 'idt-aa':
            self.images.idt_aa = self.idt_forward_image(
                self.images.idt_aa_input, self.models.gen_ba
            )

            (self.images.idt_aa_a0, self.images.idt_aa_a1) \
                = self.split_domain_a_image(self.images.idt_aa)

        elif direction == 'ema-cyc-aba':
            (self.images.fake_b, self.images.reco_a) \
                = self.cycle_forward_image(
                    self.images.real_a,
                    self.models.ema_gen_ab, self.models.ema_gen_ba
                )

            (self.images.reco_a0, self.images.reco_a1) \
                = self.split_domain_a_image(self.images.reco_a)

        elif direction == 'ema-cyc-bab':
            (self.images.fake_a, self.images.reco_b) \
                = self.cycle_forward_image(
                    self.images.real_b,
                    self.models.ema_gen_ba, self.models.ema_gen_ab
                )

            (self.images.fake_a0, self.images.fake_a1) \
                = self.split_domain_a_image(self.images.fake_a)

        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")

    def forward(self):
        if self.images.real_a is not None:
            if self.ema_momentum is not None:
                self.forward_dispatch(direction = 'ema-cyc-aba')
            else:
                self.forward_dispatch(direction = 'cyc-aba')

        if self.images.real_b is not None:
            if self.ema_momentum is not None:
                self.forward_dispatch(direction = 'ema-cyc-bab')
            else:
                self.forward_dispatch(direction = 'cyc-bab')

    def eval_loss_of_cycle_forward_aba(self):
        # NOTE: Queue is updated in discriminator backprop
        disc_pred_fake_b = queued_forward(
            self.models.disc_b, self.images.fake_b, self.queues.fake_b,
            self.data_norm, self._norm_disc_b, update_queue = False
        )

        self.losses.gen_ab = self.criterion_gan(
            disc_pred_fake_b, is_real = True, is_generator = True
        )

        self.losses.cycle_a0 = eval_loss_with_norm(
            self.criterion_cycle, self.images.reco_a0, self.images.real_a0,
            self.data_norm, self._norm_loss_a0
        )
        self.losses.cycle_a1 = eval_loss_with_norm(
            self.criterion_cycle, self.images.reco_a1, self.images.real_a1,
            self.data_norm, self._norm_loss_a1
        )

        return (
            self.lambda_adv_b * self.losses.gen_ab
            + 0.5 * self.lambda_cyc_a0 * self.losses.cycle_a0
            + 0.5 * self.lambda_cyc_a1 * self.losses.cycle_a1
        )

    def eval_loss_of_cycle_forward_bab(self):
        # NOTE: Queue is updated in discriminator backprop
        disc_pred_fake_a0 = queued_forward(
            self.models.disc_a0, self.images.fake_a0, self.queues.fake_a0,
            self.data_norm, self._norm_disc_a0, update_queue = False
        )
        disc_pred_fake_a1 = queued_forward(
            self.models.disc_a1, self.images.fake_a1, self.queues.fake_a1,
            self.data_norm, self._norm_disc_a1, update_queue = False
        )

        self.losses.gen_ba0 = self.criterion_gan(
            disc_pred_fake_a0, is_real = True, is_generator = True
        )
        self.losses.gen_ba1 = self.criterion_gan(
            disc_pred_fake_a1, is_real = True, is_generator = True
        )

        self.losses.cycle_b = eval_loss_with_norm(
            self.criterion_cycle, self.images.reco_b, self.images.real_b,
            self.data_norm, self._norm_loss_b
        )

        return (
              0.5 * self.lambda_adv_a0 * self.losses.gen_ba0
            + 0.5 * self.lambda_adv_a1 * self.losses.gen_ba1
            + self.lambda_cyc_b * self.losses.cycle_b
        )

    def eval_loss_of_idt_forward_bb(self):
        self.losses.idt_bb = eval_loss_with_norm(
            self.criterion_idt, self.images.idt_bb, self.images.real_b,
            self.data_norm, self._norm_loss_b
        )

        return self.lambda_idt_bb * self.lambda_cyc_b * self.losses.idt_bb

    def eval_loss_of_idt_forward_aa(self):
        target_a0 = self.images.real_a0
        target_a1 = self.images.real_a1

        self.losses.idt_aa_a0 = eval_loss_with_norm(
            self.criterion_idt, self.images.idt_aa_a0, target_a0,
            self.data_norm, self._norm_loss_a0
        )
        self.losses.idt_aa_a1 = eval_loss_with_norm(
            self.criterion_idt, self.images.idt_aa_a1, target_a1,
            self.data_norm, self._norm_loss_a1
        )

        # * 0.5 to account for 2 losses ist_aa_a{0,1} relative to bb
        return 0.5 * self.lambda_idt_aa * (
              self.lambda_cyc_a0 * self.losses.idt_aa_a0
            + self.lambda_cyc_a1 * self.losses.idt_aa_a1
        )

    def backward_gen(self, direction):
        if direction == 'cyc-aba':
            loss = self.eval_loss_of_cycle_forward_aba()

        elif direction == 'cyc-bab':
            loss = self.eval_loss_of_cycle_forward_bab()

        elif direction == 'idt-bb':
            loss = self.eval_loss_of_idt_forward_bb()

        elif direction == 'idt-aa':
            loss = self.eval_loss_of_idt_forward_aa()

        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")

        loss.backward()

    def backward_discriminator_base(
        self, model, real, fake, queue_real, queue_fake, gp_cacher, scale,
        normalize
    ):
        # pylint: disable=too-many-arguments
        loss_gp = None

        if self.gp is not None:
            # DDP: the double backward of the gradient penalty does not
            # reach every parameter, so its gradient is accumulated locally
            # and all-reduced by the synchronized backward below.
            with no_sync(model):
                loss_gp = gp_cacher(
                    model, fake, real,
                    model_kwargs_fake = {
                        'extra_bodies' : queue_fake.query()
                    },
                    model_kwargs_real = {
                        'extra_bodies' : queue_real.query()
                    },
                )

        pred_real = queued_forward(
            model, real, queue_real, self.data_norm, normalize,
            update_queue = True
        )
        loss_real = self.criterion_gan(
            pred_real, is_real = True, is_generator = False
        )

        pred_fake = queued_forward(
            model, fake, queue_fake, self.data_norm, normalize,
            update_queue = True
        )
        loss_fake = self.criterion_gan(
            pred_fake, is_real = False, is_generator = False
        )

        loss = (loss_real + loss_fake) * 0.5 * scale
        loss.backward()

        return (loss_gp, loss)

    def backward_discriminators(self):
        fake_a0 = self.images.fake_a0.detach()
        fake_a1 = self.images.fake_a1.detach()
        fake_b  = self.images.fake_b .detach()

        loss_gp_b, self.losses.disc_b \
            = self.backward_discriminator_base(
                self.models.disc_b, self.images.real_b, fake_b,
                self.queues.real_b, self.queues.fake_b, self.gp_cacher_b,
                self.lambda_adv_b, self._norm_disc_b
            )

        if loss_gp_b is not None:
            self.losses.gp_b = loss_gp_b

        loss_gp_a0, self.losses.disc_a0 = \
            self.backward_discriminator_base(
                self.models.disc_a0, self.images.real_a0, fake_a0,
                self.queues.real_a0, self.queues.fake_a0, self.gp_cacher_a0,
                0.5 * self.lambda_adv_a0, self._norm_disc_a0
            )

        if loss_gp_a0 is not None:
            self.losses.gp_a0 = loss_gp_a0

        loss_gp_a1, self.losses.disc_a1 = \
            self.backward_discriminator_base(
                self.models.disc_a1, self.images.real_a1, fake_a1,
                self.queues.real_a1, self.queues.fake_a1, self.gp_cacher_a1,
                0.5 * self.lambda_adv_a1, self._norm_disc_a1
            )

        if loss_gp_a1 is not None:
            self.losses.gp_a1 = loss_gp_a1

    def optimization_step_gen(self):
        discs = [ self.models.disc_a0, self.models.disc_a1, self.models.disc_b ]

        self.set_requires_grad(discs, False)
        self.optimizers.gen.zero_grad(set_to_none = True)

        dir_list = [ 'cyc-aba', 'cyc-bab' ]

        if self.lambda_idt_bb:
            dir_list += [ 'idt-bb' ]

        if self.lambda_idt_aa:
            dir_list += [ 'idt-aa' ]

        # DDP: the frozen discriminators must never synchronize, and each
        # generator synchronizes only on its last use, all-reducing the
        # gradient accumulated over the whole step at once.
        gens_used = {
            'cyc-aba' : (self.models.gen_ab, self.models.gen_ba),
            'cyc-bab' : (self.models.gen_ab, self.models.gen_ba),
            'idt-bb'  : (self.models.gen_ab, ),
            'idt-aa'  : (self.models.gen_ba, ),
        }

        last_use = {}
        for (idx, direction) in enumerate(dir_list):
            for gen in gens_used[direction]:
                last_use[id(gen)] = idx

        for (idx, direction) in enumerate(dir_list):
            unsynced = discs + [
                gen for gen in gens_used[direction]
                    if last_use[id(gen)] != idx
            ]

            with no_sync(*unsynced):
                self.forward_dispatch(direction)
                self.backward_gen(direction)

        all_reduce_gradients(self.optimizers.gen)
        clip_gradients(self.optimizers.gen, **self._grad_clip)
        self.optimizers.gen.step()

    def optimization_step_disc(self):
        self.set_requires_grad(
            [self.models.disc_a0, self.models.disc_a1, self.models.disc_b],
            True
        )
        self.optimizers.disc.zero_grad(set_to_none = True)

        self.backward_discriminators()

        all_reduce_gradients(self.optimizers.disc)
        clip_gradients(self.optimizers.disc, **self._grad_clip)
        self.optimizers.disc.step()

    def _accumulate_averages(self):
        update_average_model(
            self.models.ema_gen_ab, self.models.gen_ab, self.ema_momentum
        )
        update_average_model(
            self.models.ema_gen_ba, self.models.gen_ba, self.ema_momentum
        )

    def optimization_step(self):
        self.optimization_step_gen()
        self.optimization_step_disc()

        if self.ema_momentum is not None:
            self._accumulate_averages()

