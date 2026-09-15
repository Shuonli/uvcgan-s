from uvcgan_s.base.weight_init   import init_weights
from uvcgan_s.torch.distributed  import wrap_model
from uvcgan_s.torch.lr_equal     import apply_lr_equal
from uvcgan_s.torch.spectr_norm  import apply_sn

def default_model_init(model, model_config, device, wrap = True):
    model = model.to(device)
    init_weights(model, model_config.weight_init)

    if model_config.lr_equal:
        apply_lr_equal(model)

    if model_config.spectr_norm:
        apply_sn(model)

    if not wrap:
        return model

    # NOTE: DDP has to wrap the final module, i.e. after parametrizations
    #       have been registered, otherwise its gradient hooks bind to
    #       parameters that no longer exist.
    return wrap_model(model)
