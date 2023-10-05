# Exponential Moving Average (EMA) of model updates
# References:
# Timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py

from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

class ModelEma:
    """ Model Exponential Moving Average (DEPRECATED)
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This version is deprecated, it does not work with scripted models. Will be removed eventually.
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device='', resume='', devices=None):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        self.ema_has_module = hasattr(self.ema, 'module')
        if device and not self.ema_has_module:
            self.ema.to(device=device)
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        

    def _load_checkpoint(self, checkpoint):
        assert isinstance(checkpoint, dict)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            # ema model may have been wrapped by DataParallel, and need module prefix
            if self.ema_has_module:
                name = 'module.' + k if not k.startswith('module') else k
            else:
                name = k
            new_state_dict[name] = v
        self.ema.load_state_dict(new_state_dict)
        print("Loaded state_dict_ema")


    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
            