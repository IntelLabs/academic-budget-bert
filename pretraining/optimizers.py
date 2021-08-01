# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from torch.optim import Adam, AdamW
from transformers.optimization import Adafactor

logger = logging.getLogger(__name__)


def get_lamb(optimizer_args, lr, model_params):
    try:
        import deepspeed
    except ImportError or ModuleNotFoundError:
        logger.info("Deepspeed not installed. To use Lamb optimizer please install Deepspeed")
        raise

    from deepspeed.ops.lamb import FusedLamb

    # DS optimizer name hack
    class lamb(FusedLamb):
        pass

    return lamb(
        model_params,
        lr=lr,
        bias_correction=optimizer_args.bias_correction,
        weight_decay=optimizer_args.weight_decay,
        max_coeff=optimizer_args.max_coeff,
        min_coeff=optimizer_args.min_coeff,
    )


def get_adafactor(args, lr, params):
    return Adafactor(params, lr=lr, relative_step=False, scale_parameter=False)


def get_adam(args, lr, params):
    return Adam(
        params,
        lr=lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )


def get_adamw(args, lr, params):
    return AdamW(
        params,
        lr=lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )


OPTIMIZERS = {
    "adam": get_adam,
    "adamw": get_adamw,
    "adafactor": get_adafactor,
    "lamb": get_lamb,
}


def get_optimizer(args, lr, params):
    optimizer_type = args.optimizer_type
    if optimizer_type not in OPTIMIZERS:
        raise Exception(
            f"{optimizer_type} is not available. Please choose one of the following: {list(OPTIMIZERS.keys())}"
        )

    return OPTIMIZERS[optimizer_type](args, lr, params)
