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

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DeepspeedArguments:
    """
    DeepspeedArguments
    """

    _argument_group_name = "Deepspeed Arguments"

    deepspeed: Optional[bool] = field(default=False, metadata={"help": "Use deepspeed."})

    deepspeed_transformer_kernel: Optional[bool] = field(
        default=False, metadata={"help": "Use DeepSpeed transformer kernel to accelerate."}
    )

    stochastic_mode: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use stochastic mode for high-performance transformer kernel.",
        },
    )

    attention_dropout_checkpoint: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use DeepSpeed transformer kernel memory optimization to checkpoint dropout output.",
        },
    )

    normalize_invertible: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use DeepSpeed transformer kernel memory optimization to perform invertible normalize backpropagation.",
        },
    )

    gelu_checkpoint: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use DeepSpeed transformer kernel memory optimization to checkpoint GELU activation.",
        },
    )

    gradient_clipping: Optional[float] = field(
        default=1.0, metadata={"help": "Gradient clipping (default 1.0)"}
    )

    steps_per_print: Optional[int] = field(
        default=100, metadata={"help": "Number of steps between training steps print"}
    )

    wall_clock_breakdown: Optional[bool] = field(
        default=False, metadata={"help": "wall clock breakdown"}
    )

    prescale_gradients: Optional[bool] = field(
        default=False, metadata={"help": "wall clock breakdown"}
    )

    gradient_predivide_factor: Optional[int] = field(
        default=None, metadata={"help": "gradient predivide factor"}
    )

    fp16: Optional[bool] = field(default=False, metadata={"help": "Enable FP16 training"})
    fp16_backend: Optional[str] = field(
        default="ds",
        metadata={
            "choices": ["ds", "apex"],
            "help": "mixed precision backend (ds=deepspeed, apex=nvidia apex)",
        },
    )
    fp16_opt: Optional[str] = field(default="O2", metadata={"help": "Apex optimization level"})

    def __post_init__(self):
        if self.deepspeed_transformer_kernel:
            remove_cuda_compatibility_for_kernel_compilation()


def remove_cuda_compatibility_for_kernel_compilation():
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        del os.environ["TORCH_CUDA_ARCH_LIST"]
