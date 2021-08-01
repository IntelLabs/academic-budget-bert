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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SchedulerArgs:
    """
    Scheduler Arguments
    """

    _argument_group_name = "Scheduler Arguments"

    lr_schedule: Optional[str] = field(
        default="time",
        metadata={
            "help": "learning rate scheduler type (step/constant_step/time",
            "choices": ["step", "constant_step", "time"],
        },
    )

    curve: Optional[str] = field(
        default="linear",
        metadata={
            "help": "curve shape (linear/exp)",
            "choices": ["linear", "exp"],
        },
    )

    warmup_proportion: Optional[float] = field(default=0.06, metadata={"help": "Warmup proportion"})
    decay_rate: Optional[float] = field(default=0.99, metadata={"help": "Decay rate"})
    decay_step: Optional[int] = field(default=1000, metadata={"help": "Decay step"})
    num_warmup_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of warmup steps"}
    )
