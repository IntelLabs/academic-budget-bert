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
class PreTrainDatasetArguments:
    """
    PretrainDataArguments
    """

    _argument_group_name = "Dataset Arguments"

    dataset_path: Optional[str] = field(
        default="books_wiki_en_corpus", metadata={"help": "pretrain_dataset"}
    )
    num_workers: Optional[int] = field(default=4, metadata={"help": "num of dataloader workers"})

    async_worker: Optional[bool] = field(default=True, metadata={"help": "async_worker"})

    data_loader_type: Optional[str] = field(
        default="per_device",
        metadata={
            "help": "Dataloader to use: dist=distributed, per_device=local per device",
            "choices": ["dist", "per_device"],
        },
    )
