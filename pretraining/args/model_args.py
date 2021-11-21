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
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Model configuration arguments
    """

    _argument_group_name = "Model Arguments"

    model_type: Optional[str] = field(default="bert-mlm", metadata={"help": "bert_model_type"})

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )
    pretrain_run_args: Optional[str] = field(
        default=None, metadata={"help": "Pre-training run command (used by run_glue.py)"}
    )


@dataclass
class ModelConfigArguments:
    """
    Model topology configuration arguments
    """

    _argument_group_name = "Model Config Arguments"

    vocab_size: Optional[int] = field(
        default=30522, metadata={"help": "vocab size (BERT=30522, RoBERTa=50265"}
    )

    hidden_size: Optional[int] = field(default=1024, metadata={"help": "hidden_size"})

    num_hidden_layers: Optional[int] = field(default=24, metadata={"help": "num_hidden_layers"})

    num_attention_heads: Optional[int] = field(default=16, metadata={"help": "num_attention_heads"})

    intermediate_size: Optional[int] = field(default=4096, metadata={"help": "intermediate_size"})

    hidden_act: Optional[str] = field(default="gelu", metadata={"help": "hidden_act: [gelu, relu]"})

    hidden_dropout_prob: Optional[float] = field(
        default=0.1, metadata={"help": "hidden_dropout_prob"}
    )

    attention_probs_dropout_prob: Optional[float] = field(
        default=0.1, metadata={"help": "attention_probs_dropout_prob"}
    )

    max_position_embeddings: Optional[int] = field(
        default=512, metadata={"help": "max_position_embeddings"}
    )

    layernorm_embedding: Optional[bool] = field(
        default=False, metadata={"help": "add layernorm to embedding layer"}
    )

    type_vocab_size: Optional[int] = field(default=2, metadata={"help": "type_vocab_size"})

    initializer_range: Optional[float] = field(default=0.02, metadata={"help": "initializer_range"})

    fused_linear_layer: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use fused linear + activation + bias add",
        },
    )

    sparse_mask_prediction: Optional[bool] = field(
        default=True,
        metadata={"help": "Use sparse MLM prediction. i.e., predict only on masked words"},
    )

    encoder_ln_mode: Optional[str] = field(
        default="pre-ln",
        metadata={
            "help": "Layer Normalization position (PreLN/PostLN)",
            "choices": ["pre-ln", "post-ln"],
        },
    )

    layer_norm_type: Optional[str] = field(
        default="apex",
        metadata={
            "help": "LayerNorm type used in the model.",
            "choices": ["pytorch", "apex", "rms_norm"],
        },
    )
