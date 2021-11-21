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

from transformers import BertConfig


class PretrainedBertConfig(BertConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        encoder_ln_mode="post-ln",
        fused_linear_layer=True,
        sparse_mask_prediction=True,
        layer_norm_type="apex",
        layernorm_embedding=False,
        **kwargs
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            initializer_range,
            layer_norm_eps,
            pad_token_id,
            gradient_checkpointing,
            **kwargs
        )
        self.useLN = True
        self.encoder_ln_mode = encoder_ln_mode
        self.fused_linear_layer = fused_linear_layer
        self.sparse_mask_prediction = sparse_mask_prediction
        self.layer_norm_type = layer_norm_type
        self.layernorm_embedding = layernorm_embedding


class PretrainedRobertaConfig(PretrainedBertConfig):
    model_type = "roberta"

    def __init__(self, vocab_size=50265, **kwargs):
        super().__init__(vocab_size=vocab_size, **kwargs)
