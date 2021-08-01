# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# code taken from commit: 35b4582486fe096a5c669b6ca35a3d5c6a1ec56b
# https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert
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

import json
import logging
import os

from transformers import BertTokenizer, RobertaTokenizer

from pretraining.configs import PretrainedBertConfig, PretrainedRobertaConfig
from pretraining.modeling import BertForPreTraining, BertLMHeadModel
from pretraining.utils import to_sanitized_dict

logger = logging.getLogger(__name__)


MODELS = {
    "bert-mlm": (BertLMHeadModel, PretrainedBertConfig, BertTokenizer),
    "bert-mlm-roberta": (BertLMHeadModel, PretrainedRobertaConfig, RobertaTokenizer),
    "bert-mlm-nsp": (BertForPreTraining, PretrainedBertConfig, BertTokenizer),
}


class BasePretrainModel(object):
    def __init__(
        self,
        args,
        model_type=None,
        model_name_or_path=None,
        tokenizer=None,
        config=None,
        model_kwargs={},
    ):
        if not model_type:
            # getting default model type from args
            model_type = args.model_type
        assert model_type in MODELS, f"model_type {model_type} is not supported"
        model_cls, config_cls, token_cls = MODELS[model_type]

        self.args = args
        self.ds_file = args.ds_config if hasattr(args, "ds_config") else None

        if not tokenizer:
            if model_name_or_path is None:
                loading_path = args.tokenizer_name
                logger.info(f"Loading default tokenizer {loading_path}")
            else:
                loading_path = model_name_or_path
            tokenizer = token_cls.from_pretrained(loading_path)

        if not config:
            if model_name_or_path is None:
                logger.info(f"Loading config from args")
                config = config_cls(**args.model_config)
                config = self._init_vocab_size(config)
            else:
                config = config_cls.from_pretrained(model_name_or_path)

        self.args.vocab_size = config.vocab_size

        self.tokenizer = tokenizer
        self.config = config
        self.network = model_cls(self.config, self.args, **model_kwargs)

    def forward(self, batch):
        outputs = self.network(batch)
        return outputs[0]  # return train=loss or infer=prediction scores

    @staticmethod
    def _init_vocab_size(config):
        # Padding for divisibility by 8
        if config.vocab_size % 8 != 0:
            config.vocab_size += 8 - (config.vocab_size % 8)
        logger.info(f"VOCAB SIZE: {config.vocab_size}")
        return config

    def save_weights(self, checkpoint_id, output_dir, is_deepspeed=False) -> str:
        """Save model weights, config and tokenizer configurations + extra arguments"""
        checkpoint_dir = os.path.join(output_dir, checkpoint_id)
        logger.info("checkpointing: PATH={}".format(checkpoint_dir))
        os.makedirs(checkpoint_dir, exist_ok=True)

        if is_deepspeed:
            # deepspeed save method
            self.network.module.save_pretrained(checkpoint_dir)
            # save Deepspeed config and running args (for future use)
            ds_config_path = os.path.join(checkpoint_dir, "deepspeed_config.json")
            self.to_json_file(self.args.ds_config, ds_config_path)
            self.args.deepspeed_config = ds_config_path
        else:
            # non deepspeed saving method
            self.network.save_pretrained(checkpoint_dir)

        self.config.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        args_file_path = os.path.join(checkpoint_dir, "args.json")
        args_dict = to_sanitized_dict(self.args)
        self.to_json_file(args_dict, args_file_path)

        return checkpoint_dir

    @classmethod
    def to_json_file(cls, dict_object, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(cls.to_json_string(dict_object))

    @staticmethod
    def to_json_string(dict_object):
        return json.dumps(dict_object, indent=2, sort_keys=True) + "\n"

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def prepare_optimizer_parameters(self, weight_decay):
        param_optimizer = list(self.network.named_parameters())
        param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
