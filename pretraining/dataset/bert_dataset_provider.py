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


class BertDatasetProviderInterface:
    def get_shard(self, index, shuffle=True):
        raise NotImplementedError

    def release_shard(self, index):
        raise NotImplementedError

    def prefetch_shard(self, index):
        raise NotImplementedError

    def get_batch(self, batch_iter):
        raise NotImplementedError

    def prefetch_batch(self):
        raise NotImplementedError
