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

import threading
import queue


class AsyncWorker(threading.Thread):
    def __init__(self, dataloaders, dataset_picker):
        threading.Thread.__init__(self)
        self.req_queue = queue.Queue()
        self.ret_queue = queue.Queue()
        self.dataloaders = dataloaders
        self.dataset_picker = dataset_picker
        self.prefetch_idx = 3
        for i in range(self.prefetch_idx):
            self.req_queue.put(dataset_picker[i])

    def run(self):
        while True:
            dataset_type = self.req_queue.get(block=True)
            if dataset_type is None:
                break
            batch = next(self.dataloaders[dataset_type])
            self.req_queue.task_done()
            self.ret_queue.put(batch)

    def get(self):
        batch = self.ret_queue.get()
        self.ret_queue.task_done()
        return batch

    def prefetch(self):
        if self.prefetch_idx < len(self.dataset_picker):
            self.req_queue.put(self.dataset_picker[self.prefetch_idx])
            self.prefetch_idx += 1

    def stop(self):
        self.req_queue.put(None)
