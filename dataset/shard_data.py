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

import argparse
import pathlib
import os
from os import listdir
from os.path import join

from data import TextSharding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to dataset files (1 file per dataset)"
    )
    parser.add_argument(
        "-o", type=str, required=True, help="Output directory where the shard files will be written"
    )
    parser.add_argument(
        "--num_train_shards", type=int, default=256, help="Number of training shards"
    )
    parser.add_argument("--num_test_shards", type=int, default=256, help="Number of test shards")
    parser.add_argument(
        "--frac_test",
        type=float,
        default=0.1,
        help="Fraction of dataset to reserve for the test set",
    )

    args = parser.parse_args()

    input_files = [join(args.dir, f) for f in listdir(args.dir) if f.endswith(".txt")]
    shards_dir = pathlib.Path(args.o)
    print(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    segmenter = TextSharding.NLTKSegmenter()
    sharding = TextSharding.Sharding(
        input_files,
        str(shards_dir.absolute()) + os.sep,
        args.num_train_shards,
        args.num_test_shards,
        args.frac_test,
    )
    sharding.load_articles()
    sharding.segment_articles_into_sentences(segmenter)
    sharding.distribute_articles_over_shards()
    sharding.write_shards_to_disk()
