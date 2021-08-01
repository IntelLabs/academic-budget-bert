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
import logging
import os

from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def write_shard(lines, f_idx, out_dir_path, name=None):
    os.makedirs(out_dir_path, exist_ok=True)
    filename = f"shard_{f_idx}.txt"
    if name is not None and len(name) > 0:
        filename = f"{name}_{filename}"
    with open(os.path.join(out_dir_path, filename), "w") as fw:
        for l in lines:
            fw.write(l)


def list_files_in_dir(dir, data_prefix=".txt", file_name_grep=""):
    dataset_files = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f)) and data_prefix in f and file_name_grep in f
    ]
    return dataset_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="input directory with sharded text files", required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--ratio",
        type=int,
        default=1,
        help="Number of files to merge into a single shard",
    )
    parser.add_argument(
        "--grep",
        type=str,
        default="",
        help="A string to filter a subset of files from input directory",
    )

    args = parser.parse_args()
    dataset_files = list_files_in_dir(args.data, file_name_grep=args.grep)
    num_files = len(dataset_files)
    assert (
        num_files % args.ratio == 0
    ), f"{num_files} % {args.ratio} != 0, make sure equal shards in each merged file"
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Compacting input shards into {args.output_dir}")

    # merge input directory shards into num_files_shard
    file_lines = []
    f_idx = 0
    lines_idx = 0
    for f in tqdm(dataset_files, smoothing=1):
        with open(f) as fp:
            file_lines.extend(fp.readlines())
        if lines_idx == args.ratio - 1:
            write_shard(file_lines, f_idx, args.output_dir, name=args.grep)
            file_lines = []
            f_idx += 1
            lines_idx = 0
            continue
        lines_idx += 1
    logger.info("Done!")
