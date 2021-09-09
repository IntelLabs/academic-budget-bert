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
import subprocess

logger = logging.getLogger()


def list_files_in_dir(dir, data_prefix=".txt"):
    dataset_files = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f)) and data_prefix in f
    ]
    return dataset_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to shards dataset")
    parser.add_argument("-o", type=str, required=True, help="Output directory")

    parser.add_argument("--dup_factor", type=int, default=1, help="sentence duplication factor")
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--vocab_file", type=str, help="vocab file")
    parser.add_argument(
        "--do_lower_case", type=int, default=1, help="lower case True = 1, False = 0"
    )
    parser.add_argument(
        "--masked_lm_prob", type=float, help="Specify the probability for masked lm", default=0.15
    )
    parser.add_argument(
        "--max_seq_length", type=int, help="Specify the maximum sequence length", default=512
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Pre-trained models name (HF format): bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, roberta-base, roberta-large",
    )

    parser.add_argument(
        "--max_predictions_per_seq",
        type=int,
        help="Specify the maximum number of masked words per sequence",
        default=20,
    )
    parser.add_argument("--n_processes", type=int, default=8, help="number of parallel processes")

    args = parser.parse_args()

    # for each shard -> call prepare_data.py  x duplicated factor

    shard_files = list_files_in_dir(args.dir)
    # new_shards_output = os.path.join(args.o, "generated_hdf5_samples")
    new_shards_output = args.o
    os.makedirs(new_shards_output, exist_ok=True)

    logger.info("Creating new hdf5 files ...")

    def create_shard(f_path, shard_idx, set_group, args):
        output_filename = os.path.join(new_shards_output, f"{set_group}_shard_{shard_idx}.hdf5")
        if "roberta" in args.model_name:
            hdf5_preprocessing_cmd = "python data/create_pretraining_data_roberta.py"
        else:
            hdf5_preprocessing_cmd = "python data/create_pretraining_data.py"
        hdf5_preprocessing_cmd += f" --input_file={f_path}"
        hdf5_preprocessing_cmd += f" --output_file={output_filename}"
        hdf5_preprocessing_cmd += (
            f" --vocab_file={args.vocab_file}" if args.vocab_file is not None else ""
        )
        hdf5_preprocessing_cmd += (
            f" --bert_model={args.model_name}" if args.model_name is not None else ""
        )
        hdf5_preprocessing_cmd += f" --do_lower_case" if args.do_lower_case else ""
        hdf5_preprocessing_cmd += f" --max_seq_length={args.max_seq_length}"
        hdf5_preprocessing_cmd += f" --max_predictions_per_seq={args.max_predictions_per_seq}"
        hdf5_preprocessing_cmd += f" --masked_lm_prob={args.masked_lm_prob}"
        hdf5_preprocessing_cmd += f" --random_seed={args.seed + shard_idx}"
        hdf5_preprocessing_cmd += f" --dupe_factor=1"
        hdf5_preprocessing_cmd += f" --no_nsp"

        bert_preprocessing_process = subprocess.Popen(hdf5_preprocessing_cmd, shell=True)

        last_process = bert_preprocessing_process

        # This could be better optimized (fine if all take equal time)
        if shard_idx % args.n_processes == 0 and shard_idx > 0:
            bert_preprocessing_process.wait()
        return last_process

    last_process = None
    shard_idx = {"train": 0, "test": 0}
    for dup_idx in range(args.dup_factor):
        for f in shard_files:
            file_name = os.path.basename(f)
            set_group = "train" if "train" in file_name else "test"
            last_process = create_shard(f, shard_idx[set_group], set_group, args)
            shard_idx[set_group] += 1
    last_process.wait()
