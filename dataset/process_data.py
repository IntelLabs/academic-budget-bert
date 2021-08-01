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
import pathlib
import subprocess

from data.BookscorpusTextFormatting import BookscorpusTextFormatting
from data.WikicorpusTextFormatting import WikicorpusTextFormatting

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

try:
    from wikiextractor import WikiExtractor
except ModuleNotFoundError as e:
    logger.error("wikiextractor is not installed, please install to use script")
    quit()

WIKI_EXT_CMD = "python -m wikiextractor.WikiExtractor"

FORMATTERS = {"wiki": WikicorpusTextFormatting, "bookcorpus": BookscorpusTextFormatting}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", type=str, required=True, help="Path to wikipedia xml or bookcorpus directory"
    )
    parser.add_argument("-o", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=FORMATTERS.keys(),
        help="Dataset type [wiki, bookcorpus]",
    )
    parser.add_argument(
        "--n_processes", type=int, default=16, help="Number of concurrent processes"
    )
    args = parser.parse_args()

    merged_file = pathlib.Path(args.o, f"{args.type}_one_article_per_line.txt")
    fmt = FORMATTERS.get(args.type)

    if args.type == "wiki":
        data_path = pathlib.Path(args.o, args.type)
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting articles using wikiextractor ...")
        EXTRACT_CMD = f"{WIKI_EXT_CMD} {args.f} -b 100M --processes {args.n_processes} -o {data_path.absolute()}"
        subprocess.run(EXTRACT_CMD, shell=True, check=True)
        logger.info("Done. \n")
    elif args.type == "bookcorpus":
        data_path = pathlib.Path(args.f)

    logger.info(f"Loading {args.type} files and combining into 1 file ...")
    data_formatter = fmt(str(data_path), str(merged_file), recursive=True)
    data_formatter.merge()
    logger.info("Done.")
