import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm

# default args
data_path = "/n/tata_ddos_ceph/woojeong/data/enwiki_books_128_20"
split = 4


def parse_args():
    parser = argparse.ArgumentParser()
    # path to the whole data
    parser.add_argument("--data-path", default=data_path, type=str)
    # the number of splits
    parser.add_argument("--split", default=split, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # all files
    src_path = os.path.join(args.data_path, "total")
    file_list = os.listdir(src_path)

    # check duplicates
    index_list = sorted(
        [int(filename.replace(".hdf5", "").split("_")[2]) for filename in file_list]
    )
    duplicates = [index for index in index_list if index_list.count(index) > 1]

    # train - 2560, test - 64
    train_count = len([filename for filename in file_list if "train" in filename])
    test_count = len([filename for filename in file_list if "test" in filename])

    train_splits = np.array_split(range(train_count), split)
    test_splits = np.array_split(range(test_count), split)

    # make dirs and copy files
    for i in range(args.split):
        # make dirs
        dst_path = os.path.join(args.data_path, f"set{i}")
        os.makedirs(dst_path, exist_ok=True)
        print(f"created {dst_path} directory")

        # copy train shards
        for train_file in tqdm(train_splits[i]):
            shutil.copy(
                os.path.join(src_path, f"train_shard_{train_file}.hdf5"), dst_path
            )
            # print(f"train_shard_{train_file}.hdf5 file copied")

        # copy train shards
        for test_file in tqdm(test_splits[i]):
            shutil.copy(
                os.path.join(src_path, f"test_shard_{test_file}.hdf5"), dst_path
            )
            # print(f"test_shard_{test_file}.hdf5 file copied")
