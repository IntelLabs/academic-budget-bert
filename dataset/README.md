# Dataset Processing

Preparing the dataset includes the following steps:

- Obtain textual data
- Process dataset (wikipedia or bookcorpus) and combine into 1 text file using `process_data.py`
- Divide the data into N shards using `shard_data.py`
- Generate samples for training and testing the model using `generate_samples.py`

## Obtaining textual data

Any textual dataset can be processed and used for training a BERT-like model.

In our experiments we trained models using the English section of Wikipedia and the Toronto Bookcorpus [REF].

Wikipedia dumps can be freely downloaded from https://dumps.wikimedia.org/ and can be processed (removing HTML tags, picture, and non textual data) using [Wikiextractor.py](https://github.com/attardi/wikiextractor).

We are unable to provide a source for Bookcorpus dataset.

## Data Processing

Use `process_data.py` for pre-processing wikipedia/bookcorpus datasets into a single text file.

See `python process_data.py -h` for the full list of options.

An example for pre-processing the English Wikipedia xml dataset:

```bash
python process_data.py -f <path_to_xml> -o <output_dir> --type wiki
```

An example for pre-processing the Bookcorpus dataset:

```bash
python process_data.py -f <path_to_text_files> -o <output_dir> --type bookcorpus
```

## Initial Sharding

Use `shard_data.py` to shard multiple text files (processed with the script above) into pre-defined number of shards, and to divide the dataset into train and test sets.

IMPORTANT NOTE: the number of shards is affected by the duplication factor used when generating the samples (with masked tokens). This means that if 10 training shards are generated with `shard_data.py` and samples are generated with duplication factor 5, the final number of training shards will be 50.
This approach avoids intra-shard duplications that might overfit the model in each epoch.

IMPORTATN NOTE 2: the performance of the sharding script (we might fix in the future) might be slow if you choose to generate a small amount of shards (from our experiment under 100). If you encounter such situation we recommand to generate 256+ shards and then merging them to fewer using the merging script we provide (`merge_shards.py`). See more info the next section.

See `python shard_data.py -h` for the full list of options.

Example for sharding all texts found in the input `--dir` into `256` train shards and `128` test shard, with 10% of the samples held-out for the test set:

```bash
python shard_data.py \
    --dir <path_to_text_files> \
    -o <output_dir> \
    --num_train_shards 256 \
    --num_test_shards 128 \
    --frac_test 0.1
```

## Merging Shards (optional)

Merging existing shards into fewer shards (while maintaining 2^N shards, for example 256->128 (2:1 ratio)) can be done with `merge_shards.py` script.

See `python merge_shards.py -h` for the full list of options.

Example for merging randomly 2 shards into 1 shard:

```bash
python merge_shards.py \
    --data <path_to_shards_dir> \
    --output_dir <output_dir> \
    --ratio 2 
```

## Samples Generation

Use `generate_samples.py` for generating samples compatible with dataloaders used in the training script.

IMPORTANT NOTE: the duplication factor chosen will multiply the number of final shards by its factor. For example, 10 shards with duplication factor 5 will generate 50 shards (each shard with different randomly generated (masked) samples).

See `python generate_samples.py -h` for the full list of options.

Example for generating shards with duplication factor 10, lowercasing the tokens, masked LM probability of 15%, max sequence length of 128, tokenizer by provided (Huggingface compatible) model named `bert-large-uncased`, max predictions per sample 20 and 16 parallel processes (for processing faster):

```bash
python generate_samples.py \
    --dir <path_to_shards> \
    -o <output_path> \
    --dup_factor 10 \
    --seed 42 \
    --vocab_file <path_to_vocabulary_file> \
    --do_lower_case 1 \
    --masked_lm_prob 0.15 \ 
    --max_seq_length 128 \
    --model_name bert-large-uncased \
    --max_predictions_per_seq 20 \
    --n_processes 16
```