# Stitching pretrained models
## 1. Split enwiki books dataset
run the following command
```bash
python workspace/split_data.py --data-path {path to enwiki books} --split 4
```

## 2. Run pretraining
### Clone 24H bert
```bash
bash scripts/run_pretrain_clone.sh
```
* change `--dataset_path` and `--output_dir`
* wandb group is set to `{job_name}-{current_run_id}`
* to load tokenizer locally, add `--load_tokenizer_locally`
* NOTE: `--max_steps_per_epoch` is set to sync the number of steps of each gpu. Not necessary if using only one gpu
  
### Pretrain half-large models
* hidden size and the number of attention heads are half of bert-large
* model parameters are roughly the quarter of bert-large
  
```bash
bash scripts/run_pretrain_halflarge.sh
```
* change `--dataset_path` and `--current_run_id` to train on only subset of training set.

### Pretrain stitched models
```bash
bash scripts/run_pretrain_2xhalflarge.sh
```
* change `--lr_schedule`, `--curve`, `--warmup_proportion` to try different learning rate setups.
* change `--dataset_path` to train on different training sets
* set `--do_stitch` for stitching and pass two source model paths as `--src_model1_path` and `--src_model2_path`