#!/bin/bash
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate b2

python ./scripts/training/train_pipeline.py sample.batch_size=1 train.batch_size=1 sample.num_batches_per_epoch=1 