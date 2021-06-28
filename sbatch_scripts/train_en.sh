#!/bin/bash
#
#SBATCH --job-name=cpc
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=64000
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/train_en.out

source activate cpc37

python -u cpc/train.py --pathDB ~/data/common_voices/en/clips_16k/ --pathTrain data_splits/common_voices_splits/en/trainSeqs_all_16k.txt --file_extension .mp3 --pathCheckpoint checkpoints_en/ --n_process_loader 8 --restart --ignore_cache

# verify: --restart or continue training?
