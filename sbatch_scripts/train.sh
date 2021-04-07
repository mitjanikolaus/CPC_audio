#!/bin/bash
#
#SBATCH --job-name=cpc
#
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64000
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=cuda75|cuda61
#SBATCH --output=out/train.out

source activate cpc37

python -u cpc/train.py --pathDB ~/data/librispeech/train-clean-100/LibriSpeech/train-clean-100/ --pathTrain data_splits/train_split.txt   --debug

