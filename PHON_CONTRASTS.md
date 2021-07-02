# Evaluation of phonetic contrasts

## Preprocessing

Download EN and FR [common voices datasets](https://commonvoice.mozilla.org/en/datasets).

Adjust sampling rate (French data)
```
python cpc/eval/utils/adjust_sample_rate.py ~/data/common_voices/fr/clips/ ~/data/common_voices/fr/clips_16k
```

Adjust sampling rate (English data, includes filtering to align number of files with French, i.e. 46,385 files):
```
python cpc/eval/utils/adjust_sample_rate_en.py ~/data/common_voices/en/clips/ ~/data/common_voices/en/clips_16k
```

## Train models

French:
```
python -u cpc/train.py --pathDB ~/data/common_voices/fr/clips_16k/ --pathTrain data_splits/common_voices_splits/fr/trainSeqs_all_16k.txt --file_extension .mp3 --pathCheckpoint checkpoints_fr/ --n_process_loader 8 --restart --ignore_cache
```

English:
```
python -u cpc/train.py --pathDB ~/data/common_voices/en/clips_16k/ --pathTrain data_splits/common_voices_splits/en/trainSeqs_all_16k.txt --file_extension .mp3 --pathCheckpoint checkpoints_en/ --n_process_loader 8 --restart --ignore_cache
```

To run the training on a cluster, there are [sbatch scripts](sbatch_scripts) available.

## Evaluate

### Generate features

Generate features for test stimuli using a trained model: 
```
python cpc/eval/build_zeroSpeech_features.py ../test_stimuli/ results/out_test_stimuli_fr_epoch_25 checkpoints_fr/checkpoint_25.pt --format npy
```

### Compute distances
Compute distances and save results to [results/distances](results/distances).
```
python cpc/eval/eval_phon_contrasts.py --stimuli-dir results/out_test_stimuli_fr_epoch_25
```