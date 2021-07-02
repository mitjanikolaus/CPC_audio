import argparse
import math
import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import spatial

IDS_REFERENCE_STIMULI = [1, 4, 13, 16]


def measure_distance(stimuli_1, stimuli_2, alignment_method="trim"):
    if len(stimuli_1) > len(stimuli_2):
        if alignment_method == "trim":
            print("WARNING: trimming input stimuli to align lengths")
            stimuli_1 = stimuli_1[:len(stimuli_2)]
        elif alignment_method == "pad":
            print("WARNING: padding input stimuli to align lengths")
            stimuli_2 = np.pad(stimuli_2, (0, len(stimuli_1)-len(stimuli_2)))
        else:
            raise RuntimeError("Unknown alignment method: ", alignment_method)
    elif len(stimuli_2) > len(stimuli_1):
        if alignment_method == "trim":
            print("WARNING: trimming input stimuli to align lengths")
            stimuli_2 = stimuli_2[:len(stimuli_1)]
        elif alignment_method == "pad":
            print("WARNING: padding input stimuli to align lengths")
            stimuli_1 = np.pad(stimuli_1, (0, len(stimuli_2)-len(stimuli_1)))
        else:
            raise RuntimeError("Unknown alignment method: ", alignment_method)

    return spatial.distance.cosine(stimuli_1, stimuli_2)


def compare_stimuli(stimuli):
    distances = {}
    for id, data in sorted(stimuli.items()):
        # if id not in IDS_REFERENCE_STIMULI:
        distances[id] = {}
        min_dist = math.inf
        mind_dist_ref = 0
        for ref in IDS_REFERENCE_STIMULI:
            dist = measure_distance(data, stimuli[ref])
            distances[id][ref] = dist
            # print(f"Stimulus: {id} Ref: {ref} Distance: {dist}")
            if dist < min_dist:
                min_dist = dist
                mind_dist_ref = ref
        print(f"Stimulus: {id} Ref: {mind_dist_ref} Distance: {min_dist}")
    distances = pd.DataFrame(distances).T
    return distances


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--stimuli-dir",
        type=str,
        required=True,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    output = {}
    pathlist = Path(args.stimuli_dir).rglob('*.npy')
    for path in pathlist:
        data = np.load(str(path))

        id = int(str(path).split("/")[-1].split(".npy")[0])
        output[id] = data.flatten()

    print(f"Output for: {args.stimuli_dir}")
    distances = compare_stimuli(output)
    print(distances)

    out_dir = "results/distances"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    distances.to_csv(os.path.join(out_dir, f"{args.stimuli_dir.split('/')[-1]}.csv"))

