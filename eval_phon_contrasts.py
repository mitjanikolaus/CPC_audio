import math
from pathlib import Path

from cpc.dataset import findAllSeqs
from cpc.eval.ABX.abx_iterators import ABXFeatureLoader
from cpc.feature_loader import buildFeature, FeatureModule
from hubconf import CPC_audio
import pandas as pd
from scipy import spatial

QUANTIZED_STIMULI_PATH = "../quantized/test_stimuli/quantized_outputs.txt"
IDS_REFERENCE_STIMULI = [1, 4, 13, 16]

def measure_distance(stimuli_1, stimuli_2):
    return spatial.distance.cosine(stimuli_1, stimuli_2)

def load_quantized_stimuli():
    stimuli = {}
    # data = pd.read_csv(QUANTIZED_STIMULI_PATH, delimiter="\t", index_col=0)
    with open(QUANTIZED_STIMULI_PATH) as f:
        for line in f.readlines():
            id, data = line.split("\t")
            id = int(id)
            data = [int(s) for s in data.split(',')]
            stimuli[id] = data

    print(stimuli)
    return stimuli


def compare_stimuli(stimuli):
    for id, data in sorted(stimuli.items()):
        if id not in IDS_REFERENCE_STIMULI:
            min_dist = math.inf
            mind_dist_ref = 0
            for ref in IDS_REFERENCE_STIMULI:
                dist = measure_distance(data, stimuli[ref])
                # print(f"Stimulus: {id} Ref: {ref} Distance: {dist}")
                if dist < min_dist:
                    min_dist = dist
                    mind_dist_ref = ref
            print(f"Stimulus: {id} Ref: {mind_dist_ref} Distance: {min_dist}")


def run_eval():
    # model = CPC_audio(pretrained=True)
    # print(model)

    stimuli = load_quantized_stimuli()
    compare_stimuli(stimuli)


if __name__ == "__main__":
    run_eval()