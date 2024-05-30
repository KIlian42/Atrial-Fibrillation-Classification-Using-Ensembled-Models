# Neurokit2 documentation:
# https://neuropsychology.github.io/NeuroKit/
# https://neuropsychology.github.io/NeuroKit/functions/ecg.html

import path
import sys

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from util.load_physionet2021 import load_physionet2021_raw_ecgs
import neurokit2 as nk
import biobss

import multiprocessing
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

import json


def split_list_into_n_parts_for_multiprocessing(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def extract_features_from_ecgs(input_list, process, target_list, lock):
    for x in tqdm(
        input_list, desc=f"Extract features ({process+1}. process)", position=process
    ):
        try:
            # Detect peaks using 'pantompkins' method.
            locs_peaks = biobss.ecgtools.ecg_detectpeaks(x, 500, "pantompkins")
            # Delineate ECG signal using 'neurokit2' package.
            _, fiducials = nk.ecg_delineate(
                ecg_cleaned=x,
                rpeaks=locs_peaks,
                sampling_rate=500,
                method="peak",
            )
            all_features = {}
            # Calculate features from R peaks
            features_rpeaks = biobss.ecgtools.ecg_features.from_Rpeaks(
                x, locs_peaks, 500, average=True
            )
            all_features.update(features_rpeaks)
            # Calculate features from P, Q, R, S, T waves
            features_waves = biobss.ecgtools.ecg_features.from_waves(
                x, locs_peaks, fiducials, 500, average=True
            )
            all_features.update(features_waves)
            features = []
            for key in all_features:
                feature = key + ": " + str(round(all_features[key], 4))
                features.append(feature)
            with lock:
                target_list.extend([features])
        except Exception as e:
            with lock:
                target_list.extend(["empty"])
            # print(e)


def main():
    manager = multiprocessing.Manager()
    X_features = manager.list()
    lock = manager.Lock()
    num_processes = multiprocessing.cpu_count()

    data = load_physionet2021_raw_ecgs(labels_included="5 classes", samples=100)
    X, Y, Z = data[0], data[1], data[2]

    processes = []
    splitted_X = split_list_into_n_parts_for_multiprocessing(X, num_processes)
    for process in range(num_processes):
        p = multiprocessing.Process(
            target=extract_features_from_ecgs,
            args=(splitted_X[process], process, X_features, lock),
        )
        processes.append(p)
        p.start()
    for process in processes:
        process.join()

    json_list = []
    for example_index, features in enumerate(X_features):
        if "empty" in features:
            continue
        json_list.append(
            {"id": Z[example_index], "label": Y[example_index], "features": features}
        )

    print("Total data", len(json_list))
    with open("neurokit2_extracted_features.json", "w") as file:
        json.dump(json_list, file, indent=4)


if __name__ == "__main__":
    main()
