# Neurokit2 documentation:
# https://neuropsychology.github.io/NeuroKit/
# https://neuropsychology.github.io/NeuroKit/functions/ecg.html

import warnings
warnings.filterwarnings("ignore")

import h5py

import scipy
from scipy.signal import butter, lfilter

import multiprocessing
from tqdm import tqdm

import neurokit2 as nk
import biobss

import pandas as pd

VALID_LABELS = set(
    [
        "164889003",
        "164890007",
        "6374002",
        "426627000",
        "733534002",
        "713427006",
        "270492004",
        "713426002",
        "39732003",
        "445118002",
        "164909002",
        "251146004",
        "698252002",
        "426783006",
        "284470004",
        "10370003",
        "365413008",
        "427172004",
        "164947007",
        "111975006",
        "164917005",
        "47665007",
        "59118001",
        "427393009",
        "426177001",
        "427084000",
        "63593006",
        "164934002",
        "59931005",
        "17338001",
    ]
)

arrhyhtmia_mapping_id_to_index = {
    "426783006": 0, # sinus rhythm (SR)
    "164889003": 1, # atrial fibrillation (AF)
    "164890007": 2, # atrial flutter (AFL)
    "284470004": 3, # premature atrial contraction (PAC)
    "63593006": 3, # supraventricular premature beats (SVPB)
    "427172004": 4, # premature ventricular contractions (PVC)
    "17338001": 4, # ventricular premature beats (VPB)
    "6374002": 5, # bundle branch block (BBB)
    "426627000": 6, # bradycardia (Brady)
    "733534002": 7, # complete left bundle branch block (CLBBB)
    "164909002": 7, # left bundle branch block (LBBB)
    "713427006": 8, # complete right bundle branch block (CRBBB)
    "59118001": 8, # right bundle branch block (RBBB)
    "270492004": 9, # 1st degree av block (IAVB)
    "713426002": 10, # incomplete right bundle branch block (IRBBB)
    "39732003": 11, # left axis deviation (LAD)
    "445118002": 12, # left anterior fascicular block (LAnFB)
    "251146004": 13, # low qrs voltages (LQRSV)
    "698252002": 14, # nonspecific intraventricular conduction disorder (NSIVCB)
    "10370003": 15, # pacing rhythm (PR)
    "365413008": 16, # poor R wave Progression (PRWP)
    "164947007": 17, # prolonged pr interval (LPR)
    "111975006": 18, # prolonged qt interval (LQT)
    "164917005": 19, # qwave abnormal (QAb)
    "47665007": 20,  # right axis deviation (RAD)
    "427393009": 21, # sinus arrhythmia (SA)
    "426177001": 22, # sinus bradycardia (SB)
    "427084000": 23, # sinus tachycardia (STach)
    "164934002": 24, # t wave abnormal (TAb)
    "59931005": 25 # t wave inversion (TInv)
}

def map_arrhyhtmia_id_to_index(x: str) -> int:
    return arrhyhtmia_mapping_id_to_index[x]

def pad_or_truncate_ecg(ecg: list, max_samples: int) -> list:
    return ecg[:max_samples] + [0] * (max_samples - len(ecg))

def resample_ecg(ecg: list, resample: int):
    new_ecg = scipy.signal.resample(
        ecg, resample, t=None, axis=0, window=None, domain="time"
    )
    return list(new_ecg)

def normalize_to_minus11(ecg: list):
    max_val = max(ecg)
    min_val = min(ecg)
    # Handle the case where max_val and min_val are the same (to avoid division by zero)
    if max_val == min_val:
        return [0 for _ in ecg]
    normalized_values = [2 * (x - min_val) / (max_val - min_val) - 1 for x in ecg]
    return normalized_values

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(ecg: list, lowcut: float, highcut: float, sampling_rate: int, order: int =4):
    b, a = butter_bandpass(lowcut, highcut, sampling_rate, order=order)
    y = lfilter(b, a, ecg)
    return y

def split_list_into_n_parts_for_multiprocessing(lst: list, n: int) -> list:
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]

def extract_features_from_ecgs(input_list, process, target_list, lock):
    for x in tqdm(input_list, desc=f"Extract features ({process+1}. process)", position=process):
        try:
            # Detect peaks using 'pantompkins' method.
            locs_peaks = biobss.ecgtools.ecg_detectpeaks(x[1], 500, "pantompkins")
            # Delineate ECG signal using 'neurokit2' package.
            _, fiducials = nk.ecg_delineate(ecg_cleaned=x[1], rpeaks=locs_peaks, sampling_rate=500, method="peak")
            all_features = {}
            # Calculate features from R peaks
            features_rpeaks = biobss.ecgtools.ecg_features.from_Rpeaks(x[1], locs_peaks, 500, average=True)
            all_features.update(features_rpeaks)
            # Calculate features from P, Q, R, S, T waves
            features_waves = biobss.ecgtools.ecg_features.from_waves(x[1], locs_peaks, fiducials, 500, average=True)
            all_features.update(features_waves)
            features = []
            for key in all_features:
                feature = key + ": " + str(round(all_features[key], 4))
                features.append(feature)
            with lock:
                new_entry = [[x[0], features]]
                target_list.extend(new_entry)
        except Exception as e:
            with lock:
                target_list.extend(["empty"])

def main():
    manager = multiprocessing.Manager()
    X_features = manager.list()
    lock = manager.Lock()
    num_processes = multiprocessing.cpu_count()
    print(f"CPUs available: {num_processes}")

    path_to_ecgs = "/src/datasets/prepared/physionet2021_scoredLabels.h5"
    X_dict = {}
    h5file = h5py.File(path_to_ecgs, "r")
    IDs = list(h5file.keys())# [:100] # [:5]
    pbar = tqdm(total=len(IDs), desc="Load ECG data", position=0, leave=True)
    for key in IDs:
        X_dict[key] = pad_or_truncate_ecg(ecg=list(h5file[key][0]), max_samples=5000)
        # X_dict[key] = resample_ecg(ecg=X_dict[key], resample=2000)
        # X_dict[key] = normalize_to_minus11(ecg=X_dict[key])
        # X_dict[key] = butter_bandpass_filter(ecg=X_dict[key], lowcut=0.3, highcut=21.0, sampling_rate=200)
        pbar.update(1)

    X = []
    for key in X_dict:
        X.append([key, X_dict[key]])
    splitted_X = split_list_into_n_parts_for_multiprocessing(X, num_processes)

    processes = []
    for process in range(num_processes):
        p = multiprocessing.Process(target=extract_features_from_ecgs, args=(splitted_X[process], process, X_features, lock))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()

    X_features = list(X_features)
    csv_list = []
    for features in X_features:
        if "empty" in features:
            continue
        new_row = ""
        new_row += f"{features[0]};"
        for feature in features[1]:
            new_row += feature.replace(";", "") + ";"
        new_row = new_row[:-1]
        new_row += "\n"
        csv_list.append(new_row)
    print(f"Total data: {len(csv_list)}")

    with open("biobss_features.csv", "w") as file:
        file.writelines(csv_list)

if __name__ == "__main__":
    main()