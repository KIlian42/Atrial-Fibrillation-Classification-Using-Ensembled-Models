import h5py
import random
import pandas as pd
from tqdm import tqdm
import os
from collections import Counter

from util.label_sets import *


def load_physionet2021_raw_ecgs(
    labels_included: str = "scored", samples: int = -1, shuffle: bool = True
) -> list:
    """
    Description:
    ----
    Loads the Physionet 2021 ECGs from the .h5 file (only the first lead).

    Args:
    ----
        labels_included (str): Possible values are: "3 classes", "5 classes", "scored".
        sample (int): Whether only a subset of training samples should be returned, which reduces the loading time (for prototyping).
                      Default is -1, which means that all ECGs (filtered by labels_included) will be returned.
        shuffle (bool): Whether the (training) samples will be randomly shuffled before returned.

    Returns:
    ----
        data (list): Returns 3 lists X, Y, Z:
                     X are the raw ECGs: list[list[float]]
                     Y are the labels: list[int]
                     Z are the IDs: list[str]
    """
    if labels_included == "3 classes":
        h5file = h5py.File(
            "/Users/kiliankramer/Desktop/atrial-fibrillation-classification-using-transformer-models/src/Datasets/physionet2021_SR_AF_AFL_PAC_PVC.h5",
            "r",
        )
        valid_labels = SR_AF_AFL_labels
    elif labels_included == "5 classes":
        h5file = h5py.File(
            "/Users/kiliankramer/Desktop/atrial-fibrillation-classification-using-transformer-models/src/Datasets/physionet2021_SR_AF_AFL_PAC_PVC.h5",
            "r",
        )
        valid_labels = SR_AF_AFL_PAC_PVC_labels
    elif labels_included == "scored":
        h5file = h5py.File(
            "/Users/kiliankramer/Desktop/atrial-fibrillation-classification-using-transformer-models/src/Datasets/physionet2021_scoredLabels.h5",
            "r",
        )
        valid_labels = SCORED_labels
    else:
        # h5file = h5py.File("src/Datasets/physionet2021.h5", "r")
        # valid_labels = set()
        raise (
            "Arguments for labels_included not valid. Possible values are: '3 classes', '5 classes', 'scored'."
        )

    # Load the ECGs and their IDs to a dictionary X_dict.
    # This step is necessary to map all labels from the labels files in the step.
    IDs = list(h5file.keys())
    if samples != -1:
        IDs = IDs[: samples * 2]
    X_dict = {}
    for key in tqdm(IDs, desc="Load ECG data"):
        X_dict[key] = list(h5file[key][0])

    # Load the labels and their IDs to a dictionary Y_dict.
    # Some ECGs can contain multiple labels.
    labels_df = pd.read_csv(
        os.path.join(os.getcwd(), "src/datasets/physionet2021_references.csv"), sep=";"
    )
    Y_dict = {}
    pbar = tqdm(total=len(labels_df), desc="Load ECG labels")
    for _, row in labels_df.iterrows():
        Y_dict[row["id"]] = row["labels"].split(",")
        pbar.update(1)

    X = []
    Y = []
    Z = []
    for patient_id in tqdm(Y_dict, desc="Map labels to ECGs"):
        for label in Y_dict[patient_id]:
            try:
                if label in valid_labels:
                    # ecg = [str(i) for i in X_dict[patient_id]]
                    X.append(X_dict[patient_id])
                    Y.append(str(label))
                    Z.append(str(patient_id))
            except:
                pass

    # Map labels
    Y = [0 if x == "426783006" else x for x in Y]  # Sinus Rhythm (SR)
    Y = [1 if x == "164889003" else x for x in Y]  # Atrial Fibrillation (AF)
    Y = [2 if x == "164890007" else x for x in Y]  # Atrial Flutter (AFL)
    if labels_included == "5 classes":
        Y = [
            3 if x == "284470004" else x for x in Y
        ]  # Premature Atrial Contraction (PAC)
        Y = [
            4 if x == "427172004" else x for x in Y
        ]  # Premature Ventricular Contraction (PVC)
    if labels_included == "scored":
        Y = [5 if x == "6374002" else x for x in Y]  # Bundle Branch Block
        Y = [6 if x == "426627000" else x for x in Y]  # Bradycardia
        Y = [7 if x == "733534002" else x for x in Y]  # Coronary Heart Disease
        Y = [
            8 if x == "713427006" else x for x in Y
        ]  # Complete Right Bundle Branch Block
        Y = [9 if x == "270492004" else x for x in Y]  # 1st Degree AV Block
        Y = [
            10 if x == "713426002" else x for x in Y
        ]  # Incomplete Right Bundle Branch Block
        Y = [11 if x == "39732003" else x for x in Y]  # Left Axis Deviation
        Y = [12 if x == "445118002" else x for x in Y]  # Left Anterior Fascicular Block
        Y = [13 if x == "164909002" else x for x in Y]  # Left Bundle Branch Block
        Y = [14 if x == "251146004" else x for x in Y]  # Low QRS Voltages
        Y = [
            15 if x == "698252002" else x for x in Y
        ]  # Nonspecific Intraventricular Conduction Disorder
        Y = [16 if x == "10370003" else x for x in Y]  # Pacing Rhythm
        Y = [17 if x == "365413008" else x for x in Y]  # Inferior Ischaemia
        Y = [18 if x == "164947007" else x for x in Y]  # Prolonged PR Interval
        Y = [19 if x == "111975006" else x for x in Y]  # Prolonged QT Interval
        Y = [20 if x == "164917005" else x for x in Y]  # Q Wave Abnormal
        Y = [21 if x == "47665007" else x for x in Y]  # Right Axis Deviation
        Y = [22 if x == "59118001" else x for x in Y]  # Right Bundle Branch Block
        Y = [23 if x == "427393009" else x for x in Y]  # Sinus Arrhythmia
        Y = [24 if x == "426177001" else x for x in Y]  # Sinus Bradycardia
        Y = [25 if x == "427084000" else x for x in Y]  # Sinus Tachycardia
        Y = [
            26 if x == "63593006" else x for x in Y
        ]  # Supraventricular Premature Beats
        Y = [27 if x == "164934002" else x for x in Y]  # T Wave Abnormal
        Y = [28 if x == "59931005" else x for x in Y]  # T Wave Inversion
        Y = [29 if x == "17338001" else x for x in Y]  # Ventricular Premature Beats

    # Shuffle data
    if shuffle:
        combined = list(zip(X, Y, Z))
        random.shuffle(combined)
        X, Y, Z = zip(*combined)
        X = list(X)
        Y = list(Y)
        Z = list(Z)

    # X = ECGs, Y = labels, Z = IDs
    return X, Y, Z


def load_train_test_set(X: list = None, Y: list = None, Z: list = None, classes: str = "5 classes", samples: int = -1) -> dict:
    """
    Description:
    ----
    Description of function.

    Args:
    ----
        X (list): ECGs.
        Y (list): Labels.
        Z (list): Id's.
        classes (str): Possible values are: "3 classes", "5 classes", "scored".
        samples (int): Whether only a subset of training samples should be returned, which reduces the loading time (for prototyping).
                      Default is -1, which means that all ECGs (filtered by labels_included) will be returned.
    Returns:
    ----
        dict: { x_train, y_train, z_train, x_test, y_test, z_test }
    """
    # Load data
    if X == None and Y == None and Z == None:
        X, Y, Z = load_physionet2021_raw_ecgs(labels_included=classes, samples=samples)
    X = list(X)
    Y = list(Y)
    Z = list(Z)
    # Load train/test sets
    f = open(
        os.path.join(
            os.getcwd(), "src/datasets/Physionet2021_5classes_IDs_trainset.txt"
        ),
        "r",
        encoding="utf-8",
    )
    trainset_ids = f.readlines()
    f.close()
    f = open(
        os.path.join(
            os.getcwd(), "src/datasets/Physionet2021_5classes_IDs_testset.txt"
        ),
        "r",
        encoding="utf-8",
    )
    testset_ids = f.readlines()
    f.close()
    trainset_ids = set([x.replace("\n", "") for x in trainset_ids])
    testset_ids = set([x.replace("\n", "") for x in testset_ids])

    # Prepare training data
    x_train = []
    y_train = []
    z_train = []
    x_test = []
    y_test = []
    z_test = []
    for ecg in zip(X, Y, Z):
        if ecg[2] in trainset_ids:
            x_train.append(ecg[0])
            y_train.append(ecg[1])
            z_train.append(ecg[2])
        elif ecg[2] in testset_ids:
            x_test.append(ecg[0])
            y_test.append(ecg[1])
            z_test.append(ecg[2])

    new_test_dict = remove_multilabels_from_tests(
        x_test=x_test, y_test=y_test, z_test=z_test
    )

    counts = Counter(y_train)
    print(f"Train distribution: {counts}")
    counts = Counter(new_test_dict["y_test"])
    print(f"Test distribution: {counts}")

    return {
        "x_train": list(x_train),
        "y_train": list(y_train),
        "z_train": list(z_train),
        "x_test": list(new_test_dict["x_test"]),
        "y_test": list(new_test_dict["y_test"]),
        "z_test": list(new_test_dict["z_test"]),
    }


def remove_multilabels_from_tests(x_test: list, y_test: list, z_test: list) -> dict:
    combined = list(zip(x_test, y_test, z_test))
    combined.sort(key=lambda x: x[1], reverse=True)
    x_test, y_test, z_test = zip(*combined)
    x_test_new = []
    y_test_new = []
    z_test_new = []
    z_already_test = set()
    for ecg in zip(x_test, y_test, z_test):
        if ecg[2] not in z_already_test:
            x_test_new.append(ecg[0])
            y_test_new.append(ecg[1])
            z_test_new.append(ecg[2])
            z_already_test.add(ecg[2])
    combined = list(zip(x_test_new, y_test_new, z_test_new))
    random.shuffle(combined)
    x_test, y_test, z_test = zip(*combined)
    return {"x_test": list(x_test), "y_test": list(y_test), "z_test": list(z_test)}


def main():
    train_test_dict = load_train_test_set()


if __name__ == "__main__":
    main()
