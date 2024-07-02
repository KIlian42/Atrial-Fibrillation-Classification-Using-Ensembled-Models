import pandas as pd
import numpy as np 
import os
from collections import Counter

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from code_old.load_physionet2021 import load_train_test_set

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import json

from tqdm import tqdm
import random

import matplotlib.pyplot as plt
import seaborn as sns

def main():

    with open("/Users/kiliankramer/Desktop/atrial-fibrillation-classification-using-transformer-models/src/Datasets/preparedData/all_5.json", "r") as file:
        data = json.load(file)
    
    X = []
    Y = []
    Z = []
    ids_set = set()
    for index, row in enumerate(tqdm(data)):
        if data[index]["id"] not in ids_set:
            label = data[index]["label"]
            if label in set(["0", "3", "4"]):
                continue
            features = [float(x) for x in data[index]["features"]]
            X.append(features)
            Y.append(label)
            Z.append(data[index]["id"])
            ids_set.add(data[index]["id"])

    # combined = list(zip(X, Y, Z))
    # random.shuffle(combined)
    # X, Y, Z = zip(*combined)
    # X = list(X)
    # Y = list(Y)
    # Z = list(Z)
    
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


    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for ecg in zip(X, Y, Z):
        if ecg[2] in trainset_ids:
            X_train.append(ecg[0])
            Y_train.append(int(ecg[1]))
        elif ecg[2] in testset_ids:
            X_test.append(ecg[0])
            Y_test.append(int(ecg[1]))

    counts = Counter(Y_train)
    print(f"Train distribution: {counts}")
    counts = Counter(Y_test)
    print(f"Test distribution: {counts}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    pca = PCA(n_components=15)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_train = [list(x) for x in X_train]
    X_test = [list(x) for x in X_test]
    
    # quit()

    # Define the strategy for oversampling
    # oversample_strategy = {4: 1000}  # Reduce class x to n
    # rus = RandomOverSampler(sampling_strategy=oversample_strategy, random_state=0)
    # X_resampled, Y_resampled = rus.fit_resample(X_train, Y_train)
    # X_train = list(X_resampled)
    # Y_train = list(Y_resampled)
    undersample_strategy = {2: 1487}  # Reduce class x to n
    rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=0)
    X_resampled, Y_resampled = rus.fit_resample(X_train, Y_train)
    X_train = list(X_resampled)
    Y_train = list(Y_resampled)

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=42)
    # classifier = SVC(kernel='rbf', random_state=42)


    classifier.fit(X_train, Y_train)

    

    rf_pred = classifier.predict(X_test)
    print(f"Accuracy Random Forest biobss test:", accuracy_score(Y_test, rf_pred))

    # Define the strategy for undersampling
    undersample_strategy = {1: 323, 2: 323}  # Reduce class x to n
    rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=0)
    X_resampled, Y_resampled = rus.fit_resample(X_test, Y_test)
    X_test = list(X_resampled)
    Y_test = list(Y_resampled)

    rf_pred = classifier.predict(X_test)
    print(f"Accuracy Random Forest biobss test balanced:", accuracy_score(Y_test, rf_pred))

    # == Plot results ==
    cm = confusion_matrix(Y_test, rf_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=["AF", "AFL"], yticklabels=["AF", "AFL"]
    )
    fig = plt.gcf()
    fig.canvas.manager.set_window_title("Test: Physionet 2021")
    plt.title("Transformer")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.savefig(
        "ConfusionMatrix_Physionet2021_featureBased_1on1.png"
    )
    plt.show()


if __name__ == "__main__":
    main()
