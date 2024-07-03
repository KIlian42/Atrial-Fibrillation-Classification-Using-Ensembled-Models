import path
import sys
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent.parent.parent)

import os
import random
import pandas as pd
from tqdm import tqdm
from src.util.map_Physionet2021_labels import VALID_LABELS, map_arrhyhtmia_id_to_index, map_arrhyhtmia_index_to_id

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    f = open("src/datasets/prepared/trainset_patient_ids.txt", "r", encoding="utf-8")
    trainset_patient_ids = f.readlines()
    f.close()
    f = open("src/datasets/prepared/testset_patient_ids.txt", "r", encoding="utf-8")
    testset_patient_ids = f.readlines()
    f.close()
    trainset_patient_ids = list(map(lambda x: x.replace("\n", ""), trainset_patient_ids))
    testset_patient_ids = list(map(lambda x: x.replace("\n", ""), testset_patient_ids))
    trainset_patient_ids = set(trainset_patient_ids)
    testset_patient_ids = set(testset_patient_ids)

    f = open("src/datasets/prepared/biobss_features_imputated.csv", "r", encoding="utf-8")
    id_features = f.readlines()
    f.close()
    id_features = list(map(lambda x: x.replace("\n", "").split(";"), id_features))
    id_features = list(map(lambda x: [x[0]] + list(map(lambda y: float(y.split(": ")[1]), x[1:])), id_features))

    Y_dict = {}
    labels_df = pd.read_csv("src/datasets/physionet2021_references.csv", sep=";")
    pbar = tqdm(total=len(labels_df), desc="Load ECG labels", position=0, leave=True)
    for _, row in labels_df.iterrows():
        labels = row["labels"].strip().split(",")
        valid_labels_index = []
        for label in labels:
            if label in VALID_LABELS:
                valid_labels_index.append(map_arrhyhtmia_id_to_index(label))
        labels_binary_encoded = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for index in valid_labels_index:
            labels_binary_encoded[index] = 1 
        Y_dict[row["id"]] = labels_binary_encoded
        pbar.update(1)
    del labels_df
    
    X_train = []
    Y_train = []
    Z_train = []
    X_test = []
    Y_test = []
    Z_test = []
    for id_feature in id_features:
        if id_feature[0] in trainset_patient_ids:
            X_train.append(id_feature[1:])
            Y_train.append(Y_dict[id_feature[0]])
            Z_train.append(id_feature[0])
        elif id_feature[0] in testset_patient_ids:
            X_test.append(id_feature[1:])
            Y_test.append(Y_dict[id_feature[0]])
            Z_test.append(id_feature[0])

    combined = list(zip(X_train, Y_train, Z_train))
    random.shuffle(combined)
    X_train, Y_train, Z_train = zip(*combined)
    X_train = list(X_train)
    Y_train = list(Y_train)
    Z_train = list(Z_train)
    combined = list(zip(X_test, Y_test, Z_test))
    random.shuffle(combined)
    X_test, Y_test, Z_test = zip(*combined)
    X_test = list(X_test)
    Y_test = list(Y_test)
    Z_test = list(Z_test)

    # Train classifier
    rf_classifier = RandomForestClassifier() # RandomForestClassifier(n_estimators=100, random_state=42, max_features=30)
    rf_classifier.fit(X_train, Y_train)

    # Calculate accuracy and other metrics
    pred_prob = rf_classifier.predict(X_test)
    threshold = 0.5
    pred = (pred_prob > threshold).astype(int)
    accuracy = accuracy_score(Y_test, pred)
    precision = precision_score(Y_test, pred, average='micro')  # or 'macro'
    recall = recall_score(Y_test, pred, average='micro')
    f1 = f1_score(Y_test, pred, average='micro')
    print(f"Accuracy test set: {accuracy}")
    print(f"Precision test set: {precision}")
    print(f"Recall test set: {recall}")
    print(f"F1 Score test set: {f1}")

    # == Convert predictions unbalanced in Physionet
    import shutil
    try:
        shutil.rmtree("test_outputs")
    except FileNotFoundError:
        print(f"Error: Directory not found.")
    except PermissionError:
        print(f"Error: Permission denied.")
    try:
        os.makedirs("test_outputs")
    except OSError as e:
        print(f"Error: {e.strerror}")

    pbar = tqdm(total=len(pred), desc="Convert test_outputs", position=0, leave=True)
    for index, prediction in enumerate(tqdm(zip(pred, pred_prob))):
        pbar.update(1)
        new_file = "#"
        new_file += Z_test[index] + "\n"
        # ids
        for pred_index, _ in enumerate(prediction[0]):
            new_file += map_arrhyhtmia_index_to_id(pred_index) + ","
        new_file = new_file[:-1] + "\n"
        # pred
        for pred_index, _ in enumerate(prediction[0]):
            if prediction[0][pred_index] == 1:
                value = "True"
            elif prediction[0][pred_index] == 0:
                value = "False"
            new_file += value + ","
        new_file = new_file[:-1] + "\n"
        # pred_prob
        for pred_index, _ in enumerate(prediction[1]):
            new_file += str(prediction[1][pred_index]) + ","
        new_file = new_file[:-1]
        # with open(os.path.join(path, f"test_outputs/{Z_test[index]}.csv"), "w", encoding="utf-8") as file:
        #    file.write(new_file)
        with open(f"test_outputs/{Z_test[index]}.csv", "w", encoding="utf-8") as file:
            file.write(new_file)

if __name__ == "__main__":
    main()
