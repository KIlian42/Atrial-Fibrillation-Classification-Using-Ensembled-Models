import pandas as pd
import numpy as np 
import os
from collections import Counter

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from util.load_physionet2021 import load_train_test_set

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    FEATURES_LIBRARY = "neurokit2" # biobss neurokit2
    NUM_CLASSES = 5

    # train_test_dict = load_train_test_set(classes=f"{num_classes} classes")
    file_path = os.path.join(
        os.getcwd(),
        f"src/datasets/Phsyionet2021_5classes_{FEATURES_LIBRARY}_extracted_features_original.csv",
    )
    print("hi1")

    df = pd.read_csv(file_path, delimiter=";")
    print("hi2")
    print(df.head)
    df = df.replace([-np.inf], np.nan)
    df = df.replace([np.inf], np.nan)
    df.dropna(inplace=True)
    if NUM_CLASSES == 3:
        df = df[df['label'] != 3]
        df = df[df['label'] != 4]
    print(df.head)

    X = []
    for column in df.columns:
        if column == "id":
            Y = df[column].tolist()
            continue
        if column == "label":
            Z = df[column].tolist()
            continue
        X.append(df[column].tolist())
    # Transpose
    X = [list(row) for row in zip(*X)]
    
    train_test_dict = load_train_test_set(X=X, Y=Z, Z=Y)
    X_train = train_test_dict["x_train"]
    Y_train = train_test_dict["y_train"]
    X_test = train_test_dict["x_test"]
    Y_test = train_test_dict["y_test"]


    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)
    # pca = PCA(n_components=1)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    # X_train = [list(x) for x in X_train]
    # X_test = [list(x) for x in X_test]


    # Define the strategy for undersampling
    undersample_strategy = {0: 1500, 1: 1500, 2: 1500}  # Reduce class x to n
    rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=0)
    X_resampled, Y_resampled = rus.fit_resample(X_train, Y_train)
    X_train = list(X_resampled)
    Y_train = list(Y_resampled)
    if NUM_CLASSES == 5:
        # Define the strategy for oversampling
        oversample_strategy = {3: 1500, 4: 1500}  # Reduce class x to n
        rus = RandomOverSampler(sampling_strategy=oversample_strategy, random_state=0)
        X_resampled, Y_resampled = rus.fit_resample(X_train, Y_train)
        X_train = list(X_resampled)
        Y_train = list(Y_resampled)

    rf_classifier = RandomForestClassifier() # RandomForestClassifier(n_estimators=100, random_state=42, max_features=30)
    rf_classifier.fit(X_train, Y_train)

    

    rf_pred = rf_classifier.predict(X_test)
    print(f"Accuracy Random Forest {FEATURES_LIBRARY} test:", accuracy_score(Y_test, rf_pred))

    # Define the strategy for undersampling
    if NUM_CLASSES == 3:
        undersample_strategy = {0: 408, 1: 408, 2: 408}  # Reduce class x to n
    elif NUM_CLASSES == 5:
        undersample_strategy = {0: 53, 1: 53, 2: 53, 3: 53, 4: 53}  # Reduce class x to n
    rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=0)
    X_resampled, Y_resampled = rus.fit_resample(X_test, Y_test)
    X_test = list(X_resampled)
    Y_test = list(Y_resampled)

    rf_pred = rf_classifier.predict(X_test)
    print(f"Accuracy Random Forest {FEATURES_LIBRARY} test balanced:", accuracy_score(Y_test, rf_pred))





if __name__ == "__main__":
    main()
