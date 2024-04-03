from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from util.label_sets import SR_AF_AFL_PAC_PVC_labels


def main():
    # # ========================================================= Load data
    X_dict = {}
    X = []
    Y_labels_dict = {}
    Y_labels = []
    labels_df = pd.read_csv("src/Datasets/physionet2021_references.csv", sep=";")
    pbar = tqdm(total=len(labels_df), desc="Load ECG labels")
    for _, row in labels_df.iterrows():
        Y_labels_dict[row["id"]] = row["labels"].split(",")
        pbar.update(1)
    labels_df = pd.read_csv("src/Datasets/SNOMED_codes.csv", sep=",")
    physionet2021_labels = {}
    for _, row in labels_df.iterrows():
        physionet2021_labels[str(row["SNOMED CT Code"])] = row["Dx"]

    for id in tqdm(Y_labels_dict, desc="Map labels to ECGs"):
        for label in Y_labels_dict[id]:
            try:
                if label in physionet2021_labels and label in SR_AF_AFL_PAC_PVC_labels:
                    X.append(list(X_dict[id]))
                    Y_labels.append(
                        str(physionet2021_labels[label]) + " (" + str(label) + ")"
                    )
            except IndexError:
                pass

    # == Plot data ==
    label_counts = Counter(Y_labels)
    print(label_counts)
    combined = list(zip(label_counts.keys(), label_counts.values()))
    combined.sort(key=lambda x: x[1], reverse=True)
    label_keys, label_values = zip(*combined)
    label_keys = list(label_keys)
    label_values = list(label_values)

    for index, key in enumerate(label_keys):
        label_keys[index] = str(index + 1) + ". " + key[0].upper() + key[1:]

    plt.figure(figsize=(30, 10))
    plt.bar(label_keys, label_values, color="#1f77b4")
    plt.title("Physionet 2021 labels")
    plt.xlabel("Arrhythmia type", labelpad=7)
    plt.ylabel("Occurence")
    plt.xticks(rotation=45, ha="right", fontsize=16)  # (rotation='diagional')
    bars = plt.bar(label_keys, label_values, color="#1f77b4")
    # Adding the counts on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 5,
            yval,
            ha="center",
            va="bottom",
            fontsize=16,
        )
    plt.savefig(
        "plots/Physionet2021_labels_SR_AF_AFL_PAC_PVC.png",
        format="png",
        bbox_inches="tight",
    )
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
