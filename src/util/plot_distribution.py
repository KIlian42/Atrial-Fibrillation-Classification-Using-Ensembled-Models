from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from map_Physionet2021_labels import VALID_LABELS


def main():
    # == Load data ==
    Y_labels_dict = {}
    Y_labels = []
    Y_labels_scored = []
    labels_df = pd.read_csv("src/datasets/physionet2021_references.csv", sep=";")
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
                Y_labels.append(str(physionet2021_labels[label]) + " (" + str(label) + ")")
                if label in physionet2021_labels and label in VALID_LABELS:
                    Y_labels_scored.append(str(physionet2021_labels[label]) + " (" + str(label) + ")")
            except:
                pass

    # # == Plot data ==
    # # Plot 1
    # label_counts = Counter(Y_labels)
    # print(label_counts)
    # combined = list(zip(label_counts.keys(), label_counts.values()))
    # combined.sort(key=lambda x: x[1], reverse=True)
    # label_keys, label_values = zip(*combined)
    # label_keys = list(label_keys)
    # label_values = list(label_values)

    # for index, key in enumerate(label_keys):
    #     label_keys[index] = str(index + 1) + ". " + key[0].upper() + key[1:]
    # plt.figure(figsize=(60, 20))
    # plt.bar(label_keys, label_values, color="#1f77b4")
    # plt.title("Physionet 2021 labels")
    # plt.xlabel("Arrhythmia type", labelpad=7)
    # plt.ylabel("Occurence")
    # plt.xticks(rotation=45, ha="right", fontsize=16)  # (rotation='diagional')
    # bars = plt.bar(label_keys, label_values, color="#1f77b4")
    # # Adding the counts on top of the bars
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         yval + 5,
    #         yval,
    #         ha="center",
    #         va="bottom",
    #         fontsize=16,
    #     )
    # plt.savefig("distribution.png", format="png", bbox_inches="tight")
    # # plt.show()
    # plt.close()

    # # Plot 2
    # label_counts = Counter(Y_labels_scored)
    # print(label_counts)
    # combined = list(zip(label_counts.keys(), label_counts.values()))
    # combined.sort(key=lambda x: x[1], reverse=True)
    # label_keys, label_values = zip(*combined)
    # label_keys = list(label_keys)
    # label_values = list(label_values)

    # for index, key in enumerate(label_keys):
    #     label_keys[index] = str(index + 1) + ". " + key[0].upper() + key[1:]
    # plt.figure(figsize=(30, 10))
    # plt.bar(label_keys, label_values, color="#1f77b4")
    # plt.title("Physionet 2021 labels")
    # plt.xlabel("Arrhythmia type", labelpad=7)
    # plt.ylabel("Occurence")
    # plt.xticks(rotation=45, ha="right", fontsize=16)  # (rotation='diagional')
    # bars = plt.bar(label_keys, label_values, color="#1f77b4")
    # # Adding the counts on top of the bars
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         yval + 5,
    #         yval,
    #         ha="center",
    #         va="bottom",
    #         fontsize=16,
    #     )
    # plt.savefig("distribution.png", format="png", bbox_inches="tight")
    # # plt.show()
    # plt.close()


    # Data for Plot 1
    label_counts_1 = Counter(Y_labels)
    combined_1 = list(zip(label_counts_1.keys(), label_counts_1.values()))
    combined_1.sort(key=lambda x: x[1], reverse=True)
    label_keys_1, label_values_1 = zip(*combined_1)
    label_keys_1 = list(label_keys_1)
    label_values_1 = list(label_values_1)

    for index, key in enumerate(label_keys_1):
        label_keys_1[index] = str(index + 1) + ". " + key[0].upper() + key[1:]

    # Data for Plot 2
    label_counts_2 = Counter(Y_labels_scored)
    combined_2 = list(zip(label_counts_2.keys(), label_counts_2.values()))
    combined_2.sort(key=lambda x: x[1], reverse=True)
    label_keys_2, label_values_2 = zip(*combined_2)
    label_keys_2 = list(label_keys_2)
    label_values_2 = list(label_values_2)

    for index, key in enumerate(label_keys_2):
        label_keys_2[index] = str(index + 1) + ". " + key[0].upper() + key[1:]

    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(60, 40))

    # Plot 1
    ax1.bar(label_keys_1, label_values_1, color="#1f77b4")
    ax1.set_title("Physionet 2021 Labels (all)", fontsize=20)
    ax1.set_xlabel("Arrhythmia type", labelpad=7, fontsize=16)
    ax1.set_ylabel("Occurrence", fontsize=16)
    ax1.set_xticks(label_keys_1)
    ax1.set_xticklabels(label_keys_1, rotation=45, ha="right", fontsize=16)
    bars_1 = ax1.bar(label_keys_1, label_values_1, color="#1f77b4")

    # Adding the counts on top of the bars for Plot 1
    for bar in bars_1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 5, yval, ha="center", va="bottom", fontsize=16)

    # Plot 2
    ax2.bar(label_keys_2, label_values_2, color="#1f77b4")
    ax2.set_title("Physionet 2021 Labels (challenge)", fontsize=20)
    ax2.set_xlabel("Arrhythmia type", labelpad=7, fontsize=16)
    ax2.set_ylabel("Occurrence", fontsize=16)
    ax2.set_xticks(label_keys_2)
    ax2.set_xticklabels(label_keys_2, rotation=45, ha="right", fontsize=16)
    bars_2 = ax2.bar(label_keys_2, label_values_2, color="#1f77b4")

    # Adding the counts on top of the bars for Plot 2
    for bar in bars_2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 5, yval, ha="center", va="bottom", fontsize=16)

    plt.tight_layout()
    plt.savefig("combined_distribution.png", format="png", bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
