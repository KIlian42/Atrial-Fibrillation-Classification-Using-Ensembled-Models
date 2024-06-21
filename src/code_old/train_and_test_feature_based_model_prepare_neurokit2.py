import os
import json
from tqdm import tqdm

from util.load_physionet2021 import load_train_test_set


def main():
    features_library = "neurokit2"
    num_classes = 5

    # train_test_dict = load_train_test_set(classes=f"{num_classes} classes")
    file_path = os.path.join(
        os.getcwd(),
        f"src/datasets/Phsyionet2021_5classes_{features_library}_extracted_features_original.json",
    )
    with open(file_path, "r") as file:
        data = json.load(file)

    features_order = []
    # create csv
    csv = []
    new_row = "id;label;"
    for index, row in enumerate(data):
        if len(row["features"]) != 92:
            continue
        for feature in data[index]["features"]:
            new_row += f"{feature.split(': ')[0]};"
            features_order.append(feature.split(": ")[0])
    new_row = new_row[:-1]
    new_row += "\n"
    csv.append(new_row)
    for index, row in enumerate(tqdm(data)):
        if len(row["features"]) != 92:
            continue
        new_row = f"{data[index]['id']};{data[index]['label']};"
        for feature_index, feature in enumerate(data[index]["features"]):
            new_row += f"{feature.split(': ')[1]};"
            if features_order[feature_index] != feature.split(": ")[0]:
                print("Not in correct order!")
        new_row = new_row[:-1]
        new_row += "\n"
        csv.append(new_row)

    file_path = os.path.join(
        os.getcwd(),
        f"src/datasets/Phsyionet2021_5classes_{features_library}_extracted_features_original.csv",
    )
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(csv)

    # patient_features_dict = {}
    # for patient in data:
    #     patient_features_dict[patient["id"]] = patient["features"]


if __name__ == "__main__":
    main()
