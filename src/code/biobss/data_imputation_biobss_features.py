from tqdm import tqdm

def main():
    f = open("src/datasets/prepared/biobss_features.csv", "r", encoding="utf-8")
    data = f.readlines()
    f.close()
    data = list(map(lambda x: x.replace("\n", "").split(";"), data))

    # Load features in averages dict
    features_averages_dict = {}
    for patient in tqdm(data, desc="Load features in averages dict"):
        for feature in patient[1:]:
            feature_name = feature.split(": ")[0]
            feature_value = feature.split(": ")[1]
            if feature_value != "nan" and feature_value != "-inf" and feature_value != "inf":
                feature_value = float(feature_value)
            if feature_name not in features_averages_dict:
                if isinstance(feature_value, float): # ignore nans
                    features_averages_dict[feature_name] = [feature_value]
            else:
                if isinstance(feature_value, float):
                    features_averages_dict[feature_name].append(feature_value)

    # Calculate averages per feature 
    for feature_name in features_averages_dict:
        average = sum(features_averages_dict[feature_name]) / len(features_averages_dict[feature_name])
        features_averages_dict[feature_name] = round(average, 4)
    
    # rounded_numbers = [round(num) for num in numbers]

    # Imputate features
    for patient_index, patient in enumerate(data):
        for feature_index, feature in enumerate(patient):
            if feature_index == 0: # skip patient IDs
                continue
            feature_name = feature.split(": ")[0]
            feature_value = feature.split(": ")[1]
            if feature_value == "nan" or feature_value == "-inf" or feature_value == "inf":
                data[patient_index][feature_index] = feature_name + ": " + str(features_averages_dict[feature_name])
        data[patient_index] = ';'.join(str(feature) for feature in data[patient_index])
        data[patient_index] += "\n"
    
    with open('src/datasets/prepared/biobss_features_imputated.csv', 'w', encoding='utf-8') as file:
        file.writelines(data)


if __name__ == "__main__":
    main()