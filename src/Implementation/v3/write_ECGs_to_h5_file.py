import os
import scipy.io
from tqdm import tqdm
import h5py

def normalize_to_01(values):
    max_val = max(values)
    min_val = min(values)
    # Handle the case where max_val and min_val are the same (to avoid division by zero)
    if max_val == min_val:
        return [0.5 for _ in values]  # or return [0 for _ in values] based on preference
    normalized_values = [(x - min_val) / (max_val - min_val) for x in values]
    return normalized_values

def normalize_to_11(values):
    max_val = max(values)
    min_val = min(values)
    # Handle the case where max_val and min_val are the same (to avoid division by zero)
    if max_val == min_val:
        return [0 for _ in values]
    normalized_values = [2 * (x - min_val) / (max_val - min_val) - 1 for x in values]
    return normalized_values

def load_physionet_data(directory):
    top_folders = []
    for root, dirs, files in os.walk(directory):
        for name in dirs:
            top_folders.append(os.path.join(root, name))
    # folders = []
    # for top_folder in top_folders:
    #     for root, dirs, files in os.walk(top_folder + '/'):
    #         for name in dirs:
    #             folders.append(os.path.join(root, name))

    ecgs_ids = []
    ecgs = []
    ecg_labels = []

    for directory in top_folders:
        # List all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.mat')]
        # Extract data from each file
        for file in tqdm(files, desc=directory):
            ecg_id = file[:-4]
            filepath = os.path.join(directory, file)
            # Load the .mat file
            mat_data = scipy.io.loadmat(filepath)
            # Assuming the data is stored under the key 'val', adjust if different
            data_array = mat_data['val']

            # Read header file
            filepath = os.path.join(directory, ecg_id + ".hea")
            f = open(filepath, "r", encoding="utf-8")
            header = f.readlines()
            f.close()
            labels = str(list(set(header[15][5:].replace("\n", "").strip().split(",")))).replace("[", "").replace("]", "").replace("'", "").replace(" ", "") + "\n"
            
            ecgs_ids.append(ecg_id)
            ecgs.append(list(data_array))
            # ecgs.append(normalize_to_11(list(data_array)))
            ecg_labels.append(labels)
        
    return ecgs_ids, ecgs, ecg_labels

# Example usage
if __name__ == '__main__':
    directory_path = 'src/Implementation/v3/downloadedData'
    ecgs_ids, ecgs, ecg_labels = load_physionet_data(directory_path)

    with h5py.File("physionet2020.h5", 'w') as h5f:
        for _, ecg in enumerate(zip(ecgs_ids, ecgs)):
            h5f.create_dataset(ecg[0], data=ecg[1])

    f = open("physionet2020-references.txt", "w", encoding="utf-8")
    f.writelines(ecg_labels)
    f.close()