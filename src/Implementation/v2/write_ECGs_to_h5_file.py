import os
import scipy.io
from tqdm import tqdm
import ecg_plot
import h5py

def load_physionet_data(directory):
    ecgs_name = []
    ecgs = []
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.mat')]
    # Extract data from each file
    for file in tqdm(files):
        filepath = os.path.join(directory, file)
        # Load the .mat file
        mat_data = scipy.io.loadmat(filepath)
        # Assuming the data is stored under the key 'val', adjust if different
        data_array = mat_data['val'][0]

        # Only keep 9000 samples long ECGs
        # if len(data_array) == 9000:
        ecgs_name.append(file[:-4])
        ecgs.append(normalize_to_11(list(data_array)))
        
    return ecgs_name, ecgs

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

# Example usage
if __name__ == '__main__':
    directory_path = 'src/Datasets/Data/Physionet2017/'
    ecg_names, ecgs = load_physionet_data(directory_path)
        
    # Plot example    
    # ecg_plot.plot_1(ecgs[0], sample_rate=900, title = 'ECG', timetick=1)
    # ecg_plot.show()

    with h5py.File("physionet2017.h5", 'w') as h5f:
        for index, ecg in enumerate(zip(ecg_names, ecgs)):
            h5f.create_dataset(ecg[0], data=ecg[1])