import os
import scipy.io
import ecg_plot

def normalize_to_range(values):
    max_val = max(values)
    min_val = min(values)
    # Handle the case where max_val and min_val are the same (to avoid division by zero)
    if max_val == min_val:
        return [0 for _ in values]
    normalized_values = [2 * (x - min_val) / (max_val - min_val) - 1 for x in values]
    return normalized_values

def load_ecg_from_mat(filepath):
    """
    Load ECG data from a .mat file.
    
    Args:
    - filepath (str): Path to the .mat file.
    
    Returns:
    - numpy array containing the ECG data.
    """
    mat_data = scipy.io.loadmat(filepath)
    # Assuming the data is stored under the key 'val', adjust if different
    ecg_data = mat_data['val'][0]
    return ecg_data

def plot_ecg_from_file(filepath, sampling_rate=300):
    """
    Plot ECG data from a given .mat file using ecg_plot.
    
    Args:
    - filepath (str): Path to the .mat file.
    - sampling_rate (int): Sampling rate of the ECG data. Default is 300 Hz.
    """
    ecg_data = load_ecg_from_mat(filepath)
    ecg_data = normalize_to_range(ecg_data)
    ecg_plot.plot_1(ecg_data, sample_rate=sampling_rate, title=os.path.basename(filepath))
    ecg_plot.show()

# Example usage
filepath = 'src/Datasets/Data/Physionet2017/A00001.mat'
plot_ecg_from_file(filepath)
