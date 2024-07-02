import scipy
from scipy.signal import butter, lfilter

def pad_or_truncate_ecg(ecg: list, max_samples: int) -> list:
    return ecg[:max_samples] + [0] * (max_samples - len(ecg))

def split_list_into_n_sublists(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def normalize_to_minus11(ecg: list):
    max_val = max(ecg)
    min_val = min(ecg)
    # Handle the case where max_val and min_val are the same (to avoid division by zero)
    if max_val == min_val:
        return [0 for _ in ecg]
    normalized_values = [2 * (x - min_val) / (max_val - min_val) - 1 for x in ecg]
    return normalized_values

def resample_ecg(ecg: list, resample: int):
    return list(scipy.signal.resample(ecg, resample, t=None, axis=0, window=None, domain="time"))

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(ecg: list, lowcut: float, highcut: float, sampling_rate: int, order: int =4):
    b, a = butter_bandpass(lowcut, highcut, sampling_rate, order=order)
    return lfilter(b, a, ecg)