import h5py
from tqdm import tqdm

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def load_data():
    # ECG data
    h5file =  h5py.File("src/Datasets/Data/physio.h5", 'r')
    data_keys = list(h5file.keys())
    print("Found data:", len(data_keys))
    ecg_data = {}
    for key in tqdm(data_keys):
        data = h5file[key]['ecgdata']
        new_ecg_data = []
        for data_point in data:
            new_ecg_data.append(data_point[0])
        ecg_data[key] = new_ecg_data
    # Labels
    file = open("src/Data/Datasets/Physionet2017/REFERENCE.csv", "r")
    labels_txt = file.readlines()
    file.close()
    labels = {}
    for index, label in enumerate(labels_txt):
        splitted_label = label.replace("\n", "").split(",")
        labels[splitted_label[0]] = splitted_label[1]
    ID = []
    X = []
    Y = []
    for key in labels:
        if key in ecg_data and len(ecg_data[key]) == 9000:
            ID.append(key)
            x_formatted = ecg_data[key] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            x_formatted = [list(split_list(x_formatted, 95))]
            X.append(x_formatted)
            Y.append(labels[key])
    # print(len(ID), len(X), len(Y))
    # Sequence length
    # len_dict = {}
    # for x in X:
    #     if str(len(x)) not in len_dict:
    #         len_dict[str(len(x))] = 1
    #     else:
    #         len_dict[str(len(x))] += 1
    # print(len_dict)
    return ID, X, Y
    
if __name__ == '__main__':
    print("Start loading")
    ID, X, Y = load_data()
