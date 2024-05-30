# /Users/kiliankramer/Desktop/Master Thesis/ImplementationNew/atrial-fibrillation-classification-using-transformer-models-original/src/Datasets/physionet2021_scoredLabels.h5

import pandas as pd
import h5py
from tqdm import tqdm
import os
from collections import Counter

def main():
    # df = pd.read_csv("src/datasets/Physionet2021_scoredLabels_mapping.csv", delimiter=",")
    # # => All column names
    # columns = df.columns
    # print(columns)
    # # => Convert specific columns to list
    # name = list(df['Dx'])
    # code = list(df['SNOMEDCTCode'])
    # abbreviation = list(df['Abbreviation'])
    # notes = list(df['Notes'])
    # for entry in zip(name, code, abbreviation, notes):
    #     print(f"{entry[0]},{entry[1]},{entry[2]},{entry[3]}")
    
    path = "/Users/kiliankramer/Desktop/Master Thesis/ImplementationNew/atrial-fibrillation-classification-using-transformer-models-original/src/Datasets/physionet2021_scoredLabels.h5"
    h5file = h5py.File(
        path,
        "r",
    )

    # Load the ECGs and their IDs to a dictionary X_dict.
    # This step is necessary to map all labels from the labels files in the step.
    IDs = list(h5file.keys())
    X_dict = {}
    for key in tqdm(IDs, desc="Load ECG data"):
        X_dict[key] = list(h5file[key][0])

    # Load the labels and their IDs to a dictionary Y_dict.
    # Some ECGs can contain multiple labels.
    labels_df = pd.read_csv(
        os.path.join(os.getcwd(), "src/datasets/physionet2021_references.csv"), sep=";"
    )
    Y_dict = {}
    pbar = tqdm(total=len(labels_df), desc="Load ECG labels")
    for _, row in labels_df.iterrows():
        Y_dict[row["id"]] = row["labels"].split(",")
        pbar.update(1)
    X = []
    Y = []
    Z = []
    for patient_id in tqdm(Y_dict, desc="Map labels to ECGs"):
        for label in Y_dict[patient_id]:
            try:
                X.append(X_dict[patient_id])
                Y.append(str(label))
                Z.append(str(patient_id))
            except:
                pass

    count = Counter(Y)
    print(count)
    
    # Map labels
    Y = [0 if x == "426783006" else x for x in Y] # sinus rhythm (SR)
    Y = [1 if x == "164889003" else x for x in Y] # atrial fibrillation (AF)
    Y = [2 if x == "164890007" else x for x in Y] # atrial flutter (AFL)
    Y = [3 if x == "284470004" or x == "63593006" else x for x in Y] # premature atrial contraction (PAC), supraventricular premature beats (SVPB)
    Y = [4 if x == "427172004" or x == "17338001" else x for x in Y] # premature ventricular contractions (PVC), ventricular premature beats (VPB)
    Y = [5 if x == "6374002" else x for x in Y] # bundle branch block (BBB)
    Y = [6 if x == "426627000" else x for x in Y] # bradycardia (Brady)
    Y = [7 if x == "733534002" or x == "164909002" else x for x in Y] # complete left bundle branch block (CLBBB), left bundle branch block (LBBB)
    Y = [8 if x == "713427006" or x == "59118001" else x for x in Y] # complete right bundle branch block (CRBBB), right bundle branch block (RBBB)
    Y = [9 if x == "270492004" else x for x in Y] # 1st degree av block (IAVB)
    Y = [10 if x == "713426002" else x for x in Y] # incomplete right bundle branch block (IRBBB)
    Y = [11 if x == "39732003" else x for x in Y] # left axis deviation (LAD)
    Y = [12 if x == "445118002" else x for x in Y] # left anterior fascicular block (LAnFB)
    Y = [13 if x == "251146004" else x for x in Y] # low qrs voltages (LQRSV)
    Y = [14 if x == "698252002" else x for x in Y] # nonspecific intraventricular conduction disorder (NSIVCB)
    Y = [15 if x == "10370003" else x for x in Y] # pacing rhythm (PR)
    Y = [16 if x == "365413008" else x for x in Y] # poor R wave Progression (PRWP)
    Y = [17 if x == "164947007" else x for x in Y] # prolonged pr interval (LPR)
    Y = [18 if x == "111975006" else x for x in Y] # prolonged qt interval (LQT)
    Y = [19 if x == "164917005" else x for x in Y] # qwave abnormal (QAb)
    Y = [20 if x == "47665007" else x for x in Y] # right axis deviation (RAD)
    Y = [21 if x == "427393009" else x for x in Y] # sinus arrhythmia (SA)
    Y = [22 if x == "426177001" else x for x in Y] # sinus bradycardia (SB)
    Y = [23 if x == "427084000" else x for x in Y] # sinus tachycardia (STach)
    Y = [24 if x == "164934002" else x for x in Y] # t wave abnormal (TAb)
    Y = [25 if x == "59931005" else x for x in Y] # t wave inversion (TInv)

if __name__ == "__main__":
    main()