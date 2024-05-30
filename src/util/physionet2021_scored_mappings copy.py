import pandas as pd

def main():
    df = pd.read_csv("src/datasets/Physionet2021_scoredLabels_mapping.csv", delimiter=",")
    
    # => All column names
    columns = df.columns
    print(columns)

    # => Convert specific columns to list
    name = list(df['Dx'])
    code = list(df['SNOMEDCTCode'])
    abbreviation = list(df['Abbreviation'])
    notes = list(df['Notes'])

    for entry in zip(name, code, abbreviation, notes):
        # print(f"{entry[0]},{entry[1]},{entry[2]},{entry[3]}")
        print(f"{entry[1]},{entry[2]},{entry[3] if entry[3] != "nan"},{entry[0]}")

    # # => Iterate table
    # for index, row in df.iterrows():
    #     column1_value = row["column1_name"]
    #     column2_value = row["column2_name"]

if __name__ == "__main__":
    main()

# # Map labels
# Y = [0 if x == "426783006" else x for x in Y]  # Sinus Rhythm (SR)
# Y = [1 if x == "164889003" else x for x in Y]  # Atrial Fibrillation (AF)
# Y = [2 if x == "164890007" else x for x in Y]  # Atrial Flutter (AFL)
# Y = [3 if x == "284470004" else x for x in Y]  # Premature Atrial Contraction (PAC)
# Y = [4 if x == "427172004" else x for x in Y]  # Premature Ventricular Contraction (PVC)
# Y = [5 if x == "6374002" else x for x in Y]    # Bundle Branch Block
# Y = [6 if x == "426627000" else x for x in Y]  # Bradycardia
# Y = [7 if x == "733534002" else x for x in Y]  # Coronary Heart Disease
# Y = [8 if x == "713427006" else x for x in Y]  # Complete Right Bundle Branch Block
# Y = [9 if x == "270492004" else x for x in Y]  # 1st Degree AV Block
# Y = [10 if x == "713426002" else x for x in Y] # Incomplete Right Bundle Branch Block
# Y = [11 if x == "39732003" else x for x in Y]  # Left Axis Deviation
# Y = [12 if x == "445118002" else x for x in Y] # Left Anterior Fascicular Block
# Y = [13 if x == "164909002" else x for x in Y] # Left Bundle Branch Block
# Y = [14 if x == "251146004" else x for x in Y] # Low QRS Voltages
# Y = [15 if x == "698252002" else x for x in Y] # Nonspecific Intraventricular Conduction Disorder
# Y = [16 if x == "10370003" else x for x in Y]  # Pacing Rhythm
# Y = [17 if x == "365413008" else x for x in Y] # Inferior Ischaemia
# Y = [18 if x == "164947007" else x for x in Y] # Prolonged PR Interval
# Y = [19 if x == "111975006" else x for x in Y] # Prolonged QT Interval
# Y = [20 if x == "164917005" else x for x in Y] # Q Wave Abnormal
# Y = [21 if x == "47665007" else x for x in Y]  # Right Axis Deviation
# Y = [22 if x == "59118001" else x for x in Y]  # Right Bundle Branch Block
# Y = [23 if x == "427393009" else x for x in Y] # Sinus Arrhythmia
# Y = [24 if x == "426177001" else x for x in Y] # Sinus Bradycardia
# Y = [25 if x == "427084000" else x for x in Y] # Sinus Tachycardia
# Y = [26 if x == "63593006" else x for x in Y]  # Supraventricular Premature Beats
# Y = [27 if x == "164934002" else x for x in Y] # T Wave Abnormal
# Y = [28 if x == "59931005" else x for x in Y]  # T Wave Inversion
# Y = [29 if x == "17338001" else x for x in Y]  # Ventricular Premature Beats

# 164889003,AF,nan,atrial fibrillation
# 164890007,AFL,nan,atrial flutter
# 6374002,BBB,nan,bundle branch block
# 426627000,Brady,nan,bradycardia
# 733534002,CLBBB,We score 733534002 and 164909002 as the same diagnosis,complete left bundle branch block
# 713427006,CRBBB,We score 713427006 and 59118001 as the same diagnosis.,complete right bundle branch block
# 270492004,IAVB,nan,1st degree av block
# 713426002,IRBBB,nan,incomplete right bundle branch block
# 39732003,LAD,nan,left axis deviation
# 445118002,LAnFB,nan,left anterior fascicular block
# 164909002,LBBB,We score 733534002 and 164909002 as the same diagnosis,left bundle branch block
# 251146004,LQRSV,nan,low qrs voltages
# 698252002,NSIVCB,nan,nonspecific intraventricular conduction disorder
# 426783006,NSR,nan,sinus rhythm
# 284470004,PAC,We score 284470004 and 63593006 as the same diagnosis.,premature atrial contraction
# 10370003,PR,nan,pacing rhythm
# 365413008,PRWP,nan,poor R wave Progression
# 427172004,PVC,We score 427172004 and 17338001 as the same diagnosis.,premature ventricular contractions
# 164947007,LPR,nan,prolonged pr interval
# 111975006,LQT,nan,prolonged qt interval
# 164917005,QAb,nan,qwave abnormal
# 47665007,RAD,nan,right axis deviation
# 59118001,RBBB,We score 713427006 and 59118001 as the same diagnosis.,right bundle branch block
# 427393009,SA,nan,sinus arrhythmia
# 426177001,SB,nan,sinus bradycardia
# 427084000,STach,nan,sinus tachycardia
# 63593006,SVPB,We score 284470004 and 63593006 as the same diagnosis.,supraventricular premature beats
# 164934002,TAb,nan,t wave abnormal
# 59931005,TInv,nan,t wave inversion
# 17338001,VPB,We score 427172004 and 17338001 as the same diagnosis.,ventricular premature beats