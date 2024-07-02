# Official scored labels Physionet 2021: https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_scored.csv

# 0 = 426783006 -> sinus rhythm (SR)
# 1 = 164889003 -> atrial fibrillation (AF)
# 2 = 164890007 -> atrial flutter (AFL)
# 3 = 284470004 or 63593006 -> premature atrial contraction (PAC) or supraventricular premature beats (SVPB)
# 4 = 427172004 or 17338001 -> premature ventricular contractions (PVC), ventricular premature beats (VPB)
# 5 = 6374002 -> bundle branch block (BBB)
# 6 = 426627000 -> bradycardia (Brady)
# 7 = 733534002 or 164909002 -> complete left bundle branch block (CLBBB), left bundle branch block (LBBB)
# 8 = 713427006 or 59118001 -> complete right bundle branch block (CRBBB), right bundle branch block (RBBB)
# 9 = 270492004 -> 1st degree av block (IAVB)
# 10 = 713426002 -> incomplete right bundle branch block (IRBBB)
# 11 = 39732003 -> left axis deviation (LAD)
# 12 = 445118002 -> left anterior fascicular block (LAnFB)
# 13 = 251146004 -> low qrs voltages (LQRSV)
# 14 = 698252002 -> nonspecific intraventricular conduction disorder (NSIVCB)
# 15 = 10370003 -> pacing rhythm (PR)
# 16 = 365413008 -> poor R wave Progression (PRWP)
# 17 = 164947007 -> prolonged pr interval (LPR)
# 18 = 111975006 -> prolonged qt interval (LQT)
# 19 = 164917005 -> qwave abnormal (QAb)
# 20 = 47665007 -> right axis deviation (RAD)
# 21 = 427393009 -> sinus arrhythmia (SA)
# 22 = 426177001 -> sinus bradycardia (SB)
# 23 = 427084000 -> sinus tachycardia (STach)
# 24 = 164934002 -> t wave abnormal (TAb)
# 25 = 59931005 -> t wave inversion (TInv)

VALID_LABELS = set(
    [
        "164889003",
        "164890007",
        "6374002",
        "426627000",
        "733534002",
        "713427006",
        "270492004",
        "713426002",
        "39732003",
        "445118002",
        "164909002",
        "251146004",
        "698252002",
        "426783006",
        "284470004",
        "10370003",
        "365413008",
        "427172004",
        "164947007",
        "111975006",
        "164917005",
        "47665007",
        "59118001",
        "427393009",
        "426177001",
        "427084000",
        "63593006",
        "164934002",
        "59931005",
        "17338001",
    ]
)

SR_AF_AFL_PAC_PVC_labels = set(["426783006", "164889003", "164890007", "284470004", "427172004"])

SR_AF_AFL_labels = set(["426783006", "164889003", "164890007"])

arrhyhtmia_mapping_id_to_index = {
    "426783006": 0, # sinus rhythm (SR)
    "164889003": 1, # atrial fibrillation (AF)
    "164890007": 2, # atrial flutter (AFL)
    "284470004": 3, # premature atrial contraction (PAC)
    "63593006": 3, # supraventricular premature beats (SVPB)
    "427172004": 4, # premature ventricular contractions (PVC)
    "17338001": 4, # ventricular premature beats (VPB)
    "6374002": 5, # bundle branch block (BBB)
    "426627000": 6, # bradycardia (Brady)
    "733534002": 7, # complete left bundle branch block (CLBBB)
    "164909002": 7, # left bundle branch block (LBBB)
    "713427006": 8, # complete right bundle branch block (CRBBB)
    "59118001": 8, # right bundle branch block (RBBB)
    "270492004": 9, # 1st degree av block (IAVB)
    "713426002": 10, # incomplete right bundle branch block (IRBBB)
    "39732003": 11, # left axis deviation (LAD)
    "445118002": 12, # left anterior fascicular block (LAnFB)
    "251146004": 13, # low qrs voltages (LQRSV)
    "698252002": 14, # nonspecific intraventricular conduction disorder (NSIVCB)
    "10370003": 15, # pacing rhythm (PR)
    "365413008": 16, # poor R wave Progression (PRWP)
    "164947007": 17, # prolonged pr interval (LPR)
    "111975006": 18, # prolonged qt interval (LQT)
    "164917005": 19, # qwave abnormal (QAb)
    "47665007": 20,  # right axis deviation (RAD)
    "427393009": 21, # sinus arrhythmia (SA)
    "426177001": 22, # sinus bradycardia (SB)
    "427084000": 23, # sinus tachycardia (STach)
    "164934002": 24, # t wave abnormal (TAb)
    "59931005": 25 # t wave inversion (TInv)
}

def map_arrhyhtmia_id_to_index(x: str) -> int:
    return arrhyhtmia_mapping_id_to_index[x]

arrhyhtmia_mapping_index_to_id = {
    0: "426783006", # sinus rhythm (SR)
    1: "164889003", # atrial fibrillation (AF)
    2: "164890007", # atrial flutter (AFL)
    3: "284470004|63593006", # premature atrial contraction (PAC) | supraventricular premature beats (SVPB)
    4: "427172004|17338001", # premature ventricular contractions (PVC) | ventricular premature beats (VPB)
    5: "6374002", # bundle branch block (BBB)
    6: "426627000", # bradycardia (Brady)
    7: "733534002|164909002", # complete left bundle branch block (CLBBB) | left bundle branch block (LBBB)
    8: "713427006|59118001", # complete right bundle branch block (CRBBB) | right bundle branch block (RBBB)
    9: "270492004", # 1st degree av block (IAVB)
    10: "713426002", # incomplete right bundle branch block (IRBBB)
    11: "39732003", # left axis deviation (LAD)
    12: "445118002", # left anterior fascicular block (LAnFB)
    13: "251146004", # low qrs voltages (LQRSV)
    14: "698252002", # nonspecific intraventricular conduction disorder (NSIVCB)
    15: "10370003", # pacing rhythm (PR)
    16: "365413008", # poor R wave Progression (PRWP)
    17: "164947007", # prolonged pr interval (LPR)
    18: "111975006", # prolonged qt interval (LQT)
    19: "164917005", # qwave abnormal (QAb)
    20: "47665007",  # right axis deviation (RAD)
    21: "427393009", # sinus arrhythmia (SA)
    22: "426177001", # sinus bradycardia (SB)
    23: "427084000", # sinus tachycardia (STach)
    24: "164934002", # t wave abnormal (TAb)
    25: "59931005" # t wave inversion (TInv)
}

def map_arrhyhtmia_index_to_id(x: int) -> str:
    return arrhyhtmia_mapping_index_to_id[x]