# Atrial Fibrillation classification using ensembled models

This repository contains the research and code (Python) from my master thesis. (2023-2024)

Title: "Atrial Fibrillation classification using ensembled models" (feature-based and deep-learning models)

The research utilizes electrocardiogram (ECG) data from the Physionet 2021 challenge: https://physionet.org/content/challenge-2021/1.0.3/.

# Start

You can directly download the challenge data from the Physionet 2021 homepage: https://physionet.org/content/challenge-2021/1.0.3/#files,
or you might have look at my Google Drive, where I wrapped all the ECG files in a single .h5 (key-value) file:
https://drive.google.com/drive/folders/1e0LygPzn5tM9i2m2leXFwSs5J5og97W5
(do download physionet2021.h5, physionet2021_references.csv (labels) and codes_SNOMED.csv).

About the data:

Each ECG in physionet2021.h5 is stored as key (str): "PATIENT-ID" and value: 12-lead ECG (list[12 lists]), where each ECG has 12-leads and each lead is paddded/truncated to 5000 samples (10 second long ECGs with a sampling rate of 500Hz). The ECGs can have multiple labels, see Physionet2021_references.csv. The corresponding arrythmia label names are in the codes_SNOMED.csv listed. Notice, in my Google Drive is also the Physionet 2017 challenge data available and in the  directory "prepared" are further prepared datasets, since this thesis focuses particularly on the arrythmia types Sinus Rhythm (SR), Atrial Fibrilliation (AF), Atrial Flutter (AFL), Premature Atrial Contraction (PAC) and Premature Ventricular Contractions (PVC). Although, the Physionet 2017 labels do only distinguish between Sinus Rhythm, Atrial Fibrillation, Noise and Other, which might be useful for prototyping, the Physionet 2021 labels  on the other hand distinguish between more than 100 arrythmia types. 

Please go to section [Setup](#Setup) for an installation guide to run this repository.

<img width="700" alt="https://en.wikipedia.org/wiki/Electrocardiography" src="https://github.com/KIlian42/Atrial-Fibrillation-Classification-Using-Ensembled-Models/assets/57774167/1a2b2533-3aae-4876-8f32-2c24ce4cc90e">
<br />Source: https://en.wikipedia.org/wiki/Electrocardiography

# Setup

#### Prerequisites

If not done yet, install

1. ... Git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
2. ... Python 3.11: https://www.python.org/downloads/
3. ... Pip (Package Installer Python): https://pip.pypa.io/en/stable/installation/

#### Installation

1. Open a folder or directory on your computer, where you want to save the project.

2. Open terminal on Mac OS/Linux or cmd (Command Prompt) on Windows.

3. Clone repository
```
git clone https://github.com/KIlian42/Atrial-Fibrillation-Classification-Using-Ensembled-Models.git
```
4. Change directory
```
cd Atrial-Fibrillation-Classification-Using-Ensembled-Models
```
5. Install library to create virtual environments
```
pip install virtualenv
```
6. Create Python 3.11 environment
> Mac OS/Linux:
```
python3.11 -m venv .venv
```
> Windows:
```
py -3.11 -m venv .venv
```
7. Source enviroment
> Mac OS/Linux:
```
source .venv/bin/activate
```
> Windows:
```
.venv\Scripts\activate
```
8. Install requirements
```
pip install -r requirements.txt
```
