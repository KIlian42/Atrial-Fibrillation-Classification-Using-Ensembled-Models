# Readme.md

Please go to section [Setup](#Setup) for an installation guide to run this repository.

## Progress update

Kilian's Master Thesis: "Thesis Single-lead ECG classification based on Transformer models" (2023-2024).

This readme shows Kilian's master thesis progress bi-weekly.

### 1st - 14th September

n.a.

### 18th - 31th August

- Repository launch (26th August)
- Prototype implementation of a Transformer model using both an encoder & decoder layer (see folder v1) trained on the Physionet 2017 challenge ~ 63% accuracy.

# Setup

1. If not done yet, install Git CLI (Command Line Interface):
https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

2. Move to a directory on your computer, where you want to save the project.

3. Open terminal (Mac OS) or cmd (Windows Command Prompt): 
```
git clone https://github.com/KIlian42/Atrial-Fibrillation-Classification-using-Transformer-models.git
```
4. Change directory
```
cd Atrial-Fibrillation-Classification-using-Transformer-models
```
5. Requirement: Need to have Python 3.11 installed https://www.python.org/downloads/, see tutorials on youtube for installation help, then:
Create Python 3.11 environment:
```
py -3.11 -m venv .venv
```
6. Source enviroment
- On Mac OS:
```
source .venv/Scripts/activate
```
- On Windows:
```
.venv\Scripts\activate.bat
```