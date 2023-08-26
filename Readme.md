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

#### Prerequisites
If not done yet, install Git CLI (Command Line Interface):
https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
If not done yet, install Python 3.11: https://www.python.org/downloads/
If not done yet, install Pip (Package Installer Python): https://pip.pypa.io/en/stable/installation/

#### Installation

Open a folder or directory on your computer, where you want to save the project.

Open terminal on Mac OS or cmd (Command Prompt) on Windows.

Clone Repository
```
git clone https://github.com/KIlian42/Atrial-Fibrillation-Classification-using-Transformer-models.git
```
Change directory
```
cd Atrial-Fibrillation-Classification-using-Transformer-models
```
Install library to create virtual environments
```
pip install virtualenv
```
Create Python 3.11 environment
> Mac OS:
```
python3.11 -m venv .venv
```
> Windows:
```
py -3.11 -m venv .venv
```
Source enviroment
> Mac OS:
```
source .venv/bin/activate
```
> Windows:
```
.venv\Scripts\activate.bat
```
Install requirements
```
pip install -r requirements.txt
```