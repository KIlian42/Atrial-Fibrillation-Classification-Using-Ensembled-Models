# Readme.md

Hi All, 

This repository shows code and research from my master thesis.

Master Thesis: "Transformer based ECG (electrocardiogram) classification" (2023-2024).

Please go to section [Setup](#Setup) for an installation guide to run this repository.

For all datasets and trained models, see: https://drive.google.com/drive/folders/1L_gOMrkygu2N0k97COYuVrmE-AwEEMoQ?usp=sharing


# Setup

#### Prerequisites

If not done yet,

1. ... install Git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
<br />
2. ... install Python 3.11: https://www.python.org/downloads/
<br />
3. ... install Pip (Package Installer Python): https://pip.pypa.io/en/stable/installation/

#### Installation

1. Open a folder or directory on your computer, where you want to save the project.

2. Open terminal on Mac OS/Linux or cmd (Command Prompt) on Windows.

3. Clone repository
```
git clone https://github.com/KIlian42/Atrial-Fibrillation-Classification-using-Transformer-models.git
```
4. Change directory
```
cd Atrial-Fibrillation-Classification-using-Transformer-models
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
