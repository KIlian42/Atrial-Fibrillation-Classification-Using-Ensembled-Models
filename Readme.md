# Readme.md

Please go to section [Setup](#Setup) for an installation guide to run this repository.

## Progress update

Kilian's Master Thesis: "Single-lead ECG classification based on Transformer models" (2023-2024).

This readme shows Kilian's master thesis progress updates each two weeks.

### 15th - 28th September

#### Adaption research questions & thesis plan for related research:

##### Research quetions old:

1. How to design the encoding layer of the Transformer model for single-lead ECG classification (presence of atrial fibrillation) and how does the Transformer perform compared to a Transformer trained on extracted features from the signal?

2. How well does a decoder-only Transformer-based approach perform compared to an encoder-only and encoder- and decoder-based architecture?

3. How well do the Transformer models perform on long-term single-lead ECGs?

4. Can the Transformer model achieve 90%< accuracy on the PhysioNet 2017 challenge?


##### Research questions new:

1. How to design and improve the Transformer architecture to enable transfer learning/fine-tuning for different ECG tasks and datasets monitored with different systems/number of leads/sample rates/durations/noise filters with the focus on AF related beat classification? (new)

2. How to design the encoding layer of the Transformer model for ECG AF classification and how does the Transformer perform compared to a Transformer trained on extracted features from the signal? (reformulated)

3. How well can a decoder-only Transformer-based approach predicts the next AF episode? (adapted/specified)

4. How well do the Transformer model perform on long-term ECGs? (same)

5. Can the Transformer model achieve 90%< accuracy on the PhysioNet 2017 challenge? (same)

#### Research Datasets Physionet 2004, Physionet 2020 (distinction Atrial Flutter/Atrial Fibrillation)

#### Research ECG encoding strategies and feature extraction

#### Research Transformer functionality (positional encoding and query, key, value matrices)

### 1st - 14th September

#### Research - Papers (see research folder for related papers)

### 18th - 31th August

#### Repository launch (26th August)
#### First try/prototype implementation of a Transformer model (based on a VisionTransformer for MNIST: https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c) using both an encoder & decoder layer (see folder src > Implementation > Models > v1) trained on the Physionet 2017 challenge for 10 epochs ~ 63.88% accuracy.

# Setup

#### Prerequisites
If not done yet, install Git:
https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
<br />
If not done yet, install Python 3.11: https://www.python.org/downloads/
<br />
If not done yet, install Pip (Package Installer Python): https://pip.pypa.io/en/stable/installation/
<br /><br />
Please go to ... to download all trained models and to ... to download all necessary datasets for this repository.

#### Installation

1. Open a folder or directory on your computer, where you want to save the project.

2. Open terminal on Mac OS or cmd (Command prompt) on Windows.

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
> Mac OS:
```
python3.11 -m venv .venv
```
> Windows:
```
py -3.11 -m venv .venv
```
7. Source enviroment
> Mac OS:
```
source .venv/bin/activate
```
> Windows:
```
.venv\Scripts\activate.bat
```
8. Install requirements
```
pip install -r requirements.txt
```
