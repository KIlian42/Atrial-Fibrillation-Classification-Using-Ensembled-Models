# Readme.md

Please go to section [Setup](#Setup) for an installation guide to run this repository.

## Progress update

Kilian's Master Thesis: "Single-lead ECG classification based on Transformer models" (2023-2024).

This readme shows Kilian's master thesis progress updates each two weeks.

### 14th October - 27th October

1. Trained a baseline Residual CNN model on Physionet 2017 with 82% validation accuracy.
2. Trained a baseline Residual CNN model on Physionet 2020 (sinus, fibrillation, flutter) with 90% validation accuracy.
<br />-> for current results see plots in src/Implementation/v3/results
3. Accessed and settep up UM data (no ECG folders found, can not access them)
4. Updated src/Research/Research_Readme.md and currently investigating them closer
<br />Notices from last meeting:
- Setup UM Gitlab Repository for this project
- Split data (training/tests) by patients
- Take care of sampling rate (i.e. microvolt vs milivolt sample rates)
- Consecutive order of recordings might be important for analysis
- Try wavelet transformations
- Have a look at ICU challenge data
- Further specify research questions:
-> i.e. Can a Transformer model capture spatial information from the 12-lead ECG signals? (encoding each lead as a query -> investigate the relations among the leads)
-> i.e. Can ensemble models perform better than one big model on ECG classification (i.e. stacking of TCNN and Transformer ... SVM, LSTM)
-> notice: I will further specify the research questions during my readings through the papers any my model design
<br />Research questions so far (question 2 is partly covered in 1 -> needs to be reformulated):
1. How to design and improve the Transformer architecture to enable transfer learning or fine-tuning for different ECG tasks and datasets monitored through different systems, number of leads, sample rates, durations and noise filters, with the focus on AF related beat classification?
2. How to design the encoding layer of the Transformer model for ECG AF classification and how does the Transformer perform compared to a Transformer trained on extracted features from the signal?
3. How well can a decoder-only Transformer-based approach predicts the next AF episode?
4. How well do the Transformer model perform on long-term ECGs?
5. Can the Transformer model achieve 90%< accuracy on the PhysioNet 2017 challenge?

#### Next steps planned:
-> Repostiory will be configured under private. (need of your Github accounts to add contributors)
<br />-> Training of the baseline Residual CNN model on UM data.
<br />-> Further investigate all collected papers on ECG Transformer models (see proposal src/Research/Research_Readme.md) with focus on different approaches, identify/speficy limitations/weaknesses (research questions may be further specified during this step)
<br />-> Model adaption using Transformer encoder-based blocks and also train a baseline Temporal Convolutional network -> build solution models with PyTorch instead Keras 
<br />-> Investigatation of interesting library for deep learning forecasting, called "Darts", collects bunch of models, i.e. TCNN and Transformer models for timeseries forecasting: https://github.com/unit8co/darts
<br />-> Investigatation and implementation of different feature extraction methods, i.e. filters and spectograms, an interesting work here is: https://github.com/awerdich/physionet
<br />-> Implementation of hyperparameter tuning, cross-validation and different measure metrics (i.e. F1 score, Precision, Recall)
<br />-> Evaluation and work on thesis (create model/metrics plots) and write down (research) findings so far
<br />-> Possible more data collection on Atrial Fibrillation vs. Atrial Flutter databases, a paper here: https://www.medrxiv.org/content/10.1101/2023.08.08.23293815v1.full
<br />-> Investigation of ECG data augmentation techniques
<br />-> (Optional, if needed) Prepare training script with CUDA enabled to pre-train best model (parameter settings/input choice) using Google Colab with a larger A100 GPU -> to later fine-tune it on UM data using the UM cluster (if accessible)

### 29th September - 13th October

1. Downloaded, investigated Physionet 2020 ECG data and preprocessed it into .h5 files for training.
<br />-> See all code in src/Implementation/v2
<br />-> First run download_Physionet2020_from_kaggle.py, then write_ECGS_to_h5_file.py and then train.py
2. Adds Transformer (encoder-only) implementation in Keras and adaptable stacked block size.
<br /-> Findings/Conclusion: model did not learn, because stacking of convolutional layers were not done properly

### 15th - 28th September

#### Adaption research questions & thesis plan for related research:

##### Research quetions old:

1. How to design the encoding layer of the Transformer model for single-lead ECG classification (presence of atrial fibrillation) and how does the Transformer perform compared to a Transformer trained on extracted features from the signal?

2. How well does a decoder-only Transformer-based approach perform compared to an encoder-only and encoder- and decoder-based architecture?

3. How well do the Transformer models perform on long-term single-lead ECGs?

4. Can the Transformer model achieve 90%< accuracy on the PhysioNet 2017 challenge?

##### Research questions new:

1. How to design and improve the Transformer architecture to enable transfer learning or fine-tuning for different ECG tasks and datasets monitored through different systems, number of leads, sample rates, durations and noise filters, with the focus on AF related beat classification? (new)

2. How to design the encoding layer of the Transformer model for ECG AF classification and how does the Transformer perform compared to a Transformer trained on extracted features from the signal? (reformulated)

3. How well can a decoder-only Transformer-based approach predicts the next AF episode? (adapted/to be specified)

4. How well do the Transformer model perform on long-term ECGs? (same)

5. Can the Transformer model achieve 90%< accuracy on the PhysioNet 2017 challenge? (same)

#### Research potential Physionet Challenges Datasets

##### Physionet 2021: "Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021"

-> Objective to classify and compare 12-lead, 6-lead, 4-lead, 3-lead, 2-lead (short-term: 6 and 144 seconds) ECGs

##### Physionet 2020: "Classification of 12-lead ECGs: The PhysioNet/Computing in Cardiology Challenge 2020"

-> Dataset contains distinction of Atrial Flutter and Atrial Fibrillation

##### Physionet 2017: "AF Classification from a Short Single Lead ECG Recording: The PhysioNet/Computing in Cardiology Challenge 2017"

-> Single-lead short-term (1 minute) ECGs dataset for classification of Sinus Rythm, Atrial Flutter, Noise rythm and Other

##### Physionet 2016: "Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016"

// only contains sounds but focuses on AF classification

##### Physionet 2014: "Robust Detection of Heart Beats in Multimodal Data: The PhysioNet/Computing in Cardiology Challenge 2014"

-> Long-term ECGs (10 minutes) for QRS beat detection

##### Physionet 2013: "Noninvasive Fetal ECG: The PhysioNet/Computing in Cardiology Challenge 2013"

// (focuses on fetal ECGs)

##### Physionet 2011: "Improving the Quality of ECGs Collected using Mobile Phones: The PhysioNet/Computing in Cardiology Challenge 2011"

-> 12-lead short-term ECGs to classify signal quality: A (0.95): Excellent, B (0.85): Good, C (0.75): Adequate, D (0.60): Poor, F (0): Unacceptable

##### Physionet 2010: "Mind the Gap: The PhysioNet/Computing in Cardiology Challenge 2010"

-> Contains long-term ECGS (10 minutes) and 30 seconds signal gaps at the end -> predict/reconstruct signal

##### Physionet 2009: "Predicting Acute Hypotensive Episodes: The PhysioNet/Computing in Cardiology Challenge 2009"

-> Contains 5000 ECG -> predict which patients in the challenge dataset will experience an acute hypotensive episode beginning within the forecast window

##### Physionet 2008: "Detecting and Quantifying T-Wave Alternans: The PhysioNet/Computing in Cardiology Challenge 2008"

-> Diverse dataset mit differen lead- and length-recordings (monstly long-term) for classification of T-wave alternans

##### Physionet 2006: "QT Interval Measurement: The PhysioNet/Computing in Cardiology Challenge 2006"

-> QT Interval detection using 12-lead short-term (2 minutes) ECGs and 3 Frank (XYZ) leads

##### Physionet 2004: "Spontaneous Termination of Atrial Fibrillation: The PhysioNet/Computing in Cardiology Challenge 2004"

-> Predict AF termination using 2-lead ECGs in 1 minute length

##### Physionet 2003: "Distinguishing Ischemic from Non-Ischemic ST Changes: The PhysioNet/Computing in Cardiology Challenge 2003"

-> 2- and 3-lead long-term (20-24 hour) ECGs classifying the ST changes (events) in the test set as ischemic or non-ischemic

##### Physionet 2002: "RR Interval Time Series Modeling: The PhysioNet/Computing in Cardiology Challenge 2002"

// modified RR time series for forecasting

##### Physionet 2001: "Predicting Paroxysmal Atrial Fibrillation/Flutter: The PhysioNet/Computing in Cardiology Challenge 2001"

-> 1-lead 24-hour ECGs classifying Paroxysmal Atrial Fibrillation/Flutter

##### Physionet 2000: "Detecting and Quantifying Apnea Based on the ECG: The PhysioNet/Computing in Cardiology Challenge 2000"

-> 1-lead 8-hour ECGs for sleep apnea classification per minute

#### Research Transformer functionality (positional encoding and query, key, value matrices)

#### Research ECG encoding strategies and feature extraction

### 1st - 14th September

#### Research - read papers (see research folder for related papers)

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

2. Open terminal on Mac OS/Linux or cmd (Command prompt) on Windows.

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
.venv\Scripts\activate.bat
```
8. Install requirements
```
pip install -r requirements.txt
```
