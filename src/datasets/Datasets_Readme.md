# Datasets.md

To download the datasets to pre-train the models for this master thesis, please follow these steps:

1. Download the Physionet 2017 dataset from this link (this may take a while):
https://physionet.org/static/published-projects/challenge-2017/af-classification-from-a-short-single-lead-ecg-recording-the-physionetcomputing-in-cardiology-challenge-2017-1.0.0.zip

2. Download the Physionet 2020 dataset from this link (this may take a while):
https://physionet.org/static/published-projects/challenge-2020/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip

3. Unzip the files.

4. Rename the unzipped folders to Physionet2017 and Physionet2020.

5. Place the folders inside src/Datasets/Data.

6. (Optional) If you want to save memory delete the folders "source" in Physionet2017 and Physionet2020, since they contain all the code and models from the participants of these challenges.

7. (Not working yet) Open terminal/cmd and navigate with cd inside the folder src/Implementation/util and the run the script convert_physionet_data_to_single_h5_file.py with:
```
python convert_physionet_data_to_single_h5_file.py --target_direction "AbsolutePathToFolder/Physionet2017"
python convert_physionet_data_to_single_h5_file.py --target_direction "AbsolutePathToFolder/Physionet2020"
```
This will convert the Physionet challenges data into single .h5 files and place them inside the same folder (src/Datasets/Data) and also extract/create the files with the labels (physionet2017_references.csv and physionet2020_references.csv).