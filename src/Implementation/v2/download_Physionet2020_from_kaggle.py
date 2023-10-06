import os
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    # Load API key
    kaggle_config_dir = os.path.expanduser("~/.kaggle")
    if not os.path.exists(kaggle_config_dir):
        os.makedirs(kaggle_config_dir)
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_dir

    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download datasets
    api.dataset_download_files('bjoernjostein/china-12lead-ecg-challenge-database', path='src/Implementation/v2/downloadedData', unzip=False)
    api.dataset_download_files('bjoernjostein/china-physiological-signal-challenge-in-2018', path='src/Implementation/v2/downloadedData', unzip=False)
    api.dataset_download_files('bjoernjostein/georgia-12lead-ecg-challenge-database', path='src/Implementation/v2/downloadedData', unzip=False)
    api.dataset_download_files('bjoernjostein/ptb-diagnostic-ecg-database', path='src/Implementation/v2/downloadedData', unzip=False)
    api.dataset_download_files('bjoernjostein/ptbxl-electrocardiography-database', path='src/Implementation/v2/downloadedData', unzip=False)
    api.dataset_download_files('bjoernjostein/st-petersburg-incart-12lead-arrhythmia-database', path='src/Implementation/v2/downloadedData', unzip=False)
    # api.dataset_download_files('bjoernjostein/physionet-snomed-mappings', path='src/Implementation/v2/downloadedData', unzip=False)
    # api.dataset_download_files('bjoernjostein/physionet-challenge-models', path='src/Implementation/v2/downloadedData', unzip=False)
    # Download output files from Kaggle
    # kaggle kernels output bjoernjostein/physionet-challenge-2020 -p ./bjoernjostein

if __name__ == '__main__':
    main()

