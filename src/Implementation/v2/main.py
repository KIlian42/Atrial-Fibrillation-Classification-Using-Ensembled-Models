
import h5py

def main():
    # load af data
    path = 'af_save_path' # for ex. '/content/drive/MyDrive/ecg_data_full/af_full.h5'
    h5f = h5py.File(path,'r')
    af_array = h5f['af_tot'][:]
    h5f.close()
    # load normal data
    path = 'normal_save_path' # for ex. '/content/drive/MyDrive/ecg_data_full/normal_full.h5'
    h5f = h5py.File(path,'r')
    normal_array = h5f['normal_tot'][:]
    h5f.close()
    # can also load it to pd.DataFrame and drop any NaN values
    # df_af = pd.DataFrame(data=af_array)
    # df_af.dropna(inplace=True)

if __name__ == '__main__':
    main()