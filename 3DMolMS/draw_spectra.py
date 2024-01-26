# Draw spectrum of a given file

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import yaml


def draw_spectrum_direct(pred_spectra_smooth, true_spectra, title_str, max_wavelength, min_wavelength, resolution):
    x = np.arange(min_wavelength, max_wavelength, resolution)

    title = np.count_nonzero(true_spectra_bar)

    print(title_str)
    plt.plot(x, true_spectra, color='red', label="True smooth")

    plt.plot(x, pred_spectra_smooth, color='blue', label= "Prediction smooth")
    plt.title(str(title)+" | "+title_str[0])
    plt.legend()
    plt.show()
    return

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Draw spectrum of a given file')
    parser.add_argument('--train_data', type=str, default='./data/data_train.hdf5',
                        help='path to training data (pkl)')
    parser.add_argument('--mol_num', type=int, default=5,
                        help='mol number to draw')
    parser.add_argument('--model_config_path', type=str, default='./config/molnet.yml',
                        help='path to model and training configuration')

    args = parser.parse_args()
    
    with open(args.model_config_path, 'r') as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('Load the model & training configuration from {}'.format(args.model_config_path))
    
    spectra = load_mol_spectra(args.train_data, args.mol_num)
    
    
    draw_spectrum(spectra, config['model'])
    
    