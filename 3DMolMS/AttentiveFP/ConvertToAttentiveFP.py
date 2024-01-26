import numpy as np
# Import h5py module tor read HDF5 files
import h5py
# Import moduels needed for the navigation in the file system
import os
import pickle

save_to_lst = ['./data/train/train.pkl', './data/test/test.pkl', './data/validation/validation.pkl']

filepaths_lst = [['../data/data_train.hdf5', '../data/data_train_small.hdf5'],
             ['../data/data_test_small.hdf5', '../data/data_test.hdf5'],
             ['../data/data_validation.hdf5', '../data/data_validation_small.hdf5']]

for i ,filepaths in enumerate(filepaths_lst):
    save_path = save_to_lst[i]

    # Open h5py file
    data = []
    smiles_to_mol = {}
    ind = 0
    for filepath in filepaths:
        with h5py.File(filepath, 'r') as f:
            print("Number of molecules in file: ", len(list(f.keys())))
            for mol in list(f.keys()):
                molecule = f[mol]
                encoded_str = molecule['smiles_string'][...]
                row = np.array([encoded_str])

                spectra = molecule['y'][...]
                title = molecule['title'][...]
                title_str = np.char.decode(title, encoding='cp037')
                decoded_str = np.char.decode(encoded_str, encoding='cp037')
                smiles_to_mol[ind] = [title_str, decoded_str]
                row = np.concatenate((row, np.transpose(spectra)))
                data.append(row)
                ind = ind + 1
    print('Number of molecules: ', len(data))

    data = np.array(data)

    # Save the array using pickle
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


