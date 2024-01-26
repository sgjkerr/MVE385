import numpy as np
# Import h5py module tor read HDF5 files
import h5py
# Import moduels needed for the navigation in the file system
import os
import glob
import pickle
import yaml

yaml_file = 'molnet.yml'
yaml_path = '/../config/' + yaml_file
with open(os.path.dirname(__file__) + yaml_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
model = config['model']
min_wavelength = model['min_wavelength']
max_wavelength = model['max_wavelength']
resolution = model['resolution']

save_to_lst = ['./data/train/train.msp', './data/test/test.msp', './data/validation/validation.msp']

filepaths_lst = [['/../data/data_train.hdf5', '/../data/data_train_small.hdf5'],
             ['/../data/data_test_small.hdf5', '/../data/data_test.hdf5'],
             ['/../data/data_validation.hdf5', '/../data/data_validation_small.hdf5']]


def create_msp_file(mol_directory, filename):
    with open(mol_directory, 'rb') as f:
        mol_files = pickle.load(f)
    with open(filename, 'w') as f:
        for mol_file_dict in mol_files:
            title = mol_file_dict['title'].item()
            inchi = mol_file_dict['inchi'].item()
            inchikey = mol_file_dict['inchikey'].item()
            smiles = mol_file_dict['smiles'].item() 

            absorb_data = mol_file_dict['absorb_data']
            num_peaks = np.shape(absorb_data)[0]
            
            
            f.write('Name: ' + title + '\n')
            f.write('InChIKey: ' + inchikey + '\n')
            f.write('ExactMass: 285 \n')
            f.write('Comments: "computed SMILES=' + smiles + '" "InChI=' + inchi + '"\n')
            f.write('Num Peaks: ' + str(num_peaks) + '\n')
            for ind in range(num_peaks):
                f.write(str(absorb_data[ind][0]) + ' ' + str(absorb_data[ind][1]) + '\n')
            f.write('\n')  # Separate each molecule with an empty line

for ix ,filepaths in enumerate(filepaths_lst):
    save_to = save_to_lst[ix]
# Open h5py file
    data = []
    for filepath in filepaths:
        with h5py.File(os.path.dirname(__file__)  + filepath, 'r') as f:
            print("Number of molecules: ", len(f.keys()), " in file: ", filepath)
            for ind, mol in enumerate(f.keys()):
                molecule = f[mol]
                title_encoded = molecule['title'][...]
                title = np.char.decode(title_encoded, encoding='cp037')
                inchi_array = molecule['inchi'][...]
                inchi = np.char.decode(inchi_array[0], encoding='cp037')
                inchikey = np.char.decode(inchi_array[1], encoding='cp037')
                spectra_intensity = molecule['y_bar'][...]
                wavelength = np.arange(min_wavelength, max_wavelength+resolution, resolution)
                absorb_data = np.column_stack((wavelength, spectra_intensity))
                # Remove rows with zero intensity
                absorb_data = absorb_data[absorb_data[:,1] != 0]
                smiles_encoded = molecule['smiles_string'][...]
                smiles = np.char.decode(smiles_encoded, encoding='cp037')
                data.append({'title': title,
                            'inchi': inchi,
                            'inchikey': inchikey, 
                            'absorb_data': absorb_data,
                            'smiles': smiles})
    print('Number of molecules: ', len(data))
    # Save the array using pickle
    with open('./data/tmp.pkl', 'wb') as f:
        pickle.dump(data, f)


    create_msp_file('./data/tmp.pkl', save_to)