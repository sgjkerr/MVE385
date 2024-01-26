import os
import tarfile
import argparse
import yaml
import multiprocessing
import numpy as np
import h5py
import time
import random
from collections import defaultdict

from utils_small_set import get_xyz_from_gen_file, get_mol_attributes_file, get_mol_spectra, get_gross_charge

    
def build_molecule_array(mol, config):
    mol['mol'] = np.hstack((mol['mol'], mol['smiles_attributes']))
    mol['mol'] = np.pad(mol['mol'], ((0, config['encoding']['max_atom_num']-mol['mol'].shape[0]), (0, 0)), constant_values=0)

    return mol

def construct_h5py_mol(mol_id, source_file, config, queue, lock):
    mol_dir = os.path.join(source_directory, mol_id[0])
    # Read file names in the directory mol_dir
    file_names = os.listdir(mol_dir)    
    molecules = {}
    no_data = True
    # if file_names is empty, print warning and end function
    if not file_names or len(file_names) != 5:
        return 1
    for file_name in file_names:
        file_path = os.path.join(mol_dir, file_name)
        data_group = mol_id[1]
        if data_group not in molecules:
            molecules[data_group] = {}
        name = mol_id[0]
        if name not in molecules[data_group]:
            molecules[data_group][name] = {'title':name}
            molecules[data_group][name]['features'] = None
            molecules[data_group][name]['mol'] = None
            molecules[data_group][name]['y'] = None
            molecules[data_group][name]['y_bar'] = None
            molecules[data_group][name]['center_of_mass'] = None
            molecules[data_group][name]['smiles_attributes'] = None
            molecules[data_group][name]['smiles_string'] = None
            molecules[data_group][name]['inchi'] = None
            no_data = False

        data = []
        if file_path.endswith('detailed.out'):
            try:
                data = get_gross_charge(data, file_path, config['encoding'])
                if not data:
                    print("No detailed.out found for mol_id: ", file_path, " in file: ", mol_dir)
                    continue
                molecules[data_group][name]['features'] = np.array(data)
            except:
                print("Error in detailed.out for mol_id: ", file_path, " in file: ", mol_dir)
                return 1

        elif file_path.endswith('.DAT'):
            try:
                # Open "EXC.DAT" and read its content
                data, data_bar, center_of_mass = get_mol_spectra(data, file_path, config['encoding'])
                if not data:
                    print("No EXC.DAT found for mol_id: ", file_path, " in file: ", mol_dir)
                    continue
                data = np.array(data)
                molecules[data_group][name]['y'] = np.sqrt(data)
                molecules[data_group][name]['y_bar'] = np.sqrt(np.array(data_bar))
                molecules[data_group][name]['center_of_mass'] = center_of_mass
            except:
                print("Error in EXC.DAT for mol_id: ", file_path, " in file: ", mol_dir)
                return 1
        elif file_path.endswith('.pdb'):
            try:
                data, smiles_string, inchi, inchikey = get_mol_attributes_file(data, file_path, config['encoding'])
                if not data:
                    print("No detailed.out found for mol_id: ", file_path, " in file: ", mol_dir)
                    continue
                elif data is None:
                    return None
                molecules[data_group][name]['smiles_attributes'] = np.array(data)
                molecules[data_group][name]['smiles_string'] = np.char.encode(smiles_string, encoding='cp037')
                molecules[data_group][name]['inchi'] = np.array([np.char.encode(inchi, encoding='cp037'),
                                                                    np.char.encode(inchikey, encoding='cp037')])
            except:
                print("Error in smiles.pdb for mol_id: ", file_path, " in file: ", mol_dir)
                return 1
            
        elif file_path.endswith('.gen'):
            # Open "geo_end.gen" and read its content
            data = get_xyz_from_gen_file(data, file_path, config['encoding'])
            if not data:
                print("No geo_end.gen found for mol_id: ", file_path, " in file: ", mol_dir)
                continue
            molecules[data_group][name]['mol'] = np.array(data)
    if queue is None:
        for group_key in molecules.keys():
            for mol_key in molecules[group_key].keys():
                molecules[group_key][mol_key] = build_molecule_array(molecules[group_key][mol_key], config)
        return molecules
    if no_data:
        print("No data found for mol_id[1]: ", file_path, " in file: ", mol_dir)
        queue.put(None)
    for group_key in molecules.keys():
        for mol_key in molecules[group_key].keys():
            molecules[group_key][mol_key] = build_molecule_array(molecules[group_key][mol_key], config)
    queue.put(molecules)  # Return the data for this file


def write_to_h5py(mol, key, f):
    grp = f.create_group(key)
    grp.create_dataset('title', data=np.char.encode(mol['title'], encoding='cp037'))
    grp.create_dataset('mol', data=mol['mol'])
    grp.create_dataset('y', data=mol['y'])
    grp.create_dataset('y_bar', data=mol['y_bar'])
    grp.create_dataset('features', data=mol['features'])
    grp.create_dataset('center_of_mass', data=mol['center_of_mass'])
    grp.create_dataset('smiles_string', data=mol['smiles_string'])
    grp.create_dataset('inchi', data=mol['inchi'])
    return 0

# Define the manager process function
def write_data_to_h5py_file(h5py_writers, labels, total_written_molecules, queue, lock):
    go_sleep = False
    #total_written_molecules = 0  # Track the total number of written molecules
    while True:
        with lock:
            if not queue.empty():
                
                modes = []
                for h5py_file in h5py_files:
                    if os.path.exists(h5py_file):
                        modes.append('a')
                    else:
                        print("File ", h5py_file, " does not exist, creating file")
                        modes.append('w')
                h5py_writers = {label:h5py.File(h5py_files[ind], modes[ind])
                                    for ind, label in enumerate(labels)}
                # Get data from the queue
                mol = queue.get()
                if mol is None:
                    continue
                elif mol == 'DONE':
                    break
                for group_key in mol.keys():
                    for mol_key in mol[group_key].keys():
                        write_to_h5py(mol[group_key][mol_key],
                                    mol_key, h5py_writers[group_key])
                        total_written_molecules += 1  # Increment the total number of written molecules
                        if total_written_molecules % 1000 == 0:
                            print("Total number of written molecules so far:", total_written_molecules)
                    h5py_writers[group_key].close()
            else:
                go_sleep = True
        if go_sleep: # Doing this outside the lock
            print("Sleeping for 1 seconds")
            time.sleep(1)
            go_sleep = False
    queue.put(total_written_molecules)

def construct_h5py_data(source_directory, h5py_files, dataset_sizes, config, single_core):
    # Count the number of files in the source directory
    mol_names = [name for name in os.listdir(source_directory) if name.startswith('mol_')]
    num_mols = len(mol_names)
    num_mol_in_dataset = sum(dataset_sizes)
    labels = ['train', 'test', 'val']
    random.shuffle(mol_names)
    # Partition the molecules into train, test, and validation sets
    num_train = [labels[0]]*dataset_sizes[0]
    num_test = [labels[1]]*dataset_sizes[1]
    num_val = [labels[2]]*dataset_sizes[2]
    
    train_mol_names = mol_names[:dataset_sizes[0]]
    test_mol_names = mol_names[dataset_sizes[0]:dataset_sizes[0]+dataset_sizes[1]]
    val_mol_names = mol_names[dataset_sizes[0]+dataset_sizes[1]:]
    
    label_lst = num_train + num_test + num_val

    mol_ids = list(zip(mol_names, label_lst))
    
    if single_core:
        for mol_id in mol_ids:
            result = construct_h5py_mol(mol_id, source_directory, config, None, None)
            #if result != 1:
                #print(result.keys())
        # Write results to file TO DO...
        
    else:
        # Get the number of cores to use (total cores - 4)
        num_cores = max(multiprocessing.cpu_count() - 4, 2)
        
        t1 = time.time()

        mol_ids_divided = [mol_ids]
        total_written_molecules = 0
        for mol_ids in mol_ids_divided:
            manager = multiprocessing.Manager()
            queue = manager.Queue()
            lock = manager.Lock()
            pool = multiprocessing.Pool(processes=num_cores)
            writer_process = multiprocessing.Process(target=write_data_to_h5py_file, 
                                                        args=(h5py_files, labels, total_written_molecules, queue, lock))
            writer_process.start()
            for mol_id in mol_ids:
                pool.apply_async(construct_h5py_mol, args=(mol_id, source_directory, config, queue, lock))
            
            pool.close()
            pool.join()
            queue.put('DONE')
            t2 = time.time()
            print("Constructing h5py files took ", np.round(t2-t1, 2), " seconds")
            writer_process.join()
            total_written_molecules = queue.get()
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Molecular Mass Spectra Prediction')
    parser.add_argument('--train_data', type=str, default='./data/data_train_small.hdf5',
                        help='path to training data for traning (hdf5)')
    parser.add_argument('--test_data', type=str, default='./data/data_test_small.hdf5',
                        help='path to test data for traning (hdf5)')
    parser.add_argument('--val_data', type=str, default='./data/data_validation_small.hdf5',
                        help='path to validation data (hdf5)')    
    parser.add_argument('--dataset_pre_size', type=int, default=9000,
                        help=' Size of pretraining dataset')
    parser.add_argument('--dataset_size', type=int, default=480,
                    help=' Size of training dataset')
    parser.add_argument('--dataset_val_size', type=int, default=140,
                help='Size of validation dataset')
    parser.add_argument('--data_config_path', type=str, default='./config/preprocess_uv-vis.yml',
                        help='path to configuration')
    parser.add_argument('--path_zip', type=str, default='./datasets/10.13139_OLCF_1890227/dftb_gdb9_electronic_excitation_spectrum', 
                        help='path to zip files')
    # single_core = False if do_parallel flag is set
    parser.add_argument('--do_parallel', action='store_true', default=False,
                        help='do parallel processing')

    args = parser.parse_args()

    with open(args.data_config_path, 'r') as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    source_directory = args.path_zip
    h5py_files = [args.train_data, args.test_data, args.val_data]
    single_core = not args.do_parallel
    dataset_sizes = [args.dataset_pre_size, args.dataset_size, args.dataset_val_size]
    # If files in h5py_files already exist, remove them
    for h5py_file in h5py_files:
        if os.path.exists(h5py_file):
            os.remove(h5py_file)
    construct_h5py_data(source_directory, h5py_files, dataset_sizes, config, single_core)