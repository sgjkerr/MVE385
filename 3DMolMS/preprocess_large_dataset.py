import os
import tarfile
import argparse
import yaml
import multiprocessing
import numpy as np
import pandas as pd
import h5py
import time
import random
from collections import defaultdict

from utils_large_set import get_xyz_from_gen_file, get_mol_attributes_file, get_mol_spectra, get_gross_charge, read_smiles_from_csv_w_pandas

    
def build_molecule_array(mol, config):
    mol['mol'] = np.hstack((mol['mol'], mol['smiles_attributes']))
    mol['mol'] = np.pad(mol['mol'], ((0, config['encoding']['max_atom_num']-mol['mol'].shape[0]), (0, 0)), constant_values=0)

    return mol

def find_members_to_process(tar_members, mol_indecies, data_group, missing_mols):
    lo = int(tar_members[0].name.split('_')[-1])
    hi = int(tar_members[-6].name.split('_')[-1])
    members_to_precess = []
    for ind, mol_index in enumerate(mol_indecies):
        index = (mol_index - lo)*6 # 5+1 for number of files per molecule in ORNL dataset (5 files, 1 directory)
        if mol_index < hi:
            curr_member_index = int(tar_members[index].name.split('_')[-1])
            fine_shift = (curr_member_index - mol_index)*6
            index -= fine_shift
            curr_member_index = int(tar_members[index].name.split('_')[-1])

        else:
            index = len(tar_members)-6
            curr_member_index = int(tar_members[index].name.split('_')[-1])

        if mol_index != curr_member_index:
            if mol_index in missing_mols['mols']:
                print("Missing molecule ", "mol_"+str(mol_index).zfill(6), " sampled")
            else:
                print("Molecule ", "mol_"+str(mol_index).zfill(6), " sampled and unexpectingly missing")
            continue            
        for step in range(1, 6):
            members_to_precess.append((tar_members[index+step], data_group[ind]))
    return members_to_precess       

def construct_h5py_mol(mol_id, source_file, config, queue, lock):
    file = os.path.join(source_file, f'ornl_aisd_ex_{mol_id[0]}.tar.gz')
    csv_file = os.path.join(source_file, f'ornl_aisd_ex_csv')
    csv_file = os.path.join(csv_file, f'ornl_aisd_ex_{mol_id[0]}.csv')
    df_dataset = pd.read_csv(csv_file)
    with tarfile.open(file, 'r:gz') as tar:
        tar_members = tar.getmembers()
        members_to_precess = find_members_to_process(tar_members, mol_id[1], mol_id[2], config['missing_mols'])
        molecules = {}
        no_data = True
        for obj in members_to_precess:
            member = obj[0]
            data_group = obj[1]
            if data_group not in molecules:
                molecules[data_group] = {}
            if member.name.startswith('mol_'):
                name = member.name.split('/')[0]
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
                if member.name.endswith('detailed.out'):
                    data = get_gross_charge(data, member, tar, config['encoding'])
                    if not data:
                        print("No detailed.out found for mol_id: ", member, " in file: ", file)
                        continue
                    molecules[data_group][name]['features'] = np.array(data)
                elif member.name.endswith('.DAT'):
                    # Open "EXC.DAT" and read its content
                    data, data_bar, center_of_mass = get_mol_spectra(data, member, tar, config['encoding'])
                    if not data:
                        print("No EXC.DAT found for mol_id: ", member, " in file: ", file)
                        continue
                    data = np.array(data)
                    molecules[data_group][name]['y'] = np.sqrt(data)
                    molecules[data_group][name]['y_bar'] = np.sqrt(np.array(data_bar))
                    molecules[data_group][name]['center_of_mass'] = center_of_mass

                elif member.name.endswith('.pdb'):
                    # Open "detailed.out" and read its content
                    data = get_mol_attributes_file(data, member, tar, config['encoding'])
                    smiles_string, inchi, inchikey = read_smiles_from_csv_w_pandas(name, df_dataset)
                    if not data:
                        print("Error when reading smiles.pdb ", member, " in file: ", file)
                        continue
                    elif smiles_string is None or inchi is None or inchikey is None:
                        print(smiles_string)
                        print(inchi) 
                        print(inchikey)
                        continue
                    molecules[data_group][name]['smiles_attributes'] = np.array(data)
                    molecules[data_group][name]['smiles_string'] = np.char.encode(smiles_string, encoding='cp037')
                    molecules[data_group][name]['inchi'] = np.array([np.char.encode(inchi, encoding='cp037'),
                                                                     np.char.encode(inchikey, encoding='cp037')])
                elif member.name.endswith('.gen'):
                    # Open "geo_end.gen" and read its content
                    data = get_xyz_from_gen_file(data, member, tar, config['encoding'])
                    if not data:
                        print("No geo_end.gen found for mol_id: ", member, " in file: ", file)
                        continue
                    molecules[data_group][name]['mol'] = np.array(data)
        if queue is None:
            return molecules
        if no_data:
            print("No data found for mol_id[1]: ", member, " in file: ", file)
            queue.put(None)
        for group_key in molecules.keys():
            for mol_key in molecules[group_key].keys():
                molecules[group_key][mol_key] = build_molecule_array(molecules[group_key][mol_key], config)
    queue.put(molecules)  # Return the data for this file

def combine_molecules_in_same_file_dir(tuples_list):
    # Use defaultdict to accumulate y and z values for each x
    combined_dict = defaultdict(lambda: [[], []])
    for x, y, z in tuples_list:
        combined_dict[x][0].append(y)
        combined_dict[x][1].append(z)
    
    # Convert the dictionary back to a list of tuples
    combined_tuples = [(x, ys, zs) for x, (ys, zs) in combined_dict.items()]
    return combined_tuples

def write_to_h5py(mol, key, f, partition_groups=False):
    if partition_groups:
        num_atoms = np.count_nonzero(mol['mol'][:, 0])
        breaks = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        # Identify the group to which the molecule belongs
        if num_atoms > breaks[-1]:
            ind = len(breaks)
        else:
            for ind, break_point in enumerate(breaks):
                if num_atoms <= break_point:
                    break
        # Create a group for the number of atoms if it does not exist
        if 'num_atoms_'+str(breaks[ind-1]) not in f.keys():
            grp_by_num_atoms = f.create_group('num_atoms_'+str(breaks[ind-1]))
        else:
            grp_by_num_atoms = f['num_atoms_'+str(breaks[ind-1])]
        grp = grp_by_num_atoms.create_group(key)
        grp.create_dataset('title', data=np.char.encode(mol['title'], encoding='cp037'))
        grp.create_dataset('mol', data=mol['mol'])
        grp.create_dataset('y', data=mol['y'])
        grp.create_dataset('y_bar', data=mol['y_bar'])
        grp.create_dataset('features', data=mol['features'])
        grp.create_dataset('center_of_mass', data=mol['center_of_mass'])
        grp.create_dataset('smiles_string', data=mol['smiles_string'])
        grp.create_dataset('inchi', data=mol['inchi'])

    else:
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
def write_data_to_h5py_file(h5py_writers, labels, total_written_molecules, partition_groups, queue, lock):
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
                                    mol_key, h5py_writers[group_key], partition_groups)
                        total_written_molecules += 1  # Increment the total number of written molecules
                        if total_written_molecules % 1000 == 0:
                            print("Total number of written molecules so far:", total_written_molecules)
                    h5py_writers[group_key].close()
            else:
                go_sleep = True
        if go_sleep: # Doing this outside the lock
            #print("Sleeping for 0.2 seconds")
            time.sleep(0.2)
            go_sleep = False
    queue.put(total_written_molecules)

def construct_h5py_data(source_directory, h5py_files, dataset_sizes, config, single_core, partition_groups):
    num_mol_in_dataset = sum(dataset_sizes)
    mol_number = np.random.choice(10502916, num_mol_in_dataset, replace=False)

    print("Number of molecules in dataset: ", num_mol_in_dataset, "\n")

    
    directories_id = np.ceil((mol_number)/10500+0.000001).astype(int)
    # Set directories_id to 1000 if directories_id is >=1001
    directories_id = np.where(directories_id>=1001, 1000, directories_id)
    labels = ['train_pre', 'test_pre', 'train', 'test', 'val']
    label_lst = []
    ind = 0
    for size in dataset_sizes:
        if labels[ind] != 'val':
            label_lst += int(size*0.94)*[labels[ind]]
            label_lst += int(size*0.06)*[labels[ind+1]]
        else:
            label_lst += int(size)*[labels[ind]]
        ind += 2
    mol_ids = list(zip(directories_id, mol_number, label_lst))
    mol_ids.sort(key=lambda x: x[1])
    mol_ids = combine_molecules_in_same_file_dir(mol_ids)
    
    # Single core is currently only used for debugging
    if single_core:
        results = []
        for mol_group in mol_ids:
            results.append(construct_h5py_mol(mol_group, source_directory, config, None, None))
        # Write results to file TO DO...
        
    else:
        # Get the number of cores to use (total cores - 4)
        num_cores = max(multiprocessing.cpu_count() - 4, 2)
        
        t1 = time.time()
        # Divide mol_inds in to multiple chunks with size num_cores
        if num_mol_in_dataset > 500000:
            mol_ids_divided = [mol_ids[i:i + 200] for i in range(0, len(mol_ids), 200)]
        else:
            mol_ids_divided = [mol_ids]
        total_written_molecules = 0
        for mol_ids in mol_ids_divided:
            manager = multiprocessing.Manager()
            queue = manager.Queue()
            lock = manager.Lock()
            pool = multiprocessing.Pool(processes=num_cores)
            writer_process = multiprocessing.Process(target=write_data_to_h5py_file, 
                                                        args=(h5py_files, labels, total_written_molecules, partition_groups, queue, lock))
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
    parser.add_argument('--train_data', type=str, default='./data/data_train.hdf5',
                        help='path to training data for traning (hdf5)')
    parser.add_argument('--test_data', type=str, default='./data/data_test.hdf5',
                        help='path to test data for traning (hdf5)')
    parser.add_argument('--train_pre_data', type=str, default='./data/data_pre_train.hdf5',
                        help='path to training data for pretraining (hdf5)')
    parser.add_argument('--test_pre_data', type=str, default='./data/data_pre_test.hdf5',
                        help='path to test data for pretraining (hdf5)')
    parser.add_argument('--val_data', type=str, default='./data/data_validation.hdf5',
                        help='path to validation data (hdf5)')    
    parser.add_argument('--dataset_pre_size', type=int, default=2000,
                        help=' Size of pretraining dataset')
    parser.add_argument('--dataset_size', type=int, default=2000,
                    help=' Size of training dataset')
    parser.add_argument('--dataset_val_size', type=int, default=100,
                help='Size of validation dataset')
    parser.add_argument('--data_config_path', type=str, default='./config/preprocess_uv-vis.yml',
                        help='path to configuration')
    parser.add_argument('--path_zip', type=str, default='./datasets/10.13139_OLCF_1907919', 
                        help='path to zip files')
    # single_core = False if do_parallel flag is set
    parser.add_argument('--do_parallel', action='store_true', default=False,
                        help='do parallel processing')
    # Partition groups = False if do_partition flag is set
    parser.add_argument('--do_partition', action='store_true', default=False,
                        help='do partitioning of groups')

    args = parser.parse_args()

    with open(args.data_config_path, 'r') as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    source_directory = args.path_zip
    h5py_files = [args.train_pre_data, args.test_pre_data ,
                  args.train_data, args.test_data,
                  args.val_data]
    single_core = not args.do_parallel
    do_partition = args.do_partition
    if do_partition:
        # Add "_partitioned" to the end of the file name
        h5py_files = [file_name[:-5]+'_partitioned.hdf5' for file_name in h5py_files]
    dataset_sizes = [args.dataset_pre_size, args.dataset_size, args.dataset_val_size]
    construct_h5py_data(source_directory, h5py_files, dataset_sizes, config, single_core, do_partition)