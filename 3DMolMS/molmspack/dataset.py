import pickle
import numpy as np
import h5py
from torch.utils.data import Dataset
import random

def read_file(path, true_output, data_augmentation, use_title, use_subsample = False):
		data = []
		with h5py.File(path, 'r') as f:
			size_groups = list(f.keys())
			# Shuffle list
			random.shuffle(size_groups)
			if use_subsample:
				size_groups = size_groups[0:8301]
			for group_name in size_groups:
				data_mol = f[group_name]
				if use_title:
					smiles = data_mol['smiles_string'][...]
					smiles = str(np.char.decode(smiles, encoding='cp037').astype(str))
					data.append({'title':group_name, 'smiles':smiles, 'mol':data_mol['mol'][...], true_output:data_mol[true_output][...]})#, 'y_bar':data_mol['y_bar'][...], 'center_of_mass':data_mol['center_of_mass'][...], 'features':data_mol['features'][...]})
				else:
					data.append({'mol':data_mol['mol'][...], true_output:data_mol[true_output][...]})#, 'y_bar':data_mol['y_bar'][...], 'center_of_mass':data_mol['center_of_mass'][...], 'features':data_mol['features'][...]})
		#print('Load {} data from {}'.format(len(data), path))
		#if data_augmentation: 
		#	flipping_data = []
		#	for d in data:
		#		flipping_mol_arr = np.copy(d['mol'])
		#		flipping_mol_arr[:, 0] *= -1
		#		flipping_data.append({'title': d['title']+'_f', 'mol': flipping_mol_arr, true_output:d[true_output]})#, 'y_bar':d['y_bar'], 'center_of_mass':d['center_of_mass'], 'features':d['features']})
		#	
		#	data = data + flipping_data
		#	print('Load {} data from {} (with data augmentation by flipping coordinates)'.format(len(data), path))
		#else:
		print('Load {} data from {}'.format(len(data), path))
		return data

def read_file_partitioned(path, true_output, data_augmentation):
		data = []
		with h5py.File(path, 'r') as f:
			size_groups = list(f.keys())
			for size_group in size_groups[-3:-1]:
				for group_name in f[size_group].keys():
					data_mol = f[size_group][group_name]
					data.append({'title':group_name, 'mol':data_mol['mol'][...], true_output:data_mol[true_output][...]})#, 'y_bar':data_mol['y_bar'][...], 'center_of_mass':data_mol['center_of_mass'][...], 'features':data_mol['features'][...]})
		print('Load {} data from {}'.format(len(data), path))
		if data_augmentation: 
			flipping_data = []
			for d in data:
				flipping_mol_arr = np.copy(d['mol'])
				flipping_mol_arr[:, 0] *= -1
				flipping_data.append({'title': d['title']+'_f', 'mol': flipping_mol_arr, true_output:d[true_output]})#, 'y_bar':d['y_bar'], 'center_of_mass':d['center_of_mass'], 'features':d['features']})
			
			data = data + flipping_data
			print('Load {} data from {} (with data augmentation by flipping coordinates)'.format(len(data), path))
		else:
			print('Load {} data from {}'.format(len(data), path))
		return data

class MolORNL_Dataset(Dataset):
	def __init__(self, path, true_output, data_augmentation=True, partitioned=True):
		data = []
		if partitioned:
			func = read_file_partitioned
		else:
			func = read_file
		if isinstance(path, list):
			for p in path:
				data += func(p, true_output, data_augmentation, False)
		else:
			data = func(path, true_output, data_augmentation, False)
		self.data = data
		self.length = len(self.data)
		self.true_output = true_output
		self.data_augmentation = data_augmentation
	def __len__(self): 
		return self.length 

	def __getitem__(self, idx):
		if self.data_augmentation:
			# 50/50 radnomly flip the coordinates
			if np.random.rand() > 0.5:
				flipping_mol_arr = np.copy(self.data[idx]['mol'])
				flipping_mol_arr[:, 0] *= -1
				#return self.data[idx]['title'], flipping_mol_arr, self.data[idx][self.true_output]
				return flipping_mol_arr, self.data[idx][self.true_output]

		#return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx][self.true_output]#, self.data[idx]['y_bar'], self.data[idx]['center_of_mass'], self.data[idx]['features']
		return self.data[idx]['mol'], self.data[idx][self.true_output]#, self.data[idx]['y_bar'], self.data[idx]['center_of_mass'], self.data[idx]['features']

class MolORNL_Dataset_For_Ploting(Dataset):
	def __init__(self, path, true_output, data_augmentation=False, partitioned=False):
		data = []
		if partitioned:
			func = read_file_partitioned
		else:
			func = read_file
		if isinstance(path, list):
			for p in path:
				data += func(p, true_output, data_augmentation, True, use_subsample = True)
		else:
			data = func(path, true_output, data_augmentation, True, use_subsample = True)
		self.data = data
		self.length = len(self.data)
		self.true_output = true_output
		self.data_augmentation = data_augmentation
	def __len__(self): 
		return self.length 

	def __getitem__(self, idx):
		if self.data_augmentation:
			# 50/50 radnomly flip the coordinates
			if np.random.rand() > 0.5:
				flipping_mol_arr = np.copy(self.data[idx]['mol'])
				flipping_mol_arr[:, 0] *= -1
				#return self.data[idx]['title'], flipping_mol_arr, self.data[idx][self.true_output]
				return flipping_mol_arr, self.data[idx][self.true_output]

		#return self.data[idx]['title'], self.data[idx]['mol'], self.data[idx][self.true_output]#, self.data[idx]['y_bar'], self.data[idx]['center_of_mass'], self.data[idx]['features']
		return self.data[idx]['title'], self.data[idx]['smiles'] ,self.data[idx]['mol'], self.data[idx][self.true_output]#, self.data[idx]['y_bar'], self.data[idx]['center_of_mass'], self.data[idx]['features']
