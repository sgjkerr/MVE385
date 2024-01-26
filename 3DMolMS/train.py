import os
import argparse
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from molmspack.molnet import MolNet_MS
from molmspack.dataset import MolORNL_Dataset

w = 10.0

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def gauss_torch(a, m, x, w, log_2):
    # calculation of the Gaussian line shape
    # a = amplitude (max y, intensity)
    # x = position
    # m = maximum/median (stick position in x, wave number)
    # w = line width, FWHM
	e = torch.exp(-(log_2 * ((m-x) / w) ** 2))
	return torch.einsum('i,ij->ij', a, e)

def train_step(model, device, loader, optimizer, alpha, batch_size, num_points): 
	accuracy = 0
	bins = int((xmax_spectrum - xmin_spectrum)/spectrum_discretization_step)
	with tqdm(total=len(loader)) as bar:
		x_spectra = torch.arange(xmin_spectrum, xmax_spectrum, spectrum_discretization_step, device=device)
		x_spectra_tensor = x_spectra.unsqueeze(0)
		x_spectra_tensor = x_spectra_tensor.repeat(batch_size, 1)
		log_2 = torch.log(torch.tensor(2.0, device=device))
		for step, batch in enumerate(loader):
			x, y = batch
			x = x.to(device=device, dtype=torch.float)
			x = x.permute(0, 2, 1)
			y = y.to(device=device, dtype=torch.float)
			y = y / torch.max(y, dim=1, keepdim=True)[0]
			y = torch.sqrt(y)

			#env = env.to(device=device, dtype=torch.float)
			idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

			optimizer.zero_grad()
			model.train()
			pred = model(x, None, idx_base)
			pred = nn.LeakyReLU(0.1)(pred)

			gauss_sum = torch.zeros((batch_size, bins), dtype=torch.float, device=device)  
			for index, wn in enumerate(x_spectra):
				gauss_sum += gauss_torch(pred[:,index], x_spectra_tensor, wn, w, log_2)
			# Normalize each mini-batch with max function
			gauss_sum = gauss_sum / torch.max(gauss_sum, dim=1, keepdim=True)[0]
			loss = nn.MSELoss()(gauss_sum, y)
			loss.backward()
			bar.set_description('Train')
			bar.update(1)
			bar.set_postfix(loss=loss.item(), lr=get_lr(optimizer))

			optimizer.step()
			gauss_sum = torch.pow(gauss_sum, 2)
			y = torch.pow(y, 2)

			accuracy += torch.abs(gauss_sum - y).mean().item()
	return accuracy / (step + 1)

def eval_step(model, device, loader, batch_size, num_points): 
	model.eval()
	accuracy = 0
	bins = int((xmax_spectrum - xmin_spectrum)/spectrum_discretization_step)
	x_spectra = torch.arange(xmin_spectrum, xmax_spectrum, spectrum_discretization_step, device=device)
	x_spectra_tensor = x_spectra.unsqueeze(0)  # Add an extra dimension at the beginning
	x_spectra_tensor = x_spectra_tensor.repeat(batch_size, 1)
	log_2 = torch.log(torch.tensor(2.0, device=device))
	with tqdm(total=len(loader)) as bar:
		for step, batch in enumerate(loader):
			x, y = batch
			x = x.to(device=device, dtype=torch.float)
			x = x.permute(0, 2, 1)
			y = y.to(device=device, dtype=torch.float)
			y = y / torch.max(y, dim=1, keepdim=True)[0]
			#env = torch.arange().to(device=device, dtype=torch.float)
			idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

			with torch.no_grad(): 
				pred = model(x, None, idx_base)
			pred = nn.LeakyReLU(0.1)(pred)

			gauss_sum = torch.zeros((batch_size, bins), dtype=torch.float, device=device)  
			for index, wn in enumerate(x_spectra):
				gauss_sum += gauss_torch(pred[:,index], x_spectra_tensor, wn, w, log_2)
			gauss_sum = gauss_sum / torch.max(gauss_sum, dim=1, keepdim=True)[0]

			bar.set_description('Eval')
			bar.update(1)
			gauss_sum = torch.pow(gauss_sum, 2)
			accuracy += torch.abs(gauss_sum - y).mean().item()
	return accuracy / (step + 1)

def init_random_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	return



if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Molecular Mass Spectra Prediction (Train)')
	parser.add_argument('--train_data', type=str, default=["./data/data_train.hdf5", "./data/data_train_small.hdf5"],
						help='path to training data (hdf5)')
	parser.add_argument('--test_data', type=str, default=["./data/data_test_small.hdf5", "./data/data_test.hdf5"],
						help='path to test data (hdf5)')
	parser.add_argument('--model_config_path', type=str, default='./config/molnet.yml',
						help='path to model and training configuration')
	parser.add_argument('--data_config_path', type=str, default='./config/preprocess_uv-vis.yml',
						help='path to configuration')
	parser.add_argument('--checkpoint_path', type=str, default = './check_point/molnet_uv-vis_from_start_max_norm_transfer.pt',
						help='Path to save checkpoint')
	parser.add_argument('--resume_path', type=str, default='', 
						help='Path to pretrained model')
	parser.add_argument('--transfer', action='store_true', 
						help='Whether to load the pretrained encoder')
	parser.add_argument('--ex_model_path', type=str, default='',
						help='Path to export the whole model (structure & weights)')
	parser.add_argument('--true_output', type=str, default='y',
						help='Output vector to train against')
	parser.add_argument('--seed', type=int, default=42,
						help='Seed for random functions')
	parser.add_argument('--device', type=int, default=0,
						help='Which gpu to use if any')
	parser.add_argument('--no_cuda', type=bool, default=False,
						help='Enables CUDA training')
	args = parser.parse_args()



	init_random_seed(args.seed)
	with open(args.model_config_path, 'r') as f: 
		config = yaml.load(f, Loader=yaml.FullLoader)
	print('Load the model & training configuration from {}'.format(args.model_config_path))
	spectrum_discretization_step = config['model']['resolution']
	xmin_spectrum = config['model']['min_wavelength']
	xmax_spectrum = config['model']['max_wavelength'] + spectrum_discretization_step
		
	train_set = MolORNL_Dataset(args.train_data, args.true_output, 
                             data_augmentation=True, partitioned=False)
	train_loader = DataLoader(
					train_set,
					batch_size=config['train']['batch_size'], 
					shuffle=True,
					pin_memory=True,
					num_workers=config['train']['num_workers'], 
					drop_last=True)
	valid_set = MolORNL_Dataset(args.test_data, args.true_output, 
                             data_augmentation=True, partitioned=False)
	valid_loader = DataLoader(
					valid_set,
					batch_size=config['train']['batch_size'], 
					shuffle=True, 
					num_workers=config['train']['num_workers'], 
					drop_last=True)

	# 2. Model
	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
	print(f'Device: {device}')

	model = MolNet_MS(config['model']).to(device)
	num_params = sum(p.numel() for p in model.parameters())
	print(f'{str(model)} #Params: {num_params}')

	# 3. Train
	optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'])
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
	if args.transfer and args.resume_path != '': 
		print("Load the pretrained encoder (freeze the encoder)...")
		state_dict = torch.load(args.resume_path, map_location=device)['model_state_dict']
		encoder_dict = {}
		for name, param in state_dict.items(): 
			if not name.startswith("decoder"): 
				param.requires_grad = False # freeze the encoder
				encoder_dict[name] = param
		model.load_state_dict(encoder_dict, strict=False)
	elif args.resume_path != '':
		print("Load the checkpoints...")
		model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])
		optimizer.load_state_dict(torch.load(args.resume_path, map_location=device)['optimizer_state_dict'])
		scheduler.load_state_dict(torch.load(args.resume_path, map_location=device)['scheduler_state_dict'])
		best_valid_acc = torch.load(args.resume_path)['best_val_acc']

	if args.checkpoint_path != '':
		checkpoint_dir = "/".join(args.checkpoint_path.split('/')[:-1])
		os.makedirs(checkpoint_dir, exist_ok = True)

	best_valid_acc = 999999
	early_stop_step = 10
	early_stop_patience = 0

	for epoch in range(1, config['train']['epochs'] + 1):
		reduce_lr = True
		alpha = (config['train']['epochs']-epoch+1)/config['train']['epochs']
		print("\n=====Epoch {}".format(epoch))
		train_acc = train_step(model, device, train_loader, optimizer,alpha, 
                         batch_size=config['train']['batch_size'], num_points=config['model']['max_atom_num'])
		valid_acc = eval_step(model, device, valid_loader, 
								batch_size=config['train']['batch_size'], num_points=config['model']['max_atom_num'])
		print("Train: Acc: {}, \nValidation: Acc: {}".format(train_acc, valid_acc))
		save = True
		if valid_acc < best_valid_acc:
			reduce_lr = False
			best_valid_acc = valid_acc

			if args.checkpoint_path != '':
				print('Saving checkpoint...')
				checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_acc': best_valid_acc, 'num_params': num_params}
				torch.save(checkpoint, args.checkpoint_path)

			early_stop_patience = 0
			print('Early stop patience reset')
		else:
			early_stop_patience += 1
			print('Early stop count: {}/{}'.format(early_stop_patience, early_stop_step))

		# scheduler.step()
		if reduce_lr:
			scheduler.step(valid_acc)
		#scheduler.step(valid_acc) # ReduceLROnPlateau
		print(f'Best cosine similarity so far: {best_valid_acc}')

		if early_stop_patience == early_stop_step:
			print('Early stop!')
			break

	if args.ex_model_path != '': # export the model
		print('Export the model...')
		model_scripted = torch.jit.script(model) # Export to TorchScript
		model_scripted.save(args.ex_model_path) # Save
