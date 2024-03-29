import os
import argparse
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


from molmspack.molnet import MolNet_Oth
from molmspack.dataset import MolORNL_Dataset

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def train_step(model, device, loader, optimizer, batch_size, num_points): 
	accuracy = 0

	with tqdm(total=len(loader)) as bar:
		for step, batch in enumerate(loader):
			x, y = batch
			x = x.to(device=device, dtype=torch.float)
			x = x.permute(0, 2, 1)
			y = y.to(device=device, dtype=torch.float)
			idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

			optimizer.zero_grad()
			model.train()
			pred = model(x, None, idx_base)
			loss = torch.sqrt(nn.MSELoss()(pred, y))
			#pred = nn.ReLU()(pred / torch.max(pred))
			#loss = torch.mean(1-nn.CosineSimilarity(dim=1)(pred, y_bar))
			loss.backward()
			bar.set_description('Train')
			bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
			bar.update(1)
			optimizer.step()
			accuracy += torch.sqrt(torch.pow(pred - y, 2).mean())
			#accuracy += torch.mean(1-nn.CosineSimilarity(dim=1)(pred, y_bar))
	return accuracy / (step + 1)

def eval_step(model, device, loader, batch_size, num_points): 
	model.eval()
	accuracy = 0
	with tqdm(total=len(loader)) as bar:
		for step, batch in enumerate(loader):
			# title, x, y, y_bar, center_of_mass, features
			x, y = batch
			x = x.to(device=device, dtype=torch.float)
			x = x.permute(0, 2, 1)
			#y = y.to(device=device, dtype=torch.float)
			#y = y.view(-1, 1)
			y = y.to(device=device, dtype=torch.float)
			idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

			with torch.no_grad(): 
				pred = model(x, None, idx_base)

			bar.set_description('Eval')
			bar.update(1)
			accuracy += torch.sqrt(torch.pow(pred - y, 2).mean())
			#accuracy += torch.mean(1-nn.CosineSimilarity(dim=1)(pred, y_bar))
	return accuracy / (step + 1)

def init_random_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	return



if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Molecular Mass Spectra Prediction (Pre-train)')
	parser.add_argument('--train_data', type=str, default=["./data/data_pre_train.hdf5", "./data/data_train_small.hdf5"],
						help='path to training data (pkl)')
	parser.add_argument('--test_data', type=str, default=["./data/data_test_small.hdf5", "./data/data_pre_test.hdf5"],
						help='path to test data (pkl)')
	parser.add_argument('--model_config_path', type=str, default='./config/molnet_pre.yml',
						help='path to model and training configuration')
	parser.add_argument('--data_config_path', type=str, default='./config/preprocess_uv-vis.yml',
						help='path to configuration')
	parser.add_argument('--checkpoint_path', type=str, default = './check_point/molnet_pre_uv-vis_all_features.pt',
						help='Path to save checkpoint')
	parser.add_argument('--resume_path', type=str, default='', 
						help='Path to pretrained model')
	parser.add_argument('--ex_model_path', type=str, default='',
						help='Path to export the whole model (structure & weights)')
	parser.add_argument('--true_output', type=str, default='features',
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

	# 1. Data
	train_set = MolORNL_Dataset(args.train_data, args.true_output, 
                             data_augmentation=False, partitioned=False)
	train_loader = DataLoader(
					train_set,
					batch_size=config['train']['batch_size'], 
					shuffle=True,
					pin_memory=True,
					num_workers=config['train']['num_workers'], 
					drop_last=True)
	valid_set = MolORNL_Dataset(args.test_data, args.true_output, 
                             data_augmentation=False, partitioned=False)
	valid_loader = DataLoader(
					valid_set,
					batch_size=config['train']['batch_size'], 
					shuffle=True,
					pin_memory=True,
					num_workers=config['train']['num_workers'], 
					drop_last=True)

	# 2. Model
	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() and not args.no_cuda else torch.device("cpu")
	print(f'Device: {device}', '|\t GPU is available:\t',torch.cuda.is_available(), '|\tCurrent torch cuda version:\t', torch.version.cuda)

	model = MolNet_Oth(config['model']).to(device)
	num_params = sum(p.numel() for p in model.parameters())
	print(f'{str(model)} #Params: {num_params}')

	# 3. Train
	optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'])
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
	if args.resume_path != '':
		print("Load the checkpoints...")
		model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])
		optimizer.load_state_dict(torch.load(args.resume_path, map_location=device)['optimizer_state_dict'])
		scheduler.load_state_dict(torch.load(args.resume_path, map_location=device)['scheduler_state_dict'])
		best_valid_mae = torch.load(args.resume_path)['best_val_mae']

	if args.checkpoint_path != '':
		checkpoint_dir = "/".join(args.checkpoint_path.split('/')[:-1])
		os.makedirs(checkpoint_dir, exist_ok = True)
	best_valid_mae = 999999
	early_stop_step = 10
	early_stop_patience = 0
	for epoch in range(1, config['train']['epochs'] + 1):
		reduce_lr = True
		print("\n=====Epoch {}".format(epoch))
		train_mae = train_step(model, device, train_loader, optimizer, 
								batch_size=config['train']['batch_size'], 
								num_points=config['model']['max_atom_num'])
		valid_mae = eval_step(model, device, valid_loader, 
								batch_size=config['train']['batch_size'], 
								num_points=config['model']['max_atom_num'])
		print("Train: MAE: {}, \nValidation: MAE: {}".format(train_mae, valid_mae))

		if valid_mae < best_valid_mae: 
			best_valid_mae = valid_mae
			if args.checkpoint_path != '':
				print('Saving checkpoint...')
				checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae, 'num_params': num_params}
				torch.save(checkpoint, args.checkpoint_path)

			early_stop_patience = 0
			print('Early stop patience reset')
		else:
			early_stop_patience += 1
			print('Early stop count: {}/{}'.format(early_stop_patience, early_stop_step))

		scheduler.step(valid_mae) # ReduceLROnPlateau
		print(f'Best MAE so far: {best_valid_mae}')

		if early_stop_patience == early_stop_step: 
			print('Early stop!')
			break

		if args.ex_model_path != '': # export the model
			print('Export the model...')
			model_scripted = torch.jit.script(model) # Export to TorchScript
			model_scripted.save(args.ex_model_path) # Save

