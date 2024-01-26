import deepchem as dc
import argparse
import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json
import itertools
import pandas as pd
import numpy as np
import pubchempy as pcp
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_percentage_error, r2_score
)
MSE = 1
if MSE == 0:
    from utils import get_featurizer, GraphModel
else:
    from utils_using_MSE import get_featurizer, GraphModel




# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--sheetcsv-dir", help="sheetcsv dir", default="../smalldatacsv/ornl_aisd_ex/ornl_aisd_ex_1.csv"
)
parser.add_argument("--model-name", help="graph model name", required=True)

if MSE == 0:
    parser.add_argument(
        "--model-dir", help="dir to save model", default="../models"
    )
else:
    parser.add_argument(
        "--model-dir", help="dir to save model", default="../models2"
    )
parser.add_argument(
    "--param-dir", help="dir to hyperparams.json file",
    default="../data/params.json"
)
parser.add_argument(
    "--output-dir", help="dir to write results",
    default="../results"
)
args = parser.parse_args()


model_name = args.model_name
model_dir = args.model_dir.rstrip("/") + f"/{model_name}"
param_dir = args.param_dir
output_dir = args.output_dir.rstrip("/")

featurizer = get_featurizer(model_name)

# read params
with open(param_dir) as param_file:
    params = json.load(param_file)[model_name]


keys, values = zip(*params.items())
permutations = [
    dict(zip(keys, v)) for v in itertools.product(*values)
]

for perm in permutations:
    
  model = GraphModel(name=model_name, model_dir=model_dir, **perm)
  model.model.restore()


dic_list = ["/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/plotset/plot/dictionary_1.pkl", "/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/plotset/plot/dictionary_2.pkl", "/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/plotset/plot/dictionary_3.pkl"]
array_list = ["/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/plotset/plot/data_plot_1.pkl", "/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/plotset/plot/data_plot_2.pkl", "/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/plotset/plot/data_plot_3.pkl"]

if MSE == 0:
    folder_path = "/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/plot_folder/"
else:
    folder_path = "/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/plot_folder_MSE/"

feature_test = []
absorb_test = []
k = 0

no_mol = 10

for j in range(0,len(dic_list)):

    with open (dic_list[j], "rb") as fdic:
        loaded_dic = pickle.load(fdic)



    with open (array_list[j], "rb") as ftest:
        loaded_array_test = pickle.load(ftest)



    for i in tqdm(range(0,no_mol)):
        absorb_row = np.array(loaded_array_test[i][1:-1], dtype = float)
        absorb_test.append(absorb_row)

        encoded_smiles_str = loaded_array_test[i][0]
        decoded_smiles_str = np.char.decode(encoded_smiles_str, encoding='cp037')
        smile = str(decoded_smiles_str)

    
        feat = featurizer.featurize(smile)[0]
        feature_test.append(feat)

        f1 = [feature_test[i]]
        mol = np.array2string(loaded_dic[i][0])
        plot_path = folder_path + mol
        print(plot_path)
        
        plt.figure(mol)
        
        x = np.arange(80, 420,2)
        plt.plot(x, np.squeeze(model.predict(f1)))
        plt.plot(x, absorb_test[i], 'g')
        plt.title(mol)
        plt.xlabel("Wavelength (nm)")
        plt.legend(["Predicted Spectra", "True Spectra"])
        plt.savefig(plot_path, format = "png")
        plt.close()

