import deepchem as dc
import argparse
import pandas as pd
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

feature_val = []
absorb_val = []
smiles = []
predicted_Data = []
data = []


with open('/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/Dataset_to_run/AttentiveFP/data/validation/validation.pkl', 'rb') as fval:
    loaded_array_val = pickle.load(fval)

for i in tqdm(range(0,loaded_array_val.shape[0])):
    absorb_row = np.array(loaded_array_val[i][1:-1], dtype = float)
    absorb_val.append(absorb_row)

    encoded_smiles_str = loaded_array_val[i][0]
    decoded_smiles_str = np.char.decode(encoded_smiles_str, encoding='cp037')
    smile = str(decoded_smiles_str)
    print(type(smile))
    smiles.append(smile)

    feat = featurizer.featurize(smile)[0]
    feature_val.append(feat)
    data_pred = model.predict([feature_val[i]])
    predicted_Data.append(data_pred)
    data_entry = np.append(smiles[i], predicted_Data[i])
    data.append(data_entry)

data_array = np.array(data)
df = pd.DataFrame(data_array)
if MSE == 0:
    df.to_csv('data_attentiveFP_val_cosine.csv', index=False)
else:
    df.to_csv('data_attentiveFP_val_MSE.csv', index=False)

