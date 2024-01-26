#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_featurizer, GraphModel


# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--sheetcsv-dir", help="sheetcsv dir", default="../smalldatacsv/ornl_aisd_ex/ornl_aisd_ex_1.csv"
)
parser.add_argument("--model-name", help="graph model name", required=True)
parser.add_argument(
    "--model-dir", help="dir to save model", default="../models"
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

sheetcsv_dir = args.sheetcsv_dir

model_name = args.model_name
model_dir = args.model_dir.rstrip("/") + f"/{model_name}"
param_dir = args.param_dir
output_dir = args.output_dir.rstrip("/")

# read params
with open(param_dir) as param_file:
    params = json.load(param_file)[model_name]

# some setttings
shortset = 1


# read data sheet

csv_data = pd.read_csv(sheetcsv_dir)

# extract feature
featurizer = get_featurizer(model_name)

feature = []

Dataset_dir = "/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/smalldataset/ornl_aisd_ex_1"

with open('/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/dev_test.pkl', 'rb') as f:
    loaded_array = pickle.load(f)


shortset =0


absorb = []
if shortset == 1:
    for folder in sorted(os.listdir(Dataset_dir)):
        local_dir = os.path.join(Dataset_dir, folder, "EXC-smooth.DAT")
        df = pd.read_csv(local_dir, delim_whitespace=True)
        row = np.squeeze(df.values[0:len(df.values),[1]])
        absorb.append(row.transpose())
else:
    for i in tqdm(range(0,1044)):#tqdm(range(0,loaded_array.shape[0])):
        absorb_row = np.array(loaded_array[i][1:-1], dtype = float)
        absorb.append(absorb_row)



if shortset ==1:
    for smile in tqdm(csv_data.values[0:100,[1]]): 

        feat = featurizer.featurize(smile[0])
        feature.append(feat)
else:
    for i in tqdm(range(0,1044)):#tqdm(range(0,loaded_array.shape[0])): 
        encoded_smiles_str = loaded_array[i][0]
        decoded_smiles_str = np.char.decode(encoded_smiles_str, encoding='cp037')
        smile = str(decoded_smiles_str)


        feat = featurizer.featurize(smile)[0]
        feature.append(feat)
## read smiles as x and spectra as y

f1 = [feature[0]]

# split train test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    feature, absorb, random_state=42, test_size=0.2
)

# full permutation of params
keys, values = zip(*params.items())
permutations = [
    dict(zip(keys, v)) for v in itertools.product(*values)
]
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


for perm in permutations:
    performance = []
    cosine_performance = []
    for _, (train_idx, val_idx) in enumerate(kfold.split(X_train_val)):
        X_train, X_val, y_train, y_val = [], [], [], []
        for idx in train_idx:
            X_train.append(X_train_val[idx])
            y_train.append(y_train_val[idx])
        for idx in val_idx:
            X_val.append(X_train_val[idx])
            y_val.append(y_train_val[idx])
        
        # model training
        model = GraphModel(name=model_name, model_dir=model_dir, **perm)
        X_train=X_train
        y_train=y_train
        
        model.fit(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            epoch=1000, patience=30, interval=1, validation=True,
            metric=nn.CosineEmbeddingLoss(), store_best=True, ## use cosine as metric
            greater_is_better=False, verbose=0
        ) #metric=mean_squared_error #greater_is_better=False, #metric=cosine_similarity

        # prediction
        y_val_pred = model.predict(X_val)
        target = torch.ones(len(y_val_pred))

        cosines = nn.CosineEmbeddingLoss()(torch.tensor(y_val), torch.tensor(y_val_pred), target)
        cosine_performance.append(cosines)

    mean_cosine = np.round(np.float64(cosine_performance).mean(),4)
    std_cosine = np.round(np.float64(cosine_performance).std(),4)
    msg = (
        f"model: {model_name}\n"
        f"setting: {perm}\n"
        f"Cosine Similarity: {mean_cosine:.4f} +/- {std_cosine:.4f}\n"
    )
    print(msg)
    # write to file
    with open(f"{output_dir}/{model_name}.txt", "a") as file:
        file.write(msg)

p1 = model.predict(f1)


x = np.arange(80, 420,2)
plt.plot(x, np.squeeze(model.predict(f1)))
plt.show()