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


# extract feature
featurizer = get_featurizer(model_name)



with open('/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/Dataset_to_run/AttentiveFP/data/train/train.pkl', 'rb') as ftrain:
    loaded_array_train = pickle.load(ftrain)

with open('/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/Dataset_to_run/AttentiveFP/data/validation/validation.pkl', 'rb') as fval:
    loaded_array_val = pickle.load(fval)

with open('/home/kerrj/Documents/UV_vis_proj/attentivefp/test_env/Dataset_to_run/AttentiveFP/data/test/test.pkl', 'rb') as ftest:
    loaded_array_test = pickle.load(ftest)


absorb_train = []
feature_train = []
absorb_val = []
feature_val = []
feature_test = []
absorb_test = []

for i in tqdm(range(0,loaded_array_train.shape[0])):
    absorb_row = np.array(loaded_array_train[i][1:-1], dtype = float)
    absorb_train.append(absorb_row)


for i in tqdm(range(0,loaded_array_train.shape[0])): 
    encoded_smiles_str = loaded_array_train[i][0]
    decoded_smiles_str = np.char.decode(encoded_smiles_str, encoding='cp037')
    smile = str(decoded_smiles_str)


    feat = featurizer.featurize(smile)[0]
    feature_train.append(feat)

for i in tqdm(range(0,loaded_array_val.shape[0])):
    absorb_row = np.array(loaded_array_val[i][1:-1], dtype = float)
    absorb_val.append(absorb_row)


for i in tqdm(range(0,loaded_array_val.shape[0])): 
    encoded_smiles_str = loaded_array_val[i][0]
    decoded_smiles_str = np.char.decode(encoded_smiles_str, encoding='cp037')
    smile = str(decoded_smiles_str)


    feat = featurizer.featurize(smile)[0]
    feature_val.append(feat)

for i in tqdm(range(0,loaded_array_test.shape[0])):
    absorb_row = np.array(loaded_array_test[i][1:-1], dtype = float)
    absorb_test.append(absorb_row)


for i in tqdm(range(0,loaded_array_test.shape[0])): 
    encoded_smiles_str = loaded_array_test[i][0]
    decoded_smiles_str = np.char.decode(encoded_smiles_str, encoding='cp037')
    smile = str(decoded_smiles_str)


    feat = featurizer.featurize(smile)[0]
    feature_test.append(feat)




f1 = [feature_train[0]]

# split train test
X_train = feature_train
y_train = absorb_train
X_val = feature_val
y_val = absorb_val
X_test = feature_test
y_test = absorb_test


# full permutation of params
keys, values = zip(*params.items())
permutations = [
    dict(zip(keys, v)) for v in itertools.product(*values)
]

for perm in permutations:
        
    # model training
    model = GraphModel(name=model_name, model_dir=model_dir, **perm)
    X_train=X_train
    y_train=y_train
        
    model.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        epoch=1000, patience=30, interval=1, validation=True,
        metric=nn.CosineEmbeddingLoss(), store_best=True, ## use cosine as metric
        greater_is_better=False, verbose=0
    ) 

    # prediction
    y_test_pred = model.predict(X_test)
    target = torch.ones(len(y_test_pred))

    cosines = nn.CosineEmbeddingLoss()(torch.tensor(y_test), torch.tensor(y_test_pred), target)
    msg = (
        f"model: {model_name}\n"
        f"setting: {perm}\n"
        f"Cosine Similarity: {cosines:.4f}\n"
    )
    print(msg)
    # write to file
    with open(f"{output_dir}/{model_name}.txt", "a") as file:
        file.write(msg)



p1 = model.predict(f1)

x = np.arange(80, 420,2)
plt.plot(x, np.squeeze(model.predict(f1)))
plt.show()