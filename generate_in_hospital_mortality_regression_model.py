#this file contains the methods used to run the regression model to predict in hopsital mortality

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import os
import numpy as np
import argparse
import pandas as pd

import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help='inverse of L1 / L2 regularization')
    parser.add_argument('--penalty', type=str, help='Specify if l1 or l2 is used')
    parser.add_argument('--data', type=str, help='Path to the data used to train the model')
    parser.add_argument('--output_dir', type=str, help='Directory where all output files are stored',
                        default='.')
    args = parser.parse_args()
    C = args.c
    penalty = args.penalty
    data = args.data
    output_dir = args.output_dir

    #read all 48h timeseries files as vectors
    stay_ids = []
    data_X = []
    data_Y = []
    mortality_summary = pd.read_csv(os.path.join(data, 'mortality_summary.csv'))
    mortality = 0
    for root, dirs, files in os.walk(data):
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                stay_id = re.search('.*_(\d*)_.*', file_name).group(1)
                values = episode.values.tolist()
                stay_ids += [stay_id]
                data_X = data_X + values
                data_Y += [mortality]
                

    imputer.fit(data_X)




    scaler = StandardScaler()
