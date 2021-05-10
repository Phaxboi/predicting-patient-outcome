#this file contains the methods used to run the regression model to predict in hopsital mortality

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from metrics import *
from prepare_features_mortality_pred import scale_values

import re
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
    args = parser.parse_args()

    (data_X, data_Y) = scale_values(args.subjects_root_path)
    data_X_test = data_X[:150]
    data_X = data_X[150:]
    data_Y_test = data_Y[:150]
    data_Y = data_Y[150:]
    logreg = LogisticRegression(penalty='l2', C=0.001, random_state=42)
    logreg.fit(data_X, data_Y)
    predictions=logreg.predict_proba(data_X)
    print(sum(data_Y))
    print(predictions)
    print(logreg.score(data_X_test, data_Y_test))


    #NOTE:rest is compied from becnhmark program
    result_dir = os.path.join(args.subjects_root_path, 'results')
    try:
        os.makedirs(result_dir)
    except:
        pass

    file_name = 'result_scuffed_model'

    with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
            ret = print_metrics_binary(data_Y, logreg.predict_proba(data_X))
            ret = {k : float(v) for k, v in ret.items()}
            json.dump(ret, res_file)


    prediction = logreg.predict_proba(data_X_test)[:, 1]

    with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(data_Y_test, prediction)
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    #save_results(test_names, prediction, test_y, os.path.join(args.output_dir, 'predictions', file_name + '.csv'))



main()

