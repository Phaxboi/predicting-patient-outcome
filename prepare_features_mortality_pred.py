#used to prepare the features for in hopsital mortality predictions
#features here are currently a vector of the sum of each individual column tracked in the time series
#probably not a good feature but it will do for now, as we're just testing

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from metrics import *

import matplotlib.pyplot as plt

import re
import json


def scale_values(subjects_root_path):
    data_X = []
    data_Y = []
    mortalities = pd.read_csv(os.path.join(subjects_root_path, 'mortality_summary.csv'))
    mortalities.set_index('stay_id',inplace=True, drop=True)

    #1x15 feature vector
    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading values'):
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                stay_id = re.search('.*_(\d*)_.*', file_name).group(1)
                mortality = mortalities.loc[int(stay_id)].hospital_expire_flag

                #sum each column and delete 'hour'
                sum_column = episode.sum(axis=0)
                sum_column = sum_column.values.tolist()
                del sum_column[0]

                #delete 'pH' because there seems to be some huge outliers here
                del sum_column[15]
                
                #store the imputed values to use with scaler
                data_X += [sum_column]
                data_Y += [mortality]
    
    # #1x(15x95) feature vector(worse so far)
    # for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading values'):
    #     for file_name in files:
    #         if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
    #             episode = pd.read_csv(os.path.join(root, file_name))
    #             stay_id = re.search('.*_(\d*)_.*', file_name).group(1)
    #             mortality = mortalities.loc[int(stay_id)].hospital_expire_flag
    #             #concat every half hour and delete 'hour'
    #             #also delete 'pH' because there seems to be some huge outliers here
                
    #             episode.drop(columns=['hours', 'pH'], axis=1, inplace=True)
    #             columns = []
    #             for index, row in episode.iterrows():
    #                columns += row.tolist()
    #             #columns = [row.tolist() for index, row in episode.iterrows()]

    #             #store the imputed values to use with scaler
    #             data_X += [columns]
    #             data_Y += [mortality]



    scaler = StandardScaler()
    scaler.fit(data_X)
    data_X_standardized = scaler.transform(data_X)
    return(data_X_standardized, data_Y)
    # debug_lista = []
    # debug_lista = [lista[2] for lista in data_X]
    # debug_lista_filtered = []
    # debug_lista_filtered = [item for item in debug_lista if item <300]
    # plt.hist(debug_lista_filtered,density=True, bins=20)
    # plt.show()



#NOTE: COPIED FROM BENCHMARK PROGRAM
# def save_results(names, pred, y_true, path):
#     common_utils.create_directory(os.path.dirname(path))
#     with open(path, 'w') as f:
#         f.write("stay,prediction,y_true\n")
#         for (name, x, y) in zip(names, pred, y_true):
#             f.write("{},{:.6f},{}\n".format(name, x, y))

