#used to prepare the features for in hopsital mortality predictions
#features here are currently a vector of the sum of each individual column tracked in the time series
#probably not a good feature but it will do for now, as we're just testing

from math import isnan
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from metrics import *
from scipy.stats import skew

import matplotlib.pyplot as plt

import re
import json


def scale_values(subjects_root_path):
    data_X = []
    data_Y = []
    mortalities = pd.read_csv(os.path.join(subjects_root_path, 'mortality_summary.csv'))
    mortalities.set_index('stay_id',inplace=True, drop=True)

##OLD FEATURES
    # #1x15 feature vector
    # for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading values'):
    #     for file_name in files:
    #         if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
    #             episode = pd.read_csv(os.path.join(root, file_name))
    #             stay_id = re.search('.*_(\d*)_.*', file_name).group(1)
    #             mortality = mortalities.loc[int(stay_id)].hospital_expire_flag

    #             #sum each column and delete 'hour'
    #             sum_column = episode.sum(axis=0)
    #             sum_column = sum_column.values.tolist()

    #             #extract last row only
    #             # sum_column = episode.loc[94].tolist()

    #             del sum_column[0]

    #             #delete 'pH' because there seems to be some huge outliers here
    #             del sum_column[15]
                
    #             #store the imputed values to use with scaler
    #             data_X += [sum_column]
    #             data_Y += [mortality]
    
###NEW FEATURES
    start_stop_percentages = [(0,1), (0,0.1), (0,0.25), (0,0.5), (1,0.9), (1, 0.75), (1, 0.5)]
    #[min, max, mean, std, skew, len] for 100% of timeseries, first 10/25/50% and last 10/25/50%, for every column
    
    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading values'):
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                stay_id = re.search('.*_(\d*)_.*', file_name).group(1)
                mortality = mortalities.loc[int(stay_id)].hospital_expire_flag
            
                #drop 'pH' for now since it might be bugged
                episode.drop(columns=['pH'], axis=1, inplace=True)
                features = []

                hours = episode['hours'].tolist()
                episode.drop(columns='hours', inplace=True)

                first_value_time = hours[0]
                last_value_time = hours[-1]
                
                features = extract_features(start_stop_percentages, first_value_time, last_value_time, episode, hours)

                #store the imputed values to use with scaler
                data_X += [np.array(features)]
                data_Y += [np.array([mortality])]

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=False)
    imputer.fit(data_X)
    data_X = imputer.transform(data_X)

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

#extract features for a single episode
def extract_features(start_stop_percentages, first_value_time, last_value_time, episode, hours):
    features = []
    #get a dictionary with (time, value) pairs for each column, NOTE: only contains timestamp where there are values, so all 'nans' are removed
    episode_disctionary = convert_timeseries_to_dict(episode, hours)

    #for each column extract the 7 slices and calculate features
    for column in episode_disctionary:
        col_features = []

        for (case, percetage) in start_stop_percentages:
            data = []
            #get indexes for the slice to read
            if(case == 0):
                start_idx = first_value_time
                end_idx = first_value_time + ((last_value_time - first_value_time) * percetage)
            if(case == 1):
                start_idx = last_value_time - ((last_value_time - first_value_time) * percetage)
                end_idx = last_value_time 

            #extract all values within the given timeframe
            data = [value for (time, value) in column if (start_idx <= time <= end_idx)]

            if len(data) == 0:
                #NOTE hardcoded atm
                col_features += [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                continue
            #calculate [min, max, mean, std, skew, len]
            data = [min(data), max(data), np.mean(data), np.std(data), skew(data), len(data)]
            col_features += data

        #store after each column
        features += col_features

    return(features)

#converts a timeseries to dictionary of (time,value) pairs for each column
def convert_timeseries_to_dict(episode, hours):
    dictionary = []

    for column in episode:
        column = episode[column].tolist()

        col_dict = []
        i = 0
        for time in hours:
            value = column[i]
            i += 1
            if not np.isnan(value):
                col_dict +=[(time, value)]
        dictionary += [col_dict]

    return(dictionary)






#NOTE: COPIED FROM BENCHMARK PROGRAM
# def save_results(names, pred, y_true, path):
#     common_utils.create_directory(os.path.dirname(path))
#     with open(path, 'w') as f:
#         f.write("stay,prediction,y_true\n")
#         for (name, x, y) in zip(names, pred, y_true):
#             f.write("{},{:.6f},{}\n".format(name, x, y))

def scale_values_alt(subjects_root_path):
    data_X = []
    data_Y = []
    mortalities = pd.read_csv(os.path.join(subjects_root_path, 'mortality_summary.csv'))
    mortalities.set_index('stay_id',inplace=True, drop=True)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=False)
    #extract the data once to fit the imputer
    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading timeseries'):
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                stay_id = re.search('.*_(\d*)_.*', file_name).group(1)
                mortality = mortalities.loc[int(stay_id)].hospital_expire_flag
                values = episode.values.tolist()
                data_X = data_X + values
                data_Y += [mortality]
    imputer.fit(data_X)
    data_X_imputed = imputer.transform(data_X)

    #sum every 95 rows
    data_X_summarised = []
    n_stays = len(mortalities.index)
    index_start = 0
    index_stop = 95
    increment = 95
    for num in range(n_stays):
        cols = data_X_imputed[index_start:index_stop]
        sum_cols = [sum(i) for i in zip(*cols)]
        index_start += increment
        index_stop += increment
        
        #delete 'hours'
        del sum_cols[0]
        #delete 'pH' because there seems to be some huge outliers here
        del sum_cols[15]
        data_X_summarised += [sum_cols]

    #standardize
    scaler = StandardScaler()
    scaler.fit(data_X_summarised)
    data_X_standardized = scaler.transform(data_X_summarised)

    df = pd.DataFrame(data_X_standardized)
    df.to_csv(os.path.join(subjects_root_path, 'all_data_standardized.csv'))

    return(data_X_standardized, data_Y)


# def read_values(subjects_root_path):
#     data_X = []
#     data_Y=[]
#     mortalities = pd.read_csv(os.path.join(subjects_root_path, 'mortality_summary.csv'))

#     return(data_X, dataY)