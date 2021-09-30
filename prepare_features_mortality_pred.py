#used to prepare the features for in hopsital mortality predictions

from math import isnan
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew
from gensim.models import Word2Vec
import json
import re




def extract_features(subjects_root_path, use_categorical_flag, use_word_embeddings):
    data_X = []
    data_Y = []
    mortalities = pd.read_csv(os.path.join(subjects_root_path, 'mortality_summary.csv'))
    mortalities.set_index('stay_id',inplace=True, drop=True)

    
    # if use_word_embeddings:
    #     num_to_category = json.load("num_to_category.json")
    #     wv = Word2Vec.load(corpus_path).wv


    check_filename_function = check_filename(use_categorical_flag)
    get_stay_id = get_stay_id_function(use_categorical_flag)
    
    #NEW FEATURES
    start_stop_percentages = [(0,1), (0,0.1), (0,0.25), (0,0.5), (1,0.9), (1, 0.75), (1, 0.5)]
    #[min, max, mean, std, skew, len] for 100% of timeseries, first 10/25/50% and last 10/25/50%, for every column
    
    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading values'):
        for file_name in files:
            #read only desiered files
            if(check_filename_function(file_name)):
                episode = pd.read_csv(os.path.join(root, file_name))
                stay_id = get_stay_id(file_name)
                mortality = mortalities.loc[int(stay_id)].hospital_expire_flag
            
                features = []

                hours = episode['hours'].tolist()
                episode.drop(columns='hours', inplace=True)

                first_value_time = hours[0]
                last_value_time = hours[-1]
                
                features = extract_features_single_episode(start_stop_percentages, first_value_time, last_value_time, episode, hours)

                #store the imputed values to use with scaler
                data_X += [np.array(features)]
                data_Y += [np.array([mortality])]

    #NOTE:code to save min and max values, 16 features total, pH included
    # values = []
    # for row in data_X:
    #     values_row = []
    #     for i in range(16):
    #         values_row += [row[(i*42)], row[(i*42+1)] ]
    #     values += [values_row]

    # df = pd.DataFrame(values)
    # df.to_csv(os.path.join(subjects_root_path, 'result\\min_max_values.csv'))


    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=False)
    imputer.fit(data_X)
    data_X = imputer.transform(data_X)

    scaler = StandardScaler()
    scaler.fit(data_X)
    data_X_standardized = scaler.transform(data_X)
    return(data_X_standardized, data_Y)



#extract features for a single episode
def extract_features_single_episode(start_stop_percentages, first_value_time, last_value_time, episode, hours):
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



def check_filename(use_categorical_flag):
    if(use_categorical_flag):
        func = check_filename_categorical
    else:
        func = check_filename_numerical
    return(func)

def check_filename_categorical(filename):
    return (filename.startswith('num_category'))

def check_filename_numerical(filename):
    return (filename.startswith('episode') & filename.endswith('timeseries_48h.csv'))




def get_stay_id_function(use_categorical_flag):
    if(use_categorical_flag):
        func = get_stay_id_categorical
    else:
        func = get_stay_id_numerical
    return(func)

def get_stay_id_categorical(filename):
    return(re.search('.*_(\d*).*', filename).group(1))

def get_stay_id_numerical(filename):
    return(re.search('.*_(\d*)_.*', filename).group(1))
                










# def scale_values_alt(subjects_root_path):
#     data_X = []
#     data_Y = []
#     mortalities = pd.read_csv(os.path.join(subjects_root_path, 'mortality_summary.csv'))
#     mortalities.set_index('stay_id',inplace=True, drop=True)

#     imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=False)
#     #extract the data once to fit the imputer
#     for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading timeseries'):
#         for file_name in files:
#             if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
#                 episode = pd.read_csv(os.path.join(root, file_name))
#                 stay_id = re.search('.*_(\d*)_.*', file_name).group(1)
#                 mortality = mortalities.loc[int(stay_id)].hospital_expire_flag
#                 values = episode.values.tolist()
#                 data_X = data_X + values
#                 data_Y += [mortality]
#     imputer.fit(data_X)
#     data_X_imputed = imputer.transform(data_X)

#     #sum every 95 rows
#     data_X_summarised = []
#     n_stays = len(mortalities.index)
#     index_start = 0
#     index_stop = 95
#     increment = 95
#     for num in range(n_stays):
#         cols = data_X_imputed[index_start:index_stop]
#         sum_cols = [sum(i) for i in zip(*cols)]
#         index_start += increment
#         index_stop += increment
        
#         #delete 'hours'
#         del sum_cols[0]
#         #delete 'pH' because there seems to be some huge outliers here
#         del sum_cols[15]
#         data_X_summarised += [sum_cols]

#     #standardize
#     scaler = StandardScaler()
#     scaler.fit(data_X_summarised)
#     data_X_standardized = scaler.transform(data_X_summarised)

#     df = pd.DataFrame(data_X_standardized)
#     df.to_csv(os.path.join(subjects_root_path, 'all_data_standardized.csv'))

#     return(data_X_standardized, data_Y)

