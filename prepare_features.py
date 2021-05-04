#this files contains various functions in order to create the feature needed for the regression models

import argparse
import os
import numpy as np
import pandas as pd
import re
import statistics

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

#name of the columns that needs to be translated to numeric values
columns_to_translate = ["Capillary refill rate", "Glasgow coma scale verbal response", "Glasgow coma scale eye opening", "Glasgow coma scale motor response"]






columns_to_translate_dictionary = {
    "Capillary refill rate":{
        "Normal <3 Seconds": 0,
        "Abnormal >3 Seconds": 1
    },
    "Glasgow coma scale verbal response":{
        "No Response-ETT": 1,
        "No Response": 1,
        "1 No Response": 1,
        "1.0 ET/Trach": 1,
        "2 Incomp sounds": 2,
        "Incomprehensible sounds": 2,
        "3 Inapprop words": 3,
        "Inappropriate Words": 3,
        "4 Confused": 4,
        "Confused": 4,
        "5 Oriented": 5,
        "Oriented": 5
    }, 
    "Glasgow coma scale eye opening":{
        "None": 0,
        "1 No Response": 1,
        "2 To pain": 2, 
        "To Pain": 2,
        "3 To speech": 3, 
        "To Speech": 3,
        "4 Spontaneously": 4,
        "Spontaneously": 4
    }, 
    "Glasgow coma scale motor response":{
        "1 No Response": 1,
        "No response": 1,
        "2 Abnorm extensn": 2,
        "Abnormal extension": 2,
        "3 Abnorm flexion": 3,
        "Abnormal Flexion": 3,
        "4 Flex-withdraws": 4,
        "Flex-withdraws": 4,
        "5 Localizes Pain": 5,
        "Localizes Pain": 5,
        "6 Obeys Commands": 6,
        "Obeys Commands": 6
    }
}


#can probably be done easier and cleaner
def translate_columns(timeseries):
    for column_title in columns_to_translate:
        col = timeseries[column_title]
        for key, value in columns_to_translate_dictionary[column_title].items():
            col = col.replace(key, value)
        timeseries[column_title] = col
    return timeseries

#impute missing values for all 48h files in the given folder
#also normalizes the data to have zero mean and unit variance
#NOTE:will also read three times for debug purposes, might be too slow and can probably be done much faster
def impute_and_scale(subjects_root_path):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=False)
    data_X = []
    #extract the data once to fit the imputer
    for root, dirs, files in os.walk(subjects_root_path):
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                values = episode.values.tolist()
                data_X = data_X + values
    imputer.fit(data_X)

    # #data before imputation
    # debug_lista = [lista[2] for lista in data_X]
    # plt.hist(debug_lista)
    # plt.show()
    # debug_lista = []

    data_X = []
    column_names = episode.columns
    #open each file and impute missing values
    for root, dirs, files in os.walk(subjects_root_path):
        episode_counter = 0
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode_counter += 1
                episode = pd.read_csv(os.path.join(root, file_name))
                
                episode_imputed = np.array(imputer.transform(episode), dtype=np.float32)
                episode_imputed = pd.DataFrame(episode_imputed)
                episode_imputed.columns = column_names

                #store the imputed values to use with scaler
                values = episode_imputed.values.tolist()
                data_X = data_X + values
                
                subj_id = re.search('.*_(\d*)_.*', file_name).group(1)
                file_name = 'episode' + str(episode_counter) + '_' + str(subj_id) + '_timeseries_48h.csv'
                episode_imputed.to_csv(os.path.join(root, file_name), index=False)

    #fit scaler using the imputed
    scaler = StandardScaler()
    scaler.fit(data_X)

    # #data after imputing
    # print(scaler.get_params())
    # debug_lista = []
    # debug_lista = [lista[2] for lista in data_X]
    # plt.hist(debug_lista)
    # plt.show()

    data_X = []
    #open each file and normalizes the data to have zero mean and unit variance
    for root, dirs, files in os.walk(subjects_root_path):
        episode_counter = 0
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode_counter += 1
                episode = pd.read_csv(os.path.join(root, file_name))
                
                episode_normalized = np.array(scaler.transform(episode), dtype=np.float32)
                episode_normalized = pd.DataFrame(episode_normalized)
                episode_normalized.columns = column_names

                #for debugging
                values = episode_normalized.values.tolist()
                data_X = data_X + values
 
                subj_id = re.search('.*_(\d*)_.*', file_name).group(1)
                file_name = 'episode' + str(episode_counter) + '_' + str(subj_id) + '_timeseries_48h.csv'
                episode_normalized.to_csv(os.path.join(root, file_name), index=False)

    # #data after normalizing
    # debug_lista = []
    # debug_lista = [lista[2] for lista in data_X]
    # plt.hist(debug_lista)
    # plt.show()
    # print('Variance of col 3 is:' + str(statistics.variance(debug_lista)))

    return(episode_imputed)


#read all episodes, transtale text data into numerical values and extract only first 48h
def read_timeseries(patients_path):
    episodes_list = []
    for root, dirs, files in os.walk(subjects_root_path):
        episode_counter = 0
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                #translate string values to numeric values, comment out if dont needed
                episode = translate_columns(episode)
                episode_counter += 1
                episode_48h = extract_48h(episode)

                subj_id = re.search('.*_(\d*)_.*', file_name).group(1)
                file_name = 'episode' + str(episode_counter) + '_' + str(subj_id) + '_timeseries_48h.csv'
                episode_48h.to_csv(os.path.join(root, file_name), index=False)
    return(episodes_list)


#takes a timeseries dataframe, extract the first 48 hours, pads missing half hours with empty rows
def extract_48h(episode):
    #make sure we start at hour zero and drop all hours after 48
    first_index = episode.iloc[0]['hours']
    episode['hours'] = episode['hours'] - first_index

    #create new df with same column names
    column_names = episode.columns
    num_variables = column_names.shape[0]
    episode_48h = pd.DataFrame(columns=column_names)

    #give 'hour' column value 0-48 in 0.5 intervals
    hours = np.array([*range(0, 95)],dtype=float)
    hours = hours/2
    episode_48h['hours'] = hours

    #merge with 'episode'
    episode_48h = episode.merge(episode_48h, how='right', left_on=['hours'], right_on=['hours'])
    episode_48h = episode_48h.iloc[:,:num_variables]
    episode_48h.columns = column_names

    return(episode_48h)



parser = argparse.ArgumentParser()
parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
args = parser.parse_args()

subjects_root_path = args.subjects_root_path

#read all episodes, transtale text data into numerical values and extract only first 48h
episodes = read_timeseries(subjects_root_path)
#impute missing data
imputed_timeseries_list = impute_and_scale(subjects_root_path)








