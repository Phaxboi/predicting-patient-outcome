#this files contains various functions in order to create the feature needed for the regression models

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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


parser = argparse.ArgumentParser()
parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
args = parser.parse_args()

subjects_root_path = args.subjects_root_path

#can probably be done easier and cleaner
def translate_columns(timeseries):
    for column_title in columns_to_translate:
        col = timeseries[column_title]
        for key, value in columns_to_translate_dictionary[column_title].items():
            col = col.replace(key, value)
        timeseries[column_title] = col
    return timeseries

#impute missing values
#input: list of episodes(dataframes)
#output: list of episodes with imputed values 
def impute(episodes):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=False)
    data_X = []
    for episode in episodes:
        data_X = np.concatenate((data_X, episode))
    imputer.fit(episodes)
    episodes_imputed = np.array(imputer.transform(episodes), dtype=np.float32)
    return(episodes_imputed)


#TODO itterate over each folder in patient folder, iterate over each episodeX_timeseries.csv files
#concat to list where each entry is a timeseries
#will also write the features to an "episodeX_timeseries_features.csv" file 
def read_timeseries(patients_path):
    episodes_list = []
    for root, dirs, files in os.walk(subjects_root_path):
        # for dir_name in dirs:
        #     print(dir_name)
        episode_counter = 0
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                #translate string values to numeric values, comment out if dont needed
                episode = translate_columns(episode)
                episode_counter += 1
                episode_48h = extract_48h(episode)
                file_name = 'episode' + str(episode_counter) + '_timeseries_48h.csv'
                episode_48h.to_csv(os.path.join(root, file_name), index=False)


                print(file_name)
    return(episodes_list)


#takes a timeseries dataframe, extract the first 48 hours, pads missing half hours with empty rows
#NOTE num_of_col needs to set to the amount of variables we use
def extract_48h(episode):
    num_variables = 18
    #make sure we start at hour zero and drop all hours after 48
    first_index = episode.iloc[0]['hours']
    episode['hours'] = episode['hours'] - first_index

    #create new df with same column names
    column_names = episode.columns
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

#read all episodes
episodes = read_timeseries(subjects_root_path)
#impute missing data
#imputed_timeseries_list = impute(episodes)
#TODO create function to split the 'imputed_timeseries_list' in intervals of 96 lines to re-create 
#the original episodes


# #NOTE test code using one episode, uncomment if needed
# #replace non-numerical values for a test file 
# test_df = pd.read_csv(os.path.join(subjects_root_path, '13317644\episode1_timeseries.csv'), index_col=None)
# test_df = translate_columns(test_df)

#impute values for a test file, and try to reconstruct it
# test_df_list = test_df.values.tolist()
# test_df_list = impute(test_df_list)
# print(test_df_list.size)
# lista = np.concatenate((test_df_list, test_df_list))
# print(lista.size)
# test_df = pd.DataFrame(test_df_list)
# test_df = test_df.set_index(0)

# test_df.to_csv(os.path.join(subjects_root_path, '13317644\episode1_timeseries.csv'), index=False)


