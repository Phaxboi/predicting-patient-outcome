#Various functions used to create the 48h timeseries and prepare the feature data

import os
import numpy as np
from numpy.core.defchararray import upper
from numpy.lib.function_base import percentile
import pandas as pd
import re
from pandas.core.reshape.tile import cut
from get_outlier_thresholds import remove_outliers

from sklearn.impute import SimpleImputer
from tqdm import tqdm

import matplotlib.pyplot as plt

#name of the columns that needs to be translated to numeric values
columns_to_translate = ["Capillary refill rate", "Glasgow coma scale verbal response", "Glasgow coma scale eye opening", "Glasgow coma scale motor response"]

#NOTE WARNING SUPRESSION
pd.options.mode.chained_assignment = None


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
def translate_columns(t):
    timeseries = t
    for column_title in columns_to_translate:
        col = timeseries[column_title]
        for key, value in columns_to_translate_dictionary[column_title].items():
            col = col.replace(key, value)
        timeseries[column_title] = col
    return timeseries

#impute missing values for all 48h files in the given folder
#NOTE:will read files two times for easier debugging, can proabably be done in one read only
def impute(subjects_root_path):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=False)
    data_X = []
    #extract the data once to fit the imputer
    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading timeseries'):
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                values = episode.values.tolist()
                data_X = data_X + values
    imputer.fit(data_X)



    data_X = []
    column_names = episode.columns
    #open each file and impute missing values
    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='imputing'):
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


    return(episode_imputed)


#read all episodes, transtale text data into numerical values and extract only first 48h
def read_timeseries(patients_path, fileendswith):

    #read all timeseries values, calculate iqr values to filter outliers, will only 
    thresholds_csv = pd.read_csv(os.path.join(patients_path,'result\\outlier_thresholds' + fileendswith))
    thresholds = [(a,b) for [a,b] in list(thresholds_csv.values)]
    

    episodes_list = []
    for root, dirs, files in tqdm(os.walk(patients_path), desc='generating 48h time series'):
        episode_counter = 0
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries' + fileendswith)):
                episode = pd.read_csv(os.path.join(root, file_name))
                episode_48h = extract_48h(episode)
                #translate string values to numeric values, comment out if dont needed
                episode_48h = translate_columns(episode_48h)
                episode_counter += 1

                episode_48h = remove_outliers(episode_48h, thresholds)

                subj_id = re.search('.*_(\d*)_.*', file_name).group(1)
                file_name = 'episode' + str(episode_counter) + '_' + str(subj_id) + '_timeseries_48h' + fileendswith
                episode_48h.to_csv(os.path.join(root, file_name), index=False)


    return(episodes_list)


#takes a timeseries dataframe, extract the first 48 hours
def extract_48h(episode):
    #remove all measurements that are too far behind hour 0, 30 minutes at this time, can be changed
    first_timestamp = episode.iloc[0]['hours']
    while first_timestamp < -0.5 :
        episode = episode.iloc[1:]
        first_timestamp = episode.iloc[0]['hours']

    episode['hours'] = episode['hours'] - first_timestamp

    episode_48h = episode[episode.hours <= 48]

    return(episode_48h)


# def remove_outliers(episode_48h, thresholds):
#     timeseries = episode_48h
#     i = 0
#     for (outlier_cutoff_lower, outlier_cutoff_upper) in thresholds:
#         filtered = []
#         for x in episode_48h.iloc[:,i+1].tolist():
#             if np.isnan(x):
#                 filtered += [np.nan]
#             elif (outlier_cutoff_lower <= x <= outlier_cutoff_upper):
#                 filtered += [x]
#             else:
#                 filtered += [np.nan]

#         timeseries.iloc[:,i+1] = filtered
#         i +=1
#     return(timeseries)






#NOTE: OLD FUNCTION TO INPUT EMPTY ROWS AT TIMESTAMPS WHERE NO VALUES WERE MEASURED
# #takes a timeseries dataframe, extract the first 48 hours, pads missing half hours with empty rows
# def extract_48h(episode):
#     #make sure we start at hour zero and drop all hours after 48
#     first_index = episode.iloc[0]['hours']
#     episode['hours'] = episode['hours'] - first_index

#     #create new df with same column names
#     column_names = episode.columns
#     num_variables = column_names.shape[0]
#     episode_48h = pd.DataFrame(columns=column_names)

#     #give 'hour' column value 0-48 in 0.5 intervals
#     hours = np.array([*range(0, 95)],dtype=float)
#     hours = hours/2
#     episode_48h['hours'] = hours

#     #merge with 'episode'
#     episode_48h = episode.merge(episode_48h, how='right', left_on=['hours'], right_on=['hours'])
#     episode_48h = episode_48h.iloc[:,:num_variables]
#     episode_48h.columns = column_names

#     return(episode_48h)
          

def plotEpisode(subjects_root_path, fileendswith):
    data_X = []
    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='Plot'):
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h' + fileendswith)):
                episode = pd.read_csv(os.path.join(root, file_name))
                values = episode.values.tolist()
                data_X = data_X + values
    ep = pd.DataFrame(data_X,columns=['hours','Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glasgow coma scale eye opening','Glasgow coma scale motor response','Glasgow coma scale verbal response','Glucose','Heart rate','Height','Mean blood pressure','Oxygen saturation','Respiratory Rate','Systolic blood pressure','Temperature','Weight','pH'])
    ep.hist(column=['Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glasgow coma scale eye opening','Glasgow coma scale motor response','Glasgow coma scale verbal response','Glucose','Heart rate','Height','Mean blood pressure','Oxygen saturation','Respiratory Rate','Systolic blood pressure','Temperature','Weight','pH'],bins=100)
    plt.show()
    return ep
