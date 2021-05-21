#Various functions used to create the 48h timeseries and prepare the feature data

import argparse
import os
import numpy as np
import pandas as pd
import re
import statistics

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
def read_timeseries(patients_path):
    episodes_list = []
    for root, dirs, files in tqdm(os.walk(patients_path), desc='generating 48h time series'):
        episode_counter = 0
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                #translate string values to numeric values, comment out if dont needed
                episode_48h = extract_48h(episode)
                episode_48h = translate_columns(episode_48h)
                episode_counter += 1

                subj_id = re.search('.*_(\d*)_.*', file_name).group(1)
                file_name = 'episode' + str(episode_counter) + '_' + str(subj_id) + '_timeseries_48h.csv'
                episode_48h.to_csv(os.path.join(root, file_name), index=False)
    return(episodes_list)


#takes a timeseries dataframe, extract the first 48 hours, pads missing half hours with empty rows
def extract_48h(episode):
    #make sure we start at hour zero and drop all hours after 48
    first_index = episode.iloc[0]['hours']
    episode['hours'] = episode['hours'] - first_index

    episode_48h = episode[episode.hours <= 48]

    return(episode_48h)


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


def remove_outliers_timeseries(subjects_root_path):
    data_X = []
    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading timeseries'):
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                values = episode.values.tolist()
                data_X = data_X + values
    ep = pd.DataFrame(data_X,columns=['hours','Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glasgow coma scale eye opening','Glasgow coma scale motor response','Glasgow coma scale verbal response','Glucose','Heart rate','Height','Mean blood pressure','Oxygen saturation','Respiratory Rate','Systolic blood pressure','Temperature','Weight','pH'])
    ep_mean = ep.mean(axis=0, skipna=True)
    ep_median = ep.median(axis=0, skipna=True)
    ep_std = ep.std(axis=0, skipna=True)
    ep_std_median = (ep - ep_median) ** 2
    print(ep_std_median.sum(axis=0, skipna=True))
    print(len(ep_std_median))
    ep_std_median = np.sqrt(ep_std_median.sum(axis=0, skipna=True) / len(ep_std_median))
    ep_q25 = ep.quantile(q=0.25, axis=0)
    ep_q75 = ep.quantile(q=0.75, axis=0)
    ep_iqr = (ep_q75 - ep_q25)*1.5
    lower = ep_q25 - ep_iqr
    high = ep_q75 + ep_iqr
    
    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading timeseries'):
        episode_counter = 0
        scale_factor = 2
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode_counter += 1
                episode = pd.read_csv(os.path.join(root, file_name),index_col = False)

                ## Standard deviation with median ##
                # a = np.array(episode['Diastolic blood pressure'].values.tolist())
                # episode['Diastolic blood pressure'] = np.where(a > (ep_median['Diastolic blood pressure'] + scale_factor*ep_std_median['Diastolic blood pressure']), np.nan, a).tolist()
                # episode['Diastolic blood pressure'] = np.where(a < (ep_median['Diastolic blood pressure'] - scale_factor*ep_std_median['Diastolic blood pressure']), np.nan, a).tolist()
                
                # a = np.array(episode['Fraction inspired oxygen'].values.tolist())
                # episode['Fraction inspired oxygen'] = np.where(a > (ep_median['Fraction inspired oxygen'] + scale_factor*ep_std_median['Fraction inspired oxygen']), np.nan, a).tolist()
                # episode['Fraction inspired oxygen'] = np.where(a < (ep_median['Fraction inspired oxygen'] - scale_factor*ep_std_median['Fraction inspired oxygen']), np.nan, a).tolist()
                
                # a = np.array(episode['Glucose'].values.tolist())
                # episode['Glucose'] = np.where(a > (ep_median['Glucose'] + scale_factor*ep_std_median['Glucose']), np.nan, a).tolist()
                # episode['Glucose'] = np.where(a < (ep_median['Glucose'] - scale_factor*ep_std_median['Glucose']), np.nan, a).tolist()
                
                # a = np.array(episode['Heart rate'].values.tolist())
                # episode['Heart rate'] = np.where(a > (ep_median['Heart rate'] + scale_factor*ep_std_median['Heart rate']), np.nan, a).tolist()
                # episode['Heart rate'] = np.where(a < (ep_median['Heart rate'] - scale_factor*ep_std_median['Heart rate']), np.nan, a).tolist()
                
                # a = np.array(episode['Height'].values.tolist())
                # episode['Height'] = np.where(a > (ep_median['Height'] + scale_factor*ep_std_median['Height']), np.nan, a).tolist()
                # episode['Height'] = np.where(a < (ep_median['Height'] - scale_factor*ep_std_median['Height']), np.nan, a).tolist()
                
                # a = np.array(episode['Mean blood pressure'].values.tolist())
                # episode['Mean blood pressure'] = np.where(a > (ep_median['Mean blood pressure'] + scale_factor*ep_std_median['Mean blood pressure']), np.nan, a).tolist()
                # episode['Mean blood pressure'] = np.where(a < (ep_median['Mean blood pressure'] - scale_factor*ep_std_median['Mean blood pressure']), np.nan, a).tolist()
                
                # a = np.array(episode['Oxygen saturation'].values.tolist())
                # episode['Oxygen saturation'] = np.where(a > (ep_median['Oxygen saturation'] + scale_factor*ep_std_median['Oxygen saturation']), np.nan, a).tolist()
                # episode['Oxygen saturation'] = np.where(a < (ep_median['Oxygen saturation'] - scale_factor*ep_std_median['Oxygen saturation']), np.nan, a).tolist()
                
                # a = np.array(episode['Respiratory Rate'].values.tolist())
                # episode['Respiratory Rate'] = np.where(a > (ep_median['Respiratory Rate'] + scale_factor*ep_std_median['Respiratory Rate']), np.nan, a).tolist()
                # episode['Respiratory Rate'] = np.where(a < (ep_median['Respiratory Rate'] - scale_factor*ep_std_median['Respiratory Rate']), np.nan, a).tolist()
                
                # a = np.array(episode['Systolic blood pressure'].values.tolist())
                # episode['Systolic blood pressure'] = np.where(a > (ep_median['Systolic blood pressure'] + scale_factor*ep_std_median['Systolic blood pressure']), np.nan, a).tolist()
                # episode['Systolic blood pressure'] = np.where(a < (ep_median['Systolic blood pressure'] - scale_factor*ep_std_median['Systolic blood pressure']), np.nan, a).tolist()
                
                # a = np.array(episode['Temperature'].values.tolist())
                # episode['Temperature'] = np.where(a > (ep_median['Temperature'] + scale_factor*ep_std_median['Temperature']), np.nan, a).tolist()
                # episode['Temperature'] = np.where(a < (ep_median['Temperature'] - scale_factor*ep_std_median['Temperature']), np.nan, a).tolist()
                
                # a = np.array(episode['Weight'].values.tolist())
                # episode['Weight'] = np.where(a > (ep_median['Weight'] + scale_factor*ep_std_median['Weight']), np.nan, a).tolist()
                # episode['Weight'] = np.where(a < (ep_median['Weight'] - scale_factor*ep_std_median['Weight']), np.nan, a).tolist()
                
                # a = np.array(episode['pH'].values.tolist())
                # episode['pH'] = np.where(a > (ep_median['pH'] + scale_factor*ep_std_median['pH']), np.nan, a).tolist()
                # episode['pH'] = np.where(a < (ep_median['pH'] - scale_factor*ep_std_median['pH']), np.nan, a).tolist()
                
                ## iqr ##
                a = np.array(episode['Diastolic blood pressure'].values.tolist())
                episode['Diastolic blood pressure'] = np.where(a > high['Diastolic blood pressure'], np.nan, a).tolist()
                episode['Diastolic blood pressure'] = np.where(a < lower['Diastolic blood pressure'], np.nan, a).tolist()
                
                a = np.array(episode['Fraction inspired oxygen'].values.tolist())
                episode['Fraction inspired oxygen'] = np.where(a > high['Fraction inspired oxygen'], np.nan, a).tolist()
                episode['Fraction inspired oxygen'] = np.where(a < lower['Fraction inspired oxygen'], np.nan, a).tolist()
  
                a = np.array(episode['Glucose'].values.tolist())
                episode['Glucose'] = np.where(a > high['Glucose'], np.nan, a).tolist()
                episode['Glucose'] = np.where(a < lower['Glucose'], np.nan, a).tolist()
                
                a = np.array(episode['Heart rate'].values.tolist())
                episode['Heart rate'] = np.where(a > high['Heart rate'], np.nan, a).tolist()
                episode['Heart rate'] = np.where(a < lower['Heart rate'], np.nan, a).tolist()
                
                a = np.array(episode['Height'].values.tolist())
                episode['Height'] = np.where(a > high['Height'], np.nan, a).tolist()
                episode['Height'] = np.where(a < lower['Height'], np.nan, a).tolist()
                
                a = np.array(episode['Mean blood pressure'].values.tolist())
                episode['Mean blood pressure'] = np.where(a > high['Mean blood pressure'], np.nan, a).tolist()
                episode['Mean blood pressure'] = np.where(a < lower['Mean blood pressure'], np.nan, a).tolist()
                
                a = np.array(episode['Oxygen saturation'].values.tolist())
                episode['Oxygen saturation'] = np.where(a > high['Oxygen saturation'], np.nan, a).tolist()
                episode['Oxygen saturation'] = np.where(a < lower['Oxygen saturation'], np.nan, a).tolist()
                
                a = np.array(episode['Respiratory Rate'].values.tolist())
                episode['Respiratory Rate'] = np.where(a > high['Respiratory Rate'], np.nan, a).tolist()
                episode['Respiratory Rate'] = np.where(a < lower['Respiratory Rate'], np.nan, a).tolist()
         
                a = np.array(episode['Systolic blood pressure'].values.tolist())
                episode['Systolic blood pressure'] = np.where(a > high['Systolic blood pressure'], np.nan, a).tolist()
                episode['Systolic blood pressure'] = np.where(a < lower['Systolic blood pressure'], np.nan, a).tolist()
                
                a = np.array(episode['Temperature'].values.tolist())
                episode['Temperature'] = np.where(a > high['Temperature'], np.nan, a).tolist()
                episode['Temperature'] = np.where(a < lower['Temperature'], np.nan, a).tolist()
                
                a = np.array(episode['Weight'].values.tolist())
                episode['Weight'] = np.where(a > high['Weight'], np.nan, a).tolist()
                episode['Weight'] = np.where(a < lower['Weight'], np.nan, a).tolist()
         
                a = np.array(episode['pH'].values.tolist())
                episode['pH'] = np.where(a > high['pH'], np.nan, a).tolist()
                episode['pH'] = np.where(a < lower['pH'], np.nan, a).tolist()

                ## mean value ##
                # a = np.array(episode['Diastolic blood pressure'].values.tolist())
                # episode['Diastolic blood pressure'] = np.where(a > (ep_mean['Diastolic blood pressure'] + scale_factor*ep_std['Diastolic blood pressure']), np.nan, a).tolist()
                # episode['Diastolic blood pressure'] = np.where(a < (ep_mean['Diastolic blood pressure'] - scale_factor*ep_std['Diastolic blood pressure']), np.nan, a).tolist()
                
                # a = np.array(episode['Fraction inspired oxygen'].values.tolist())
                # episode['Fraction inspired oxygen'] = np.where(a > (ep_mean['Fraction inspired oxygen'] + scale_factor*ep_std['Fraction inspired oxygen']), np.nan, a).tolist()
                # episode['Fraction inspired oxygen'] = np.where(a < (ep_mean['Fraction inspired oxygen'] - scale_factor*ep_std['Fraction inspired oxygen']), np.nan, a).tolist()
                
                # a = np.array(episode['Glucose'].values.tolist())
                # episode['Glucose'] = np.where(a > (ep_mean['Glucose'] + scale_factor*ep_std['Glucose']), np.nan, a).tolist()
                # episode['Glucose'] = np.where(a < (ep_mean['Glucose'] - scale_factor*ep_std['Glucose']), np.nan, a).tolist()
                
                # a = np.array(episode['Heart rate'].values.tolist())
                # episode['Heart rate'] = np.where(a > (ep_mean['Heart rate'] + scale_factor*ep_std['Heart rate']), np.nan, a).tolist()
                # episode['Heart rate'] = np.where(a < (ep_mean['Heart rate'] - scale_factor*ep_std['Heart rate']), np.nan, a).tolist()
                
                # a = np.array(episode['Height'].values.tolist())
                # episode['Height'] = np.where(a > (ep_mean['Height'] + scale_factor*ep_std['Height']), np.nan, a).tolist()
                # episode['Height'] = np.where(a < (ep_mean['Height'] - scale_factor*ep_std['Height']), np.nan, a).tolist()
                
                # a = np.array(episode['Mean blood pressure'].values.tolist())
                # episode['Mean blood pressure'] = np.where(a > (ep_mean['Mean blood pressure'] + scale_factor*ep_std['Mean blood pressure']), np.nan, a).tolist()
                # episode['Mean blood pressure'] = np.where(a < (ep_mean['Mean blood pressure'] - scale_factor*ep_std['Mean blood pressure']), np.nan, a).tolist()
                
                # a = np.array(episode['Oxygen saturation'].values.tolist())
                # episode['Oxygen saturation'] = np.where(a > (ep_mean['Oxygen saturation'] + scale_factor*ep_std['Oxygen saturation']), np.nan, a).tolist()
                # episode['Oxygen saturation'] = np.where(a < (ep_mean['Oxygen saturation'] - scale_factor*ep_std['Oxygen saturation']), np.nan, a).tolist()
                
                # a = np.array(episode['Respiratory Rate'].values.tolist())
                # episode['Respiratory Rate'] = np.where(a > (ep_mean['Respiratory Rate'] + scale_factor*ep_std['Respiratory Rate']), np.nan, a).tolist()
                # episode['Respiratory Rate'] = np.where(a < (ep_mean['Respiratory Rate'] - scale_factor*ep_std['Respiratory Rate']), np.nan, a).tolist()
                
                # a = np.array(episode['Systolic blood pressure'].values.tolist())
                # episode['Systolic blood pressure'] = np.where(a > (ep_mean['Systolic blood pressure'] + scale_factor*ep_std['Systolic blood pressure']), np.nan, a).tolist()
                # episode['Systolic blood pressure'] = np.where(a < (ep_mean['Systolic blood pressure'] - scale_factor*ep_std['Systolic blood pressure']), np.nan, a).tolist()
                
                # a = np.array(episode['Temperature'].values.tolist())
                # episode['Temperature'] = np.where(a > (ep_mean['Temperature'] + scale_factor*ep_std['Temperature']), np.nan, a).tolist()
                # episode['Temperature'] = np.where(a < (ep_mean['Temperature'] - scale_factor*ep_std['Temperature']), np.nan, a).tolist()
                
                # a = np.array(episode['Weight'].values.tolist())
                # episode['Weight'] = np.where(a > (ep_mean['Weight'] + scale_factor*ep_std['Weight']), np.nan, a).tolist()
                # episode['Weight'] = np.where(a < (ep_mean['Weight'] - scale_factor*ep_std['Weight']), np.nan, a).tolist()
                
                # a = np.array(episode['pH'].values.tolist())
                # episode['pH'] = np.where(a > (ep_mean['pH'] + scale_factor*ep_std['pH']), np.nan, a).tolist()
                # episode['pH'] = np.where(a < (ep_mean['pH'] - scale_factor*ep_std['pH']), np.nan, a).tolist()
                

                subj_id = re.search('.*_(\d*)_.*', file_name).group(1)
                file_name = 'episode' + str(episode_counter) + '_' + str(subj_id) + '_timeseries_48h.csv'
                episode.to_csv(os.path.join(root, file_name), index=False)
    return episode            

def plotEpisode(subjects_root_path):
    data_X = []
    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='Plot'):
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))
                values = episode.values.tolist()
                data_X = data_X + values
    ep = pd.DataFrame(data_X,columns=['hours','Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glasgow coma scale eye opening','Glasgow coma scale motor response','Glasgow coma scale verbal response','Glucose','Heart rate','Height','Mean blood pressure','Oxygen saturation','Respiratory Rate','Systolic blood pressure','Temperature','Weight','pH'])
    # print(ep.mean(axis=0, skipna=True))
    # print(ep.median(axis=0, skipna=True))
    # print(ep.std(axis=  0, skipna=True))
    ep.boxplot(column=['Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glasgow coma scale eye opening','Glasgow coma scale motor response','Glasgow coma scale verbal response','Glucose','Heart rate','Height','Mean blood pressure','Oxygen saturation','Respiratory Rate','Systolic blood pressure','Temperature','Weight','pH'])
    ep.hist(column=['Capillary refill rate','Diastolic blood pressure','Fraction inspired oxygen','Glasgow coma scale eye opening','Glasgow coma scale motor response','Glasgow coma scale verbal response','Glucose','Heart rate','Height','Mean blood pressure','Oxygen saturation','Respiratory Rate','Systolic blood pressure','Temperature','Weight','pH'],bins=100)
    plt.show()
    return ep