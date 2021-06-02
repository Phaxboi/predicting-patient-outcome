#contains helper functions for extract_subjects


import os
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
from tqdm import tqdm


#read the 'core/patients.csv' file and extracts the fields we want
def read_patients_table(mimic4_path):
    patients = pd.read_csv(os.path.join(mimic4_path, 'core', 'patients.csv'))
    patients['birth_year'] = patients.anchor_year - patients.anchor_age
    patients = patients[['subject_id', 'gender', 'birth_year', 'dod']]
    return patients

#read the 'core/admissions.csv' file and extract the fields we want
def read_admissions_table(mimic4_path):
    admissions = pd.read_csv(os.path.join(mimic4_path, 'core', 'admissions.csv'))
    admissions = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag']]
    admissions.admittime = pd.to_datetime(admissions.admittime)
    admissions.dischtime = pd.to_datetime(admissions.dischtime)
    admissions.deathtime = pd.to_datetime(admissions.deathtime)
    return admissions

#read the 'core/transfers.csv' file and extract the fields we want
def read_transfers_table(mimic4_path):
    transfers = pd.read_csv(os.path.join(mimic4_path, 'core', 'transfers.csv'))
    transfers = transfers[['subject_id', 'hadm_id', 'transfer_id', 'careunit', 'intime', 'outtime']]
    transfers.intime = pd.to_datetime(transfers.intime)
    transfers.outtime = pd.to_datetime(transfers.outtime)
    return transfers

#read the 'ICU/icustays.csv' file and extract the field we want
def read_icustays_table(mimic4_path):
    icustays = pd.read_csv(os.path.join(mimic4_path, 'ICU', 'icustays.csv'))
    icustays = icustays[['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los']]
    icustays.intime = pd.to_datetime(icustays.intime)
    icustays.outtime = pd.to_datetime(icustays.outtime)
    return icustays

#filter out stays which begin and end in different care units
def filter_icustays_with_transfers(patients_info):
    patients_info = patients_info[(patients_info.first_careunit == patients_info.last_careunit)]
    return patients_info

#filter out ICU stays less than 48 hours
def filter_icustays_48h(patients_info):
    patients_info = patients_info[patients_info.los >= 2]
    return patients_info


#merge per-patient information with per-admission information
def merge_patients_admissions(patients, admissions):
    patients_info = patients.merge(admissions, how='inner', left_on='subject_id', right_on='subject_id')
    return patients_info


#merge with icu stay information
def merge_admissions_stays(patients_info, icustays):
    patients_info = patients_info.merge(icustays, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])
    return patients_info


#calculate the age of all patients and add that as a column
def add_patient_age(patients_info):
    patients_info['age'] = patients_info.apply(lambda s : ((s['admittime']).year - s['birth_year']), axis=1)
    return patients_info


#filter patients on age
def filter_patients_age(patients_info, min_age=18, max_age=np.inf):
    patients_info = patients_info[(patients_info.age >= min_age) & (patients_info.age <= max_age)]
    return patients_info

#function to fix patients with missing the 'deathtime' field, desptie dying in hospital
#NOTE: currently calculates this by 'intime' + 'los', one can also look at the discharge time, but this seems to be 
#inaccurate a lot of times
def fix_missing_deathtimes(patients_info):
    indices = patients_info.index[(patients_info['hospital_expire_flag'] == 1) & (patients_info['deathtime'].isnull())].tolist()
    for index in indices:
        patients_info.at[index, 'deathtime'] = patients_info.at[index, 'intime'] + datetime.timedelta(patients_info.at[index, 'los'])
    return patients_info

#rearranges columns in the given order
def rearrange_columns(patients_info, columns_title):
    patients_info = patients_info.reindex(columns=columns_title)
    return patients_info


#break up stays by patients
#creates a folder for each patient and create a file with a summary of their hospital stays
#also create a file 'mortality_summary' with 'stay_id' and 'hospital_expire_flag'
def break_up_stays_by_subject(patients_info, output_path):
    result_path = os.path.join(output_path, 'result')
    patients = patients_info.subject_id.unique()
    number_of_patients = patients.shape[0] 
    ids = []
    mortality = []
    for pat_id in tqdm(patients, total=number_of_patients, desc='Breaking up stays by subjects'):
        directory = os.path.join(output_path, str(pat_id))
        try:
            os.makedirs(directory)
        except:
            pass

        df = patients_info[patients_info.subject_id == pat_id]
        df.to_csv(os.path.join(directory, 'patient_info_summary.csv'), index=False)
        ids = ids + df['stay_id'].tolist()
        mortality += df['hospital_expire_flag'].tolist()
    mortality_summary = pd.DataFrame({'stay_id': ids, 'hospital_expire_flag':mortality})
    mortality_summary.to_csv(os.path.join(output_path, 'mortality_summary.csv'), index=False)

# Merge patients_info and chartevents.csv. Drop unnecessary columns 
def merge_stays_chartevents(patients_info, chart):
    events = patients_info.merge(chart, how='inner', left_on=['subject_id', 'hadm_id', 'stay_id'], right_on=['subject_id', 'hadm_id', 'stay_id'])
    events = events[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'intime', 'charttime', 'storetime', 'value', 'valueuom']]
    return events

#converts all weight fields to a unified scale (kg)
def fix_weight(events):
    indices = events.index[(events['variable_name'] == 'Weight') & (events['uom'] == 'lbs')]
    for index in indices:
        events.at[index, 'value'] = round(float(events.at[index, 'value']) * 0.453592,1)
        events.at[index, 'uom'] = 'kg'
    return events

# Converts all heights to a unified scale (cm)
def fix_height(events):
    indices = events.index[(events['variable_name'] == 'Height') & (events['uom'] == 'Inch')]
    for index in indices:
        events.at[index, 'value'] = round(float(events.at[index, 'value']) * 2.54,1)
        events.at[index, 'uom'] = 'cm'
    return events

# Converts all temperatures to a unified scale Celsius 
def fix_temperature(events):
    indices = events.index[(events['variable_name'] == 'Temperature') & (events['uom'] == 'F')]
    for index in indices:
        events.at[index, 'value'] = round((float(events.at[index, 'value']) - 32) * (5/9),1)
        events.at[index, 'uom'] = 'C'
    return events

# Convert Events table to Timeseries
def convert_events_timeserie(events, variables):
    meta_data = events[['charttime', 'stay_id']]\
                .sort_values(by=['charttime', 'stay_id'])\
                .drop_duplicates(keep='first').set_index('charttime')
    timeseries = events[['charttime','variable_name', 'value']]\
                .sort_values(by=['charttime', 'variable_name', 'value'], axis=0)\
                .drop_duplicates(subset=['charttime', 'variable_name',], keep='last')
    timeseries = timeseries.pivot(index='charttime', columns='variable_name', values='value')\
                .merge(meta_data, left_index=True, right_index=True)\
                .sort_index(axis=0).reset_index()
    for i in variables:
        if i not in timeseries:
            timeseries[i] = np.nan
    return timeseries

# Get episodes
def get_episode(events, stay_id, intime=None, outtime=None):
    idx = (events.stay_id == stay_id)
    if intime is not None and outtime is not None:
        idx = idx | ((events.charttime >= intime) & (events.charttime <= outtime))
    events = events[idx]
    return events

# Calculate hour and output is rounded to every half hour
def intime_to_hours(episode, intime, remove_charttime=True, remove_stay_id=True):
    episode = episode.copy()
    intime = pd.to_datetime(intime)
    episode['charttime'] = pd.to_datetime(episode.charttime)
    episode['hours'] = episode['charttime'] - intime

    # Datetime to minutes and then round to nearest half hour. round(x/a)*a where x = hour and a = the factor you want (30 min = 0.5)
    episode['hours'] = round(episode['hours'].dt.total_seconds() /(60 * 60 * 0.5))*0.5
    
    if remove_charttime:
        del episode['charttime']
    if remove_stay_id:
        del episode['stay_id']
    return episode

# All events that happen in the same half-hour merge into one row. The mean of the values, and last result for Capillary refill rate. NA does not count.
# NOTE Need to be changed when the labels change. These are comments right now.
def merge_same_hour_to_one_row(episode):
    episode = episode.groupby('hours', as_index=True, sort=False).last()
    return episode
