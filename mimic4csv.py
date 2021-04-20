#contains helper functions for extract_subjects


import os
import pandas as pd
import numpy as np
import datetime

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

# #merge patient+admission information with transfer information
# def merge_admissions_transfers(patients_info, transfers):
#     patients_info = patients_info.merge(transfers, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])
#     return patients_info

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
#NOTE: currently calculates this by 'admittime' + 'los', alternatively we could 
def fix_missing_deathtimes(patients_info):
    indices = patients_info.index[(patients_info['hospital_expire_flag'] == 1) & (patients_info['deathtime'].isnull())].tolist()
    for index in indices:
        patients_info.at[index, 'deathtime'] = patients_info.at[index, 'admittime'] + datetime.timedelta(patients_info.at[index, 'los'])
    return patients_info
#15566609,22132197,2123-04-15 19:52:00,2123-04-21 20:10:00,2123-04-21 20:10:00,EW EMER.,EMERGENCY ROOM,DIED,Medicare,ENGLISH,MARRIED,WHITE,2123-04-15 17:16:00,2123-04-15 20:20:00,1



#break up stays by patients
#creates a folder for each patient and create a file with a summary of their hospital stays 
def break_up_stays_by_subject(patients_info, output_path):
    patients = patients_info.subject_id.unique()
    number_of_patients = patients.shape[0]
    for pat_id in tqdm(patients, total=number_of_patients, desc='Breaking up stays by subjects'):
        directory = os.path.join(output_path, str(pat_id))
        try:
            os.makedirs(directory)
        except:
            pass

        df = patients_info[patients_info.subject_id == pat_id]
        df.to_csv(os.path.join(directory, 'stays.csv'), index=False)
