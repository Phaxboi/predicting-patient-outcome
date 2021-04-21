#contains helper functions for extract_subjects


import os
import pandas as pd
import numpy as np

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



#merge per-patient information with per-admission information
def merge_patients_admissions(patients, admissions):
    patients_info = patients.merge(admissions, how='inner', left_on='subject_id', right_on='subject_id')
    return patients_info

#merge patient+admission information with transfer information
def merge_admissions_transfers(patients_info, transfers):
    patients_info = patients_info.merge(transfers, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])
    return patients_info

#merge with icu stay information
def merge_admissions_stays(patients_info, icustays):
    patients_info = patients_info.merge(icustays, how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])
    return patients_info

#calculate the age of all patients and add that as a column
def add_patient_age(patients_info):
    patients_info['age'] = patients_info.apply(lambda s : ((s['admityear']).dt.year - s['birth_year']), axis=1)
    return patients_info

#filter patients on age
def filter_patients_age(patients_info, min_age=18, max_age=np.inf):
    patients_info = patients_info[(patients_info.age >= min_age) & (patients_info.age <= max_age)]
    return patients_info

def merge_stays_chartevents(patients_info, chart):
    events = patients_info.merge(chart, how='inner', left_on=['subject_id', 'hadm_id', 'stay_id'], right_on=['subject_id', 'hadm_id', 'stay_id'])
    events = events[['subject_id', 'hadm_id', 'stay_id', 'transfer_id', 'itemid', 'admittime', 'dischtime', 'intime_x', 'outtime_x', 'intime_y', 'outtime_y', 'charttime', 'storetime', 'value', 'valuenum', 'valueuom']]
    return events

def expand_events_itemid(events, itemids):
    events = events.merge(itemids, how='inner', left_on=['itemid'], right_on=['itemid'])
    return events