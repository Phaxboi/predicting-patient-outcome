#contains helper functions for extract_subjects


import os
import pandas as pd
import numpy as np

from tqdm import tqdm


#read the 'core/patients.csv' file and extracts the fields we want
def read_patients_table(mimic4_path)
    patients = pd.read_csv(os.path.join(mimic4_path, 'core', 'patients.csv'))
    patients['dob'] = patients.anchor_year - patients.anchor_age
    patients = patients[['subject_id', 'gender', 'dob', 'dod']]
    return patients

#read the 'core/admissions.csv' file and extract the fields we want
def read_admissions_table(mimic4_path)
    admissions = pd.read_csv(os.path.join(mimic4_path, 'core', 'admissions.csv'))
    admissions = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag']]
    admissions.admittime = pd.to_datetime(admissions.admittime)
    admissions.dischtime = pd.to_datetime(admissions.dischtime)
    admissions.deathtime = pd.to_datetime(admissions.deathtime)
    return admissions

#read the 'core/transfers.csv' file and extract the fields we want
def read_transfers_table(mimic4_path)
    transfers = pd.read_csv(os.path.join(mimic4_path, 'core', 'transfers.csv'))
    transfers = transfers[['subject_id', 'hadm_id', 'transfer_id', 'careunit', 'intime', 'outtime']]
    transfers.intime = pd.to_datetime(transfers.intime)
    transfers.outtime = pd.to_datetime(transfers.outtime)
    return transfers

#read the 'ICU/icustays.csv' file and extract the field we want
def read_icustays_table(mimic4_path)
    icustays = pd.read_csv(os.path.join(mimic4_path, 'ICU', 'icustays.csv'))
    icustays = icustays[['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los']]
    icustays.intime = pd.to_datetime(icustays.intime)
    icustays.outtime = pd.to_datetime(icustays.outtime)
    return icustays



