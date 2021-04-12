#helper functions for extract_subjects
#will have functions to read the csv files and put the data into dataframes

import os
import pandas as pd


#Read patients data, added dob (only year), do not change to to_datetime
def read_patients_table(mimic4_path):
    patients = pd.read_csv((os.path.join(mimic4_path, 'core', 'patients.csv')))
    patients = patients[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod']]
    patients['dob'] = patients.anchor_year-patients.anchor_age
    return patients

#Read admissions, diagnosis are removed compered to mimic3
def read_admissions_table(mimic4_path):
    admits = pd.read_csv((os.path.join(mimic4_path, 'core', 'admissions.csv')))
    admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']]
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischtime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    return admits

#Read icu stay
def read_icustays_table(mimic4_path):
    stays = pd.read_csv((os.path.join(mimic4_path, 'icu', 'icustays.csv')))
    stays = stays[['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los']]
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    return stays


#Merge on subject admission
def merge_on_subject_admission(table1, table2):
    return table1.merge(table2,  how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

    #Merge on subject 
def merge_on_subject(table1, table2):
    return table1.merge(table2,  how='inner', left_on=['subject_id'], right_on=['subject_id'])



def remove_icustays_with_transfers(stays):
    stays = stays[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]
