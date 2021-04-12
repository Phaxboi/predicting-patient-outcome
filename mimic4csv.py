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
    stays = stays[['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'last_careunit' 'intime', 'outtime', 'los']]
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    return stays


#Merge on subject admission
def merge_on_subject_admission(table1, table2):
    return table1.merge(table2,  how='inner', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

#Merge on subject 
def merge_on_subject(table1, table2):
    return table1.merge(table2,  how='inner', left_on=['subject_id'], right_on=['subject_id'])


#removes ICU stays which start and end in different ICU types
#NOTE: in MIMIC III there was a first/last wardID variable in order to filter out those who changed ICU unit,
# but stayed in the same ICU type, not sure how to do this for MIMIC-IV since the variable is gone
def remove_icustays_with_transfers(stays):
    stays = stays[(stays.first_careunit == stays.last_careunit)]
    return stays[['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'last_careunit' 'intime', 'outtime', 'los']]


#filter out admissions which have more(or less) than one ICU stay per admission
def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('hadm_id').count()[['stay_id']].reset_index()
    to_keep = to_keep[(to_keep.stay_id >= min_nb_stays) & (to_keep.stay_id <= max_nb_stays)][['hadm_id']]
    stays = stays.merge(to_keep, how='inner', left_on='hadm_id', right_on='hadm_id')
    return stays


#filter patients younger than 18
def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays[(stays.age >= min_age) & (stays.age <= max_age)]
    return stays

