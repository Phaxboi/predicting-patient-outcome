#helper functions for extract_subjects
#will have functions to read the csv files and put the data into dataframes

import pandas as pd

#Create datafram from csv files
def datafram_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col)

#Read patients data, added dob, funkar det att göra så???? 
def read_patients_table(mimic4_path):
    patients = datafram_from_csv(os.path.join(mimic4_path, 'core/patients.csv'))
    patients = patients[['subject_id', 'gender','anchor_age', 'anchor_year', 'dod', 'dob']]
    patients.dob = pd.to_datetime((patients.anchor_year-patients.age))
    patients.dod = pd.to_datetime(patients.dod)
    return patients

#Read admissions, diagnosis are removed compered to mimic3
def read_admissions_table(mimic4_path):
    admits = datafram_from_csv(os.path.join(mimic4_path, 'core/admissions.csv'))
    admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']]
    admits.admittime = pd.to_datetime(admit.admittime)
    admits.dischtime = pd.to_datetime(admit.dischtime)
    admits.deathtime = pd.to_datetime(admit.deathtime)
    return admits

#Read icu stay
def read_icustays_table(mimic4_path):
    stays = datafram_from_csv(os.path.join(mimic4_path, 'icu/icustays.csv'))
    stays = stays[['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los']]
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    return stays

#Merge on subject admission ((stays, admits))
def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=[subject_id, hadm_id], right_on=[subject_id, hadm_id])
    