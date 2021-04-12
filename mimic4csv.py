#helper functions for extract_subjects
#will have functions to read the csv files and put the data into dataframes

import pandas as pd

#Create datafram from csv files
def datafram_from_csv(path, header=0, index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col)

#Read patients data
def read_patients_table(mimic4_path):
    patients = datafram_from_csv(os.path.join(mimic4_path, 'core/patients.csv'))
    patients = patients[['subject_id', 'gender','anchor_age', 'anchor_year', 'dod', 'dob']]
    patients.dob = pd.to_datetime((patients.anchor_year-patients.age))
    patients.dod = pd.to_datetime(patients.dod)
    return patients

def read_admissions_table(mimic4_path):
    admits = datafram_from_csv(os.path.join(mimic4_path, 'core/admissions.csv'))
    admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']]
    admits.admittime = pd.to_datetime(admit.admittime)
    admits.dischtime = pd.to_datetime(admit.dischtime)
    admits.deathtime = pd.to_datetime(admit.deathtime)
    return admits