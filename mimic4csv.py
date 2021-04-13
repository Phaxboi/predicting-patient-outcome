#helper functions for extract_subjects
#will have functions to read the csv files and put the data into dataframes

import os
import pandas as pd


#Read patients data, added dob (only year), do not change to to_datetime
def read_patients_table(mimic4_path):
    patients = pd.read_csv((os.path.join(mimic4_path, 'core', 'patients.csv')))
    patients = patients[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod', 'hospital_expire_flag']]
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
    stays = stays[['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los']]
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    return stays

#read the diagnoses table
#NOTE: MIMIC-IV has both ICD9 and ICD10 codes, so we have to look at both code and version when merging, not sure
#if this is needed, will have to confirm that this is correct later
def read_icd_diagnoses_table(mimic4_path):
    codes = dataframe_from_csv(os.path.join(mimic4_path, 'd_icd_diagnoses.csv'))
    codes = codes[['icd_code', 'icd_version', 'long_title']]
    diagnoses = dataframe_from_csv(os.path.join(mimic4_path, 'diagnoses_icd.csv'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on=['icd_code', 'icd_version'], right_on=['icd_code', 'icd_verison'])
    diagnoses[['subject_id', 'hadm_id', 'seq_num']] = diagnoses[['subject_id', 'hadm_id', 'seq_num']].astype(int)
    return diagnoses


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


#calculate the add of a patient by subtracting their year of birth by their year of admission
#NOTE: this assumes that 'admittime' in core/admission.csv' is relative to the 'anchor_age', will need
#to verify this
def add_age_to_icustays(stays):
    stays['admityear'] = (stays['admittime']).dt.year
    stays['age'] = stays.apply(lambda s : (s['admityear'] - s['dob']), axis=1)
    #stays.loc[stays.AGE < 0, 'age'] = 90
    return stays


#filter patients younger than 18
def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays[(stays.age >= min_age) & (stays.age <= max_age)]
    return stays


#add an indicator whether the patient died in hospital 
#NOTE: should be the same as hospital_expire_flag, if they're the same this can be removed
def add_inhospital_mortality_to_icustays(stays):
    mortality = stays.dod.notnull() & ((stays.admittime <= stays.dod) & (stays.dischtime >= stays.dod))
    mortality = mortality | (stays.deathtime.notnull() & ((stays.admittime <= stays.deathtime) & (stays.dischtime >= stays.deathtime)))
    stays['mortality_inhospital'] = mortality.astype(int)
    return stays


#add an indicator whether the patient died in the current care unit
#should compare this to 'mortality_inhospitalä' to ensure it working properly
def add_inunit_mortality_to_icustays(stays):
    mortality = stays.dod.notnull() & ((stays.intime <= stays.dod) & (stays.outtime >= stays.dod))
    mortality = mortality | (stays.deathtime.notnull() & ((stays.intime <= stays.deathtime) & (stays.outtime >= stays.deathtime)))
    stays['mortality_inunit'] = mortality.astype(int)
    return stays


#match diagnoses to a hospital stay by merging 
def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['subject_id', 'hadm_id', 'icustay_id']].drop_duplicates(), how='inner',
                           left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

