#helper functions for extract_subjects
#will have functions to read the csv files and put the data into dataframes

import os
import pandas as pd
import numpy as np

from tqdm import tqdm


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
    return stays[['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los']]


#filter out admissions on the number of ICU stay per admission
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


#match diagnoses to a patients hospital stay by merging 
def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['subject_id', 'hadm_id', 'icustay_id']].drop_duplicates(), how='inner',
                           left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])


#identify each unique subject ID, create a folder for them and create a CSV file with a summary of their stays
def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subj_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subj_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays[stays.subject_id == subj_id].sort_values(by='intime').to_csv(os.path.join(dn, 'stays.csv'),
                                                                              index=False)


#identifiy unique subjects, create a folder with a summary of their diagnoses
def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.subject_id.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subj_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        dn = os.path.join(output_path, str(subj_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses[diagnoses.subject_id == subj_id].sort_values(by=['icustay_id', 'seq_num'])\
                                                     .to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)


#identify unique subjects, create folder with csv of their events
#NOTE: not sure how this works so some variable names are wrong since this is a copy paste, will have to
#revisit once we have data to debug/test with
def read_events_table_and_break_up_by_subject(mimic3_path, table, output_path,
                                              items_to_keep=None, subjects_to_keep=None):
    #header for the events.csv file
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    nb_rows_dict = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
    nb_rows = nb_rows_dict[table.lower()]

    for row, row_no, _ in tqdm(read_events_table_by_row(mimic3_path, table), total=nb_rows,
                                                        desc='Processing {} table'.format(table)):

        if (subjects_to_keep is not None) and (row['SUBJECT_ID'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None) and (row['ITEMID'] not in items_to_keep):
            continue

        row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                   'HADM_ID': row['HADM_ID'],
                   'ICUSTAY_ID': '' if 'ICUSTAY_ID' not in row else row['ICUSTAY_ID'],
                   'CHARTTIME': row['CHARTTIME'],
                   'ITEMID': row['ITEMID'],
                   'VALUE': row['VALUE'],
                   'VALUEUOM': row['VALUEUOM']}
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['SUBJECT_ID']

    if data_stats.curr_subject_id != '':
        write_current_observations()
