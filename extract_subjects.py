#used to generate a file per subject which includes their general information


import argparse
import os
import numpy as np
#import matplotlib.pyplot as plt

from mimic4csv import *

#parsing the command to split the data and describing its parameters
parser = argparse.ArgumentParser(description='Extract per subject data from the MIMIC-IV dataset')
parser.add_argument('--mimic_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
parser.add_argument('--output_path', type=str, help='Directory to write the per-subject files to.')
parser.add_argument('-generate_small_subset', action='store_true', help='Set this if you only want to generate a small 5 percent subject set instead.')
args = parser.parse_args()
mimic_path = args.mimic_path
output_path = args.output_path
generate_small_subset = args.generate_small_subset

#create output directory
try:
    os.makedirs(output_path)
except:
    pass


#start reading data from the 'core' folder
patients = read_patients_table(mimic_path)
admissions = read_admissions_table(mimic_path)

#read icustays table from the 'ICU' folder
icustays = read_icustays_table(mimic_path)

#exclude cases we don't want
icustays = filter_icustays_with_transfers(icustays)

#filter icustays less than 48 hours
icustays = filter_icustays_48h(icustays)


#merge per-patient information with per-admission information
patients_info = merge_patients_admissions(patients, admissions)

#merge per-admission data with transfers
#patients_info = merge_admissions_transfers(patients_info, transfers)

#merge admission data with icu stay data
patients_info = merge_admissions_stays(patients_info, icustays)



#calcualte age of all patients at the time of their stays
patients_info = add_patient_age(patients_info)

#filter patients on age
patients_info = filter_patients_age(patients_info)

#fix patients missing the 'deathtime' value
fix_missing_deathtimes(patients_info)

#the order we want our fields to be in is manually set up by this variable
columns_title = ['subject_id', 'gender', 'age', 'birth_year', 'dod', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'stay_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los', 'hospital_expire_flag']
patients_info = rearrange_columns(patients_info, columns_title)


if generate_small_subset:
    length = len(patients_info.index)
    end_index = int(length*0.05)
    patients_info = patients_info[:end_index]

#write a csv summary of all patients
patients_info.to_csv(os.path.join(output_path, 'stays.csv'), index=False)

#break up subjects
#NOTE: will generate a folder and file for each subject which takes ages to delete, so only run
#this if you really need it
patients_info = pd.read_csv(os.path.join(output_path, 'stays.csv'))
break_up_stays_by_subject(patients_info, output_path)



