#used to generate a file per subject which includes their general information


import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from mimic4csv import *

#parsing the command to split the data and describing its parameters
parser = argparse.ArgumentParser(description='Extract per subject data from the MIMIC-IV dataset')
parser.add_argument('--mimic4_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
parser.add_argument('--output_path', type=str, help='Directory to write the per-subject files to.')
args = parser.parse_args()
mimic4_path = args.mimic4_path
output_path = args.output_path

#create output directory
try:
    os.makedirs(output_path)
except:
    pass


#start reading data from the 'core' folder
patients = read_patients_table(mimic4_path)
admissions = read_admissions_table(mimic4_path)
transfers = read_transfers_table(mimic4_path)

#read icustays table from the 'ICU' folder
icustays = read_icustays_table(mimic4_path)

#TODO: exclude cases we don't want


#merge per-patient information with per-admission information
patients_info = merge_patients_admissions(patients, admissions)

#merge per-admission data with transfers
patients_info = merge_admissions_transfers(patients_info, transfers)

#merge admission data with icu stay data
patients_info = merge_admissions_stays(patients_info, icustays)



#calcualte age of all patients at the time of their stays
patients_info = add_patient_age(patients_info)

#filter patients on age
patients_info = filter_patients_age(patients_info)


#write a csv
patients_info.to_csv(os.path.join(output_path, 'stays.csv'), index=False)

