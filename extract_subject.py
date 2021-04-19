#used to generate a file per subject which includes their general information


import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


#parsing the command to split the data and describing its parameters
parser = argparse.ArgumentParser(description='Extract per subject data from the MIMIC-IV dataset')
parser.add_argument('mimic4_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
parser.add_argument('output_path', type=str, help='Directory to write the per-subject files to.')
args = parser.parse_args()


#create output directory
try:
    os.makedirs(args.output_path)
except:
    pass


#start reading data from the 'core' folder
patients = read_patients_table(args.mimic4_path)
admissions = read_admissions_table(args.mimic4_path)
transfers = read_transfers_table(args.mimic4_path)

#read icustays table from the 'ICU' folder
icustays = read_icustays_table(args.mimic4_path)

#TODO: exclude cases we don't want


#merge per-patient information with per-admission information
patients_info = merge_patients_admissions(patients, admissions)

#merge per-admission data with transfers
patients_info = merge_admissions_transfers(patients_info, transfers)

#merge admission data with icu stay data
patients_info = merge_admissions_stays(patients_info, icustays)