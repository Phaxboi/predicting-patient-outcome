#used to split data by patients, each patient could include multiple ICU stays
#each patient will have its own serparete folder 


import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from mimic4csv import *

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


#arbitrary CSV file
test_frame = pd.read_csv(args.mimic4_path)
#will make row 'subject_id' to index but still keep the original column(in order to access the data)
test_frame.set_index('subject_id', inplace=True, drop=False)
#sample 10 random items from column 'subject_id'
pat_idx = test_frame['subject_id'].sample(n=10)
#this now works because 'subject_id' is also the index
temp = test_frame.loc[pat_idx]

#path
#temp.to_csv(r'C:\Users\Philip Svensson\Documents\Exjobb\predicting-patient-outcome\data\data.csv')




#read data from the core/patient.csv file
patients = read_patients_table(args.mimic4_path)
pat_idx = np.random.choice(patients.shape[0], size=10)
patients = patients.iloc[pat_idx]
#read data from the core/patient.csv file
admits = read_admissions_table(args.mimic4_path)
#read data from the icu/icustays.csv file
stays = read_icustays_table(args.mimic4_path)
stays = stays.merge(patients[['subject_id']], left_on='subject_id', right_on='subject_id')

#remove admissions with transfers between different ICU units or wards
stays = remove_icustays_with_transfers(stays)
#merges stays and admits based on the patient IDs
stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)
stays = filter_admissions_on_nb_icustays(stays)

#add admityear and age to stays
stays = add_age_to_icustays(stays)

stays = add_inunit_mortality_to_icustays(stays)

########################### TEST ########################### 

#DOB test
#plt.hist(patients.dob)
#plt.show()

#Age test
#plt.hist(stays.age)
#plt.show()

