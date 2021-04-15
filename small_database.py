import numpy as np
import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser(description='Extract per subject data from the MIMIC-IV dataset')
parser.add_argument('mimic4_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
# For separate runs
# parser.add_argument('mimic4_path_pat', type=str, help='Directory containing patiens files.')
parser.add_argument('output_path', type=str, help='Directory to write the small set of MIMIC-IV CSV files.')
args = parser.parse_args()

try:
    os.makedirs(args.output_path)
except:
    pass

#To run all
patients = pd.read_csv(os.path.join(args.mimic4_path, 'core', 'patients.csv'))
pat_idx_arr = patients['subject_id'].sample(n=100)
pat_idx = pd.DataFrame()
pat_idx['subject_id'] = pat_idx_arr

# #For separate runs
# patients = pd.read_csv(os.path.join(args.mimic4_path_pat, 'patients.csv'))
# pat_idx = pd.DataFrame()
# pat_idx['subject_id'] = patients['subject_id']

patients = patients.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(patients)
patients.to_csv(os.path.join(args.output_path, 'patients.csv'), index=False)

admissions = pd.read_csv(os.path.join(args.mimic4_path, 'core', 'admissions.csv'))
admissions = admissions.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(admissions)
admissions.to_csv(os.path.join(args.output_path, 'admissions.csv'), index=False)


transfers = pd.read_csv(os.path.join(args.mimic4_path, 'core', 'transfers.csv'))
transfers = transfers.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(transfers)
transfers.to_csv(os.path.join(args.output_path, 'transfers.csv'), index=False)

diagnoses_icd = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'diagnoses_icd.csv'))
diagnoses_icd = diagnoses_icd.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(diagnoses_icd)
diagnoses_icd.to_csv(os.path.join(args.output_path, 'diagnoses_icd.csv'), index=False)

drgcodes = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'drgcodes.csv'))
drgcodes = drgcodes.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(drgcodes)
drgcodes.to_csv(os.path.join(args.output_path, 'drgcodes.csv'), index=False)

emar = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'emar.csv'))
emar = emar.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
# print(emar)
emar.to_csv(os.path.join(args.output_path, 'emar.csv'), index=False)

emar_detail = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'emar_detail.csv'))
emar_detail = emar_detail.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
# print(emar_detail)
emar_detail.to_csv(os.path.join(args.output_path, 'emar_detail.csv'), index=False)

hcpcsevents = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'hcpcsevents.csv'))
hcpcsevents = hcpcsevents.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(hcpcsevents)
hcpcsevents.to_csv(os.path.join(args.output_path, 'hcpcsevents.csv'), index=False)

#Kolla om denna fungerar, det ligger i 2a kolumnen! 
labevents = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'labevents.csv'))
labevents = labevents.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(labevents)
labevents.to_csv(os.path.join(args.output_path, 'labevents.csv'), index=False)

#Kolla om denna fungerar, det ligger i 2a kolumnen! 
microbiologyevents = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'microbiologyevents.csv'))
microbiologyevents = microbiologyevents.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(microbiologyevents)
microbiologyevents.to_csv(os.path.join(args.output_path, 'microbiologyevents.csv'), index=False)


pharmacy = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'pharmacy.csv'))
pharmacy = pharmacy.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(pharmacy)
pharmacy.to_csv(os.path.join(args.output_path, 'pharmacy.csv'), index=False)

#Kolla om denna fungerar, det ligger i 3e kolumnen! 
poe = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'poe.csv'))
poe = poe.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(poe)
poe.to_csv(os.path.join(args.output_path, 'poe.csv'), index=False)

#Kolla om denna fungerar, det ligger i 3e kolumnen! 
poe_detail = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'poe_detail.csv'))
poe_detail = poe_detail.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(poe_detail)
poe_detail.to_csv(os.path.join(args.output_path, 'poe_detail.csv'), index=False)

prescriptions = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'prescriptions.csv'))
prescriptions = prescriptions.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(prescriptions)
prescriptions.to_csv(os.path.join(args.output_path, 'prescriptions.csv'), index=False)

procedures_icd = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'procedures_icd.csv'))
procedures_icd = procedures_icd.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(procedures_icd)
procedures_icd.to_csv(os.path.join(args.output_path, 'procedures_icd.csv'), index=False)

services = pd.read_csv(os.path.join(args.mimic4_path, 'hosp', 'services.csv'))
services = services.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(services)
services.to_csv(os.path.join(args.output_path, 'services.csv'), index=False)

# chartevents = pd.read_csv(os.path.join(args.mimic4_path, 'icu', 'chartevents.csv'))
# chartevents = chartevents.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
# #print(chartevents)
# chartevents.to_csv(os.path.join(args.output_path, 'chartevents.csv'), index=False)

datetimeevents = pd.read_csv(os.path.join(args.mimic4_path, 'icu', 'datetimeevents.csv'))
datetimeevents = datetimeevents.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(datetimeevents)
datetimeevents.to_csv(os.path.join(args.output_path, 'datetimeevents.csv'), index=False)

icustays = pd.read_csv(os.path.join(args.mimic4_path, 'icu', 'icustays.csv'))
icustays = icustays.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(icustays)
icustays.to_csv(os.path.join(args.output_path, 'icustays.csv'), index=False)

inputevents = pd.read_csv(os.path.join(args.mimic4_path, 'icu', 'inputevents.csv'))
inputevents = inputevents.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(inputevents)
inputevents.to_csv(os.path.join(args.output_path, 'inputevents.csv'), index=False)

outputevents = pd.read_csv(os.path.join(args.mimic4_path, 'icu', 'outputevents.csv'))
outputevents = outputevents.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(outputevents)
outputevents.to_csv(os.path.join(args.output_path, 'outputevents.csv'), index=False)

procedureevents = pd.read_csv(os.path.join(args.mimic4_path, 'icu', 'procedureevents.csv'))
procedureevents = procedureevents.merge(pat_idx[['subject_id']], how='inner', left_on=['subject_id'], right_on=['subject_id'])
#print(procedureevents)
procedureevents.to_csv(os.path.join(args.output_path, 'procedureevents.csv'), index=False)
