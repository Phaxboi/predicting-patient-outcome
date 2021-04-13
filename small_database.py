import numpy as np
import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser(description='Extract per subject data from the MIMIC-IV dataset')
parser.add_argument('mimic4_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
parser.add_argument('output_path', type=str, help='Directory to write the small set of MIMIC-IV CSV files.')
args = parser.parse_args()

try:
    os.makedirs(args.output_path)
except:
    pass

patients = pd.read_csv((os.path.join(args.mimic4_path, 'core', 'patients.csv')))
pat_idx = np.random.choice(patients.shape[0], size=1)
pat_idx = pd.array(pat_idx, dtype=string)
patients = patients.iloc[pat_idx]

# admissions = pd.read_csv((os.path.join(args.mimic4_path, 'core', 'admissions.csv')))
# admissions = admissions.iloc[pat_idx]

# transfers = pd.read_csv((os.path.join(args.mimic4_path, 'core', 'transfers.csv')))
# transfers = transfers.iloc[pat_idx]

# diagnoses_icd = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'diagnoses_icd.csv')))
# diagnoses_icd = diagnoses_icd.iloc[pat_idx]

# drgcodes = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'drgcodes.csv')))
# drgcodes = drgcodes.iloc[pat_idx]

# emar = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'emar.csv')))
# emar = emar.iloc[pat_idx]

# emar_detail = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'emar_detail.csv')))
# emar_detail = emar_detail.iloc[pat_idx]

# hcpcsevents = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'hcpcsevents.csv')))
# hcpcsevents = hcpcsevents.iloc[pat_idx]

#Kolla om denna fungerar, det ligger i 2a kolumnen! 
#labevents = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'labevents.csv')))
#labevents = labevents.iloc[pat_idx]

#Kolla om denna fungerar, det ligger i 2a kolumnen! 
microbiologyevents = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'microbiologyevents.csv')))
#microbiologyevents.set_index('subject_id', inplace=True)
#microbiologyevents = microbiologyevents.iloc[pat_idx]
microbiologyevents = microbiologyevents[microbiologyevents.subject_id.isin(pat_idx)]

print(microbiologyevents)

pharmacy = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'pharmacy.csv')))
pharmacy = pharmacy.iloc[pat_idx]

#Kolla om denna fungerar, det ligger i 3e kolumnen! 
poe = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'poe.csv')))
poe = poe.iloc[pat_idx]

#Kolla om denna fungerar, det ligger i 3e kolumnen! 
poe_detail = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'poe_detail.csv')))
poe_detail = poe_detail.iloc[pat_idx]

prescriptions = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'prescriptions.csv')))
prescriptions = prescriptions.iloc[pat_idx]

procedures_icd = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'procedures_icd.csv')))
procedures_icd = procedures_icd.iloc[pat_idx]

services = pd.read_csv((os.path.join(args.mimic4_path, 'hosp', 'services.csv')))
services = services.iloc[pat_idx]

chartevents = pd.read_csv((os.path.join(args.mimic4_path, 'icu', 'chartevents.csv')))
chartevents = chartevents.iloc[pat_idx]

datetimeevents = pd.read_csv((os.path.join(args.mimic4_path, 'icu', 'datetimeevents.csv')))
datetimeevents = datetimeevents.iloc[pat_idx]

icustays = pd.read_csv((os.path.join(args.mimic4_path, 'icu', 'icustays.csv')))
icustays = icustays.iloc[pat_idx]

inputevents = pd.read_csv((os.path.join(args.mimic4_path, 'icu', 'inputevents.csv')))
inputevents = inputevents.iloc[pat_idx]

outputevents = pd.read_csv((os.path.join(args.mimic4_path, 'icu', 'outputevents.csv')))
outputevents = outputevents.iloc[pat_idx]

procedureevents = pd.read_csv((os.path.join(args.mimic4_path, 'icu', 'procedureevents.csv')))
procedureevents = procedureevents.iloc[pat_idx]

patients.to_csv()
admissions.to_csv()
transfers.to_csv()

diagnoses_icd.to_csv()
drgcodes.to_csv()
emar.to_csv()
emar_detail.to_csv()
hcpcsevents.to_csv()
labevents.to_csv()
microbiologyevents.to_csv()
pharmacy.to_csv()
poe.to_csv()
poe_detail.to_csv()
prescriptions.to_csv()
procedures_icd.to_csv()
services.to_csv()

chartevents.to_csv()
datetimeevents.to_csv()
icustays.to_csv()
inputevents.to_csv()
outputevents.to_csv()
procedureevents.to_csv()