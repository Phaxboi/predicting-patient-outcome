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

chart = pd.read_csv(os.path.join(args.mimic4_path, 'icu', 'chartevents.csv'))
chart = chart['subject_id'].unique()
chartevents = pd.DataFrame()
chartevents['subject_id'] = chart
chartevents.to_csv(os.path.join(args.output_path, 'only_patients_with_chartevents.csv'), index=False)