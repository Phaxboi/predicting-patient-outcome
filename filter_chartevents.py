#run this file to filter the charevents file to only contain events of the patients in your 'stays.csv' file

import argparse
import os
import time
import numpy as np
import pandas as pd
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
parser.add_argument('--mimic_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
args = parser.parse_args()
 
#reading original charevents file
start_time_read_chartevents = time.time()
chartevents = pd.read_csv(os.path.join(args.mimic_path, 'icu', 'chartevents.csv'), usecols=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'warning'], dtype={'subject_id':int, 'hadm_id':int, 'stay_id':int, 'charttime':object, 'itemid': int, 'value':object, 'valuenum':float, 'valueuom':object, 'warning':int})    
end_time_read_chartevents = time.time() - start_time_read_chartevents
print('Time to read chartevents: ' + str(end_time_read_chartevents))   
    

#read the summary file and extract a list of all subject_ids that are elevant
stays_summary = pd.read_csv(os.path.join(args.subjects_root_path, 'stays.csv'), usecols=['subject_id'], dtype={'subject_id':int})
ids_summary = pd.unique(stays_summary.subject_id).tolist()

#code to filter the chartevents file
start_time_isin= time.time()
chart_ids = chartevents['subject_id'].isin(ids_summary)
end_time_isin = time.time() - start_time_isin
print('time isin: ' + str(end_time_isin ))
print('Number of relevant events:' + str(np.sum(chart_ids)))

start_time_loc= time.time()
chartevents_filtered = chartevents.loc[chart_ids]
end_time_loc = time.time() - start_time_loc
print('time loc: ' + str(end_time_loc ))
chartevents_filtered.to_csv(os.path.join(args.mimic_path, 'icu', 'chartevents_filtered.csv'), index=False)