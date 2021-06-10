#this file reads the stays_summary file and for each patient will generate a summary of all their events (chartevents) 

import argparse
import os
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from mimic4csv import *

def create_event(patient_id):
    try:
        #read the summary file for each patient
        patient_summary = pd.read_csv(os.path.join(args.subjects_root_path, str(patient_id), 'patient_info_summary.csv'), usecols=['stay_id', 'intime', 'outtime'], dtype={'stay_id':int, 'intime':object, 'outtime':object})
    except:
        sys.stderr.write('Error, when trying to read the following: {}\n'.format(patient_id))
        return()

    #create dictionary to map a stay_id to an intime
    stay_ids = patient_summary.stay_id.tolist()
    key_list = stay_ids
    value_list = patient_summary.intime.tolist()
    dictionary = dict(zip(key_list, value_list))
    

    #create a event table for each patient.
    chart_ids = chartevents['stay_id'].isin(stay_ids)
    events = chartevents[chart_ids]
    
    stay_ids = events.stay_id.tolist()
    events['intime'] = np.vectorize(dictionary.get)(np.array(stay_ids))

    events = events[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'intime', 'charttime', 'value']]

    #save event to csv file
    events.to_csv(os.path.join(args.subjects_root_path, str(patient_id), 'events.csv'), index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
    parser.add_argument('--mimic_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
    args = parser.parse_args()

    # reading tables
    chartevents = pd.read_csv(os.path.join(args.mimic_path, 'icu', 'chartevents.csv'), usecols=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value'], dtype={'subject_id':int, 'hadm_id':int, 'stay_id':int, 'charttime':object, 'itemid': int, 'value':object, 'valuenum':float, 'valueuom':object, 'warning':int})
    
    #read the summary file and extract a list of all subject_ids that are elevant
    stays_summary = pd.read_csv(os.path.join(args.subjects_root_path, 'stays.csv'), usecols=['subject_id'], dtype={'subject_id':int})
    ids_summary = pd.unique(stays_summary.subject_id).tolist()

    for patient_id in tqdm(ids_summary, desc='Iterating over subjects'):
        create_event(patient_id)
