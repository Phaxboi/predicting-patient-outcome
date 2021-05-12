import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from mimic4csv import *

pd.options.mode.chained_assignment = None 


def extract_and_split(patient_id):
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
    

    # Create a event table for each patient.
    #new read
    chart_ids = chartevents['stay_id'].isin(stay_ids)
    events = chartevents[chart_ids]
    
    stay_ids = events.stay_id.tolist()
    events['intime'] = np.vectorize(dictionary.get)(np.array(stay_ids))


    events = events[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'intime', 'charttime', 'value']]

    events.to_csv(os.path.join(args.subjects_root_path, str(patient_id), 'events.csv'), index=False)
    # Check for valid events for this subject

    if events.shape[0] == 0:
        return()
    
    # Not implemented - If charttime missing, replace with storetime
    # events = [replace with storetime if charttime missing in event] 

    # Merge Event table with item the model look at.
    events = events.merge(maps, left_on='itemid', right_index=True)

    # Convert lb -> kg
    events = fix_weight(events)

    # Convert inch -> cm 
    events = fix_height(events)

    # Convert F -> C
    events = fix_temperature(events)


    # Convert event table to a time serie
    timeseries = convert_events_timeserie(events,  variables=variable_map)


    # Extracting separate episodes
    for i in range(patient_summary.shape[0]):
        stay_id = patient_summary.stay_id.iloc[i]
        intime = patient_summary.intime.iloc[i]
        outtime = patient_summary.outtime.iloc[i]

        # Get all episodes
        episode = get_episode(timeseries, stay_id, intime, outtime)

        # Change time to hours, count from intime = 0 
        episode = intime_to_hours(episode, intime).set_index('hours').sort_index(axis=0)

        # All events that happen in the same half-hour merge into one row
        episode = merge_same_hour_to_one_row(episode)

        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x =="hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, str(patient_id), 'episode{}_'.format(i+1) + str(stay_id) + '_timeseries.csv'), index_label='hours')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
    parser.add_argument('--mimic_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
    #parser.add_argument('--map_path', type=str, help='Directory to item_id to variable map.')
    args = parser.parse_args()
    print(args)

    maps = pd.read_csv(os.path.join( 'itemid_to_variable_map.csv'), index_col=None).fillna('')
    maps['itemid'] = pd.to_numeric(maps['itemid'])
    maps = maps.set_index('itemid')

    variable_map = maps.variable_name.unique()

    # reading tables
    start_time_read_chartevents = time.time()
    
    chartevents = pd.read_csv(os.path.join(args.mimic_path, 'icu', 'chartevents.csv'), usecols=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'warning'], dtype={'subject_id':int, 'hadm_id':int, 'stay_id':int, 'charttime':object, 'itemid': int, 'value':object, 'valuenum':float, 'valueuom':object, 'warning':int})
    
    end_time_read_chartevents = time.time() - start_time_read_chartevents
    print('Time to read chartevents: ' + str(end_time_read_chartevents ))
    
    item_id = pd.read_csv(os.path.join(args.mimic_path, 'icu', 'd_items.csv'))

    #read the summary file and extract a list of all subject_ids that are elevant
    stays_summary = pd.read_csv(os.path.join(args.subjects_root_path, 'stays.csv'), usecols=['subject_id'], dtype={'subject_id':int})
    ids_summary = pd.unique(stays_summary.subject_id).tolist()
 
    #NOTE:code to filter the chartevents file, uncomment if desired
    # start_time_isin= time.time()
    # chart_ids = chartevents['subject_id'].isin(ids_summary)
    # end_time_isin = time.time() - start_time_isin
    # print('time isin: ' + str(end_time_isin ))
    # print('Number of relevant events:' + str(np.sum(chart_ids)))

    # start_time_loc= time.time()
    # chartevents_filtered = chartevents.loc[chart_ids]
    # end_time_loc = time.time() - start_time_loc
    # print('time loc: ' + str(end_time_loc ))
    # print(chartevents_filtered.subject_id.unique())
    # chartevents_filtered.to_csv(os.path.join(args.data_path, 'icu', 'chartevents_filtered.csv'), index=False)



    start_time_whole = time.time()
    for patient_id in tqdm(ids_summary, desc='Iterating over subjects'):
        extract_and_split(patient_id)

    end_time_whole = time.time() - start_time_whole
    print('time whole: ' + str(end_time_whole))
