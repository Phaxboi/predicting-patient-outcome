import argparse
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from mimic4csv import *

def create_episode(patient_id):
    try:
        #read the summary file for each patient
        patient_summary = pd.read_csv(os.path.join(args.subjects_root_path, str(patient_id), 'patient_info_summary.csv'), usecols=['stay_id', 'intime', 'outtime'], dtype={'stay_id':int, 'intime':object, 'outtime':object})
        events = pd.read_csv(os.path.join(args.subjects_root_path, str(patient_id), 'events.csv'))
    except:
        sys.stderr.write('Error, when trying to read the following: {}\n'.format(patient_id))
        return()

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
        episode = intime_to_hours(episode, intime, half_hour).set_index('hours').sort_index(axis=0)
        if half_hour:
            # All events that happen in the same half-hour merge into one row
            episode = merge_same_hour_to_one_row(episode)

        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x =="hours" else x))
        episode = episode[columns_sorted]
        if half_hour:
            episode.to_csv(os.path.join(args.subjects_root_path, str(patient_id), 'episode{}_'.format(i+1) + str(stay_id) + '_timeseries_half_hour.csv'), index_label='hours')
        else:
            episode.to_csv(os.path.join(args.subjects_root_path, str(patient_id), 'episode{}_'.format(i+1) + str(stay_id) + '_timeseries.csv'), index_label='hours')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
    parser.add_argument('--mimic_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
    parser.add_argument('-half_hour', action='store_true', help='Set this if you want to generate time series with half hours interval.')
    args = parser.parse_args()

    half_hour = args.half_hour

    maps = pd.read_csv(os.path.join( 'itemid_to_variable_map.csv'), index_col=None).fillna('')
    maps['itemid'] = pd.to_numeric(maps['itemid'])
    maps = maps.set_index('itemid')

    variable_map = maps.variable_name.unique()

    #read the summary file and extract a list of all subject_ids that are elevant
    stays_summary = pd.read_csv(os.path.join(args.subjects_root_path, 'stays.csv'), usecols=['subject_id'], dtype={'subject_id':int})
    ids_summary = pd.unique(stays_summary.subject_id).tolist()

    for patient_id in tqdm(ids_summary, desc='Iterating over subjects'):
        create_episode(patient_id)