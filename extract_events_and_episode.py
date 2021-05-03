import argparse
import os
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from mimic4csv import *

parser = argparse.ArgumentParser()
parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
parser.add_argument('--data_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
parser.add_argument('--map_path', type=str, help='Directory to item_id to variable map.')
args = parser.parse_args()
print(args)

maps = pd.read_csv(os.path.join(args.map_path, 'itemid_to_variable_map.csv'), index_col=None).fillna('')
maps['itemid'] = pd.to_numeric(maps['itemid'])
maps = maps.set_index('itemid')

variable_map = maps.variable_name.unique()

for subject in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
    subject_path = os.path.join(args.subjects_root_path, subject)
    try:
        subject_id = int(subject)
        if not os.path.isdir(subject_path):
            raise Exception
    except:
        continue
    
    try:
        # reading tables of this subject
        stays = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'patient_info_summary.csv'), index_col=False)
        chartevents = pd.read_csv(os.path.join(args.data_path, 'icu', 'chartevents.csv'))
        item_id = pd.read_csv(os.path.join(args.data_path, 'icu', 'd_items.csv'))
    except:
        sys.stderr.write('Error reading from disk for subject: {}\n'.format(subject))
        continue
     
    # Create a event table for each patient.
    events = merge_stays_chartevents(stays, chartevents)

    events.to_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index=False)

    # Check for valid events for this subject
    if events.shape[0] == 0:
        continue

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
    for i in range(stays.shape[0]):
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        # Get all episodes
        episode = get_episode(timeseries, stay_id, intime, outtime)

        # Change time to hours, count from intime = 0 
        episode = intime_to_hours(episode, intime).set_index('hours').sort_index(axis=0)

        # All events that happen in the same half-hour merge into one row
        episode = merge_same_hour_to_one_row(episode)

        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x =="hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject, 'episode{}_'.format(i+1) + str(stay_id) + '_timeseries.csv'), index_label='hours')