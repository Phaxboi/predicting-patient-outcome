#various functions to interact with the data of a specific subject, given the path to its folder

import numpy as np
import os
import pandas as pd


#given a patients folder path, returns the content of its 'stays' file as a dataframe
def read_stays(subject_path):
    stays = pd.read_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.intime = pd.to_datetime(stays.intime)
    stays.outtime = pd.to_datetime(stays.outtime)
    stays.dob = pd.to_datetime(stays.dob)
    stays.dod = pd.to_datetime(stays.dod)
    stays.deathtime = pd.to_datetime(stays.deathtime)
    stays.sort_values(by=['intime', 'outtime'], inplace=True)
    return stays


#given a patients folder path, returns the content of its 'diagnosis' file as a dataframe
def read_diagnoses(subject_path):
    return pd.read_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)


#given a patients folder path, returns the content of its 'events' file as a dataframe
def read_events(subject_path, remove_null=True):
    events = pd.read_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events[events.value.notnull()]
    events.charttime = pd.to_datetime(events.charttime)
    events.hadm_id = events.hadm_id.fillna(value=-1).astype(int)
    events.icustay_id = events.icustay_id.fillna(value=-1).astype(int)
    events.valueuom = events.vallueuom.fillna('').astype(str)
    # events.sort_values(by=['CHARTTIME', 'ITEMID', 'ICUSTAY_ID'], inplace=True)
    return events

#given an ID for an icu stay, returns the events for that particular stay
def get_events_for_stay(events, icustayid, intime=None, outtime=None):
    idx = (events.icustay_id == icustayid)
    if intime is not None and outtime is not None:
        idx = idx | ((events.charttime >= intime) & (events.charttime <= outtime))
    events = events[idx]
    del events['icustay_id']
    return events

#NOTE: not completely sure what this does, think it re-arranges the data as a "time series" with events
#in cronological order, will have to look into this later
def convert_events_to_timeseries(events, variable_column='variable', variables=[]):
    metadata = events[['charttime', 'icustay_id']].sort_values(by=['charttime', 'icustay_id'])\
                    .drop_duplicates(keep='first').set_index('charttime')
    timeseries = events[['charttime', variable_column, 'value']]\
                    .sort_values(by=['charttime', variable_column, 'value'], axis=0)\
                    .drop_duplicates(subset=['charttime', variable_column], keep='last')
    timeseries = timeseries.pivot(index='charttime', columns=variable_column, values='value')\
                    .merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries

#adds elapsed hours as a column to the events dataframe
#NOTE:also seems to delete charttime, not sure why
def add_hours_elpased_to_events(events, dt, remove_charttime=True):
    events = events.copy()
    events['hours'] = (events.charttime - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    if remove_charttime:
        del events['charttime']
    return events

#NOTE: this is also confusing, not exactly sure what it does
def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan


