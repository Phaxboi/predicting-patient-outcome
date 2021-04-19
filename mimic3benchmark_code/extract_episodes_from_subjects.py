#This breaks up the per-subject data into seperate episodes(ICU stay)
#Will store time series of events in 

import argparse
import os
import sys
from tqdm import tqdm


from subject import read_stays, read_diagnoses, read_events, get_events_for_stay,\
    add_hours_elpased_to_events
from subject import convert_events_to_timeseries, get_first_valid_from_timeseries