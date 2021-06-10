#this will create 48h timeseries CSVs for all timeseries in the given folder
#will create new files

import argparse
import os
import numpy as np
import pandas as pd


from prepare_features import read_timeseries
#from prepare_features import plotEpisode

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
parser.add_argument('-half_hour', action='store_true', help='Set this if you want to generate time series 48h with half hours interval.')
args = parser.parse_args()

subjects_root_path = args.subjects_root_path
if args.half_hour:
    fileendswith = '_half_hour.csv'
else:
    fileendswith = '.csv'

#read all episodes, transtale text data into numerical values and extract only first 48h
episodes = read_timeseries(subjects_root_path, fileendswith)

#plot
#ep = plotEpisode(subjects_root_path, fileendswith)







