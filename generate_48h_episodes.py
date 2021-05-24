#this will create 48h timeseries CSVs for all timeseries in the given folder
#will create new files

import argparse
import os
import numpy as np
import pandas as pd
import re
import statistics

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from prepare_features import read_timeseries
from prepare_features import remove_outliers_timeseries

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
args = parser.parse_args()

subjects_root_path = args.subjects_root_path

#read all episodes, transtale text data into numerical values and extract only first 48h
episodes = read_timeseries(subjects_root_path)
#impute missing data NOTE: this is probably a bad idea, we should probably just leave this commented and not run it
#imputed_timeseries_list = impute(subjects_root_path)

#plot
#ep = plotEpisode(subjects_root_path)

#remove outliers
#rm_outliers_timeseries = remove_outliers_timeseries(subjects_root_path)

#ep = plotEpisode(subjects_root_path)







