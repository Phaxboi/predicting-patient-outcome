#used to split data by patients, each patient could include multiple ICU stays
#each patient will have its own serparete folder 

<<<<<<< HEAD
from mimic4csv import *




#Read patiens table
patients = read_patients_table(args.mimic4_path)
admits = read_admissions_table(args.mimic4_path)
=======

import argparse
import os

from mimic4csv import *

#parsing the command to split the data and describing its parameters
parser = argparse.ArgumentParser(description='Extract per subject data from the MIMIC-IV dataset')
parser.add_argument('mimic4_path', type=str, help='Directory containing all MIMIC-IV CSV files.')
parser.add_argument('output_path', type=str, help='Directory to write the per-subject files to.')
args = parser.parse_args()


#create output directory
try:
    os.makedirs(args.output_path)
except:
    pass

patients = read_patients_table(args.mimic4_path)
admits = read_admissions_table(args.mimic4_path)
>>>>>>> cd0cef95b02eb342b044f06da93d9536b1b7bb1b
