#used to split data by patients, each patient could include multiple ICU stays
#each patient will have its own serparete folder 

from mimic4csv import *




#Read patiens table
patients = read_patients_table(args.mimic4_path)
admits = read_admissions_table(args.mimic4_path)