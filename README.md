# predicting-patient-outcome


This program is run in many small steps in order to have to have some control of what happens in between functions.
Modules need to be run in this order to work, can be re-run to re-generate to files in any order however. 

*****EXTRACTING SUBJECTS THAT FULFILL OUR CRITERIAS*****

python -m extract_subjects --mimic4_path {path to mimic files} --output_path {desired output path}

This command will generate the output folder with a 'stays.csv' file which is a summary of all patients and their stays info.
Only keeps patients with stays >48h. 

It will then generate one folder per patient, named using their subject id. There a 'patient_info_summary.csv' file be
generated with the details of that specific patients stays.


*****EXTRACT EVENTS AND GENERATE EPISODE FILES PER SUBJECTS*****

python -m extract_events_and_episode --subjects_root_path {root directory for generated subject folders} --data_path {path to mimic files} --map_path {directory to item_id to variable mapping}

This command will extract events for each patient, generate a time series file for each episode and write them as seperate
files to respective patients folder. The map_path is a mapping from mimics own variable names (item_id) to our own names.



*****PREPARE FEATURE DATA AND GENERAL CLEANUP OF THE DATA*****

python -m prepare_features --subjects_root_path {root directory for generated subject folders}

Since the time series data for each episode will have different formats for all patients (measurements not done at the same time
for all patients) we need to do some more work on it before we can generate fefature vectors. This mostly consist of imputing
missing values. This module will generate an 'episodeX_timeseries_48h.csv' for each episode, it consists of values every 30 minutes
of the first 48h for each patient, missing values have been imputed using mean values.


