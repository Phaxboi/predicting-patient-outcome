# predicting-patient-outcome


This program is run in many small steps in order to have to have some control of what happens in between functions.
Modules need to be run in this order to work, can be re-run to re-generate to files in any order however. 

*****EXTRACTING SUBJECTS THAT FULFILL OUR CRITERIAS*****

python -m extract_subjects --mimic_path {path to mimic files} --output_path {desired output path} -generate_small_subset (set this to only generate a small subset (5%) of esubjects)

This command will generate the output folder with a 'stays.csv' file which is a summary of all patients and their stays info.
Only keeps patients with stays >48h and are older than 18. 
It will then generate one folder per patient, named using their subject id. There a 'patient_info_summary.csv' file will be
generated with the details of that specific patients stays. Will also generate a "result" folder where most output data that is not patients specific is put.


*****DOWNSIZE CHARTEVENTS FILE*****

python -m filter_chartevents --subjects_root_path {root directory for generated subject folders} --mimic_path {path to mimic files}

This command will significantly downsize the chartevents.csv file from 29Gb to 17GB. This will greatly speed up the following tasks.
***NOTE:*** After it has been run it will create a 'chartevents_filtered.csv' at 'mimic_path\ which you then need to rename to 'chartevents.csv' if you want to use it. 


*****CREATE AN EVENT FILE PER SUBJECTS*****

python -m create_events --subjects_root_path {root directory for generated subject folders} --mimic_path {path to mimic files}

This command will extract an event file for each patient, and write the file to respective patients folder.


*****CREATE EPISODES FILES PER SUBJECTS*****

python -m create_time_series --subjects_root_path {root directory for generated subject folders} -half_hour (if you want to round all events to the nearest half-hour)

This command will create all episodes files for each patient, generate a time series file for each episode and write them as seperate
files to respective patients folder.

*****REMOVE OUTLIERS*****

python -m get_outlier_thresholds --subjects_root_path {root directory for generated subject folders} -half_hour (Set if you want to calculate the outliers with half hours interval.)

Will iterate through all episode files and summarise all values, these will be saved as 'subjects_root_path\result\values_summary(no_filtering).csv'.
Using these values thresholds for filtering outliers wiill then be calculated and saved as 'subjects_root_path\result\outlier_thresholds.csv'.
Note that this only generates these files, the outlier filtering itself is done in the 'generate_48h_episodes' module. 


*****EXTRACT 48H EPISODES AND GENERAL CLEANUP OF THE DATA*****

python -m generate_48h_episodes --subjects_root_path {root directory for generated subject folders} -half_hour (Set if you want to extract 48h on half hours.)

Since the time series data for each episode will have different formats for all patients (measurements not done at the same time
for all patients) we need to do some more work on it before we can generate feature vectors. This mostly consist of imputing
missing values. This module will generate an 'episodeX_timeseries_48h.csv' for each episode, it consists of values every 30 minutes
of the first 48h for each patient. Outliers wll be removed according to the 'subjects_root_path\results\outlier_thresholds.csv' file.


*****CATEGORY*****
python -m category --subjects_root_path {root directory for generated subject folders} -half_hour (Set if you run extract 48h on half hours.)

Convert extract 48h episodes numerical values to discrete values (categories). Generates two files, one with both text information and the numerical value for the category and the second file with only the numerical value for the category, the second file will be used when generating features and creating a simple in-hospital prediction model.


*****GENERATE FEATURES AND CREATE A SIMPLE IN-HOSPITAL PREDICTION MODEL*****

python -m generate_in_hospital_mortality_model --subjects_root_path {root directory for generated subject folders} -use_generated_features_file (set if you want to use the previously generated features) -categorical {Set this if you want to run the categorical model instead of the numerical}

This will generate features and create a prediction model for the in-hospital prediction task. Can be either using our numerical apporach or categorical approach.
Will write the features and outcomes to results\features_X.csv and results\outcomes_X.csv, where X is either 'numerical' or 'categorical' depending on the -categorical flag.
After this the model can then be trained again with different parameters by setting the "-use_generated_features_file" flag. 




*****HOW TO GENERATE WORD EMBEDDINGS, FILES LOCATED IN 'word_embeddings'*****


**download articles from PubMed**

python -m download-data-pubtator-search --search {search term, e.g. "cardiac failure"} --file_name {name of output file, should preferably be search term with underscores} --limit {number of articles to fetch from pubmed}

This will fetch articles from PubMed according to the search term, will output two files to the 'data' directory, a list of all articles downloaded and an associated list of article IDs


**preprocess documents and turn into corpus**

python -m preprocessing --target_file_name {'what file in "data/" to preporcess'}

This will run gensims "simple_preprocess" on each line in the targeted file. Will then output a corpus file to the "corpus" map.
NOTE: target_file_name should NOT include path, only file name in the "data" map.


**generate word2vec model**

python -m word2vec.py --target_file_name {what corpus file to preprocess} --output_name {name of the output model file}

This will create a word2vec model based on the corpus from the "target_file_name", a model will be saved in the "model" map.
NOTE: target_file_name should NOT include path, only file name in the "data" map.