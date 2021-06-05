

import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

#gets the thresholds to filter outliers, return a list of touples(lower threshold, upper threshold), indexed by their column index
def get_outlier_thresholds(patients_path, fileendswith):
    data_X = []
    for root, dirs, files in tqdm(os.walk(patients_path), desc='reading'):
        for file_name in files:
            if(file_name.startswith('episode') & file_name.endswith(fileendswith)):
                episode = pd.read_csv(os.path.join(root, file_name))
                data_X += episode.values.tolist()

    rows = [i for i in zip(*data_X)]

    values_summary = pd.DataFrame(rows)
    values_summary.to_csv(os.path.join(patients_path, 'result\\', 'values_summary(no_filtering)' + fileendswith), index = False)

    #NOTE: some rows does not make sense to calculate outliers on, so we remove them, rows 4-6 are not numerical, row 1 has no outliers
    #delete 'Glasgow coma scale verbal response'
    del rows[6]
    #delte 'Glasgow coma scale motor response'
    del rows[5]
    #delete'Glasgow coma scale eye opening'
    del rows[4]
    #delete 'Capillary refill rate'
    del rows[1]  
    #delete 'hours'
    del rows[0]

    rows = [np.array(i)[~np.isnan(np.array(i))] for i in rows]

    thresholds = []
    for row in rows:
        percentile_25 = np.percentile(row, 25)
        percentile_75 = np.percentile(row, 75)
        iqr = percentile_75 - percentile_25
        #NOTE at the moment we set this factor to 10 to only remove the worst outliers, could perhaps be tweed to perform better
        cutoff = iqr*10
        outlier_cutoff_lower = percentile_25 - cutoff
        outlier_cutoff_upper = percentile_75 + cutoff
        thresholds += [(outlier_cutoff_lower, outlier_cutoff_upper)]
    
    #set 'Capillary refill rate' manually
    thresholds.insert(0, (0,1))
    #set 'Glasgow coma scale eye opening' manually
    thresholds.insert(3, (0,4))
    #set 'Glasgow coma scale motor response' manually
    thresholds.insert(4, (1,6))
    #set 'Glasgow coma scale verbal response' manually
    thresholds.insert(5, (1,5))

    #write the thresholds to a csv to be used for later
    outlier_thresholds = pd.DataFrame(thresholds)
    outlier_thresholds.to_csv(os.path.join(patients_path, 'result\\', 'outlier_thresholds' + fileendswith), index = False)

    return(thresholds)


#for a give time series removes outliers according to the thresholds from the given list of the form [(lower_threshold, upper_threshold),...]
def remove_outliers(episode_48h, thresholds):
    timeseries = episode_48h
    i = 0
    for (outlier_cutoff_lower, outlier_cutoff_upper) in thresholds:
        filtered = []
        for x in episode_48h.iloc[:,i+1].tolist():
            if np.isnan(x):
                filtered += [np.nan]
            elif (outlier_cutoff_lower <= x <= outlier_cutoff_upper):
                filtered += [x]
            else:
                filtered += [np.nan]
                #print('removed: ' + str(i) + ' ' + str(x))

        timeseries.iloc[:,i+1] = filtered
        i +=1
    return(timeseries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories.')
    parser.add_argument('-half_hour', action='store_true', help='Set this if you want to calculate the outliers with half hours interval.')
    args = parser.parse_args()

    subjects_root_path = args.subjects_root_path
    
    if args.half_hour:
        fileendswith = '_half_hour.csv'
    else:
        fileendswith = '.csv'

    outlier_thresholds = get_outlier_thresholds(subjects_root_path,fileendswith)



if __name__ == "__main__":
    main()