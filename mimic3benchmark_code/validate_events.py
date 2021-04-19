#this module aims to fix some issues such as missing ICU stay IDs and remove events with missing information 

import os
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    
    n_events = 0                    #total number of events
    empty_hadm = 0                  #hadm_id is empty in events.cvs, such events are excluded
    no_hadm_in_stay = 0             #hadm_id does not appear in stays.csv, such events are excluded
    no_icustay = 0                  #icustay_id is empty in events.csv, such events are attempted to be fixed
    recovered = 0                   #empty icustay_ids recovered according to stays.csv(given hadm_id)
    could_not_recover = 0           #empty icustay_ids not recovered according, should hopefully be zero
    icustay_missing_in_stays = 0    #icustay_ids does not appear in stays.csv, such events are excluded

    parser = argparse.ArgumentParser()
    parser.add_argument('subjects_root_path', type=str, help='Directory of subjects subdirectories')
    args = parser.parse_args()

    subdirectories = os.listdir(args.subjects_root_path)
    subjects = list(filter(str.isdigit, subdirectories))

    
    for subject in tqdm(subjects, desc='Iterating over subjects'):
        stays_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'stays.csv'), index_col=False,
                                dtype={'hadm_id':str, 'icustay_id':str})
        #converts column labels to uppercase
        #NOTE: chekc if this is needed when we have data to test with
       # stays_df.columns = stays_df.columns.str.upper()

        #assert that there is no row with empty icustay_id or hadm_id
        assert(not stays_df['hadm_id'].isnull().any())
        assert(not stays_df['icustay_id'].isnull().any())

        #assert there are no repetitions of icustay_id or hadm_id
        #since admission with multiple ICU stays should've been excluded
        assert(len(stays_df['hadm_id']).unique() == len(stays_df['hadm_id']))
        assert(len(stays_df['icustay_id']).unique() == len(stays_df['icustay_id']))


        events_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index_col=False,
                                                dtype={'hadm_id':str, 'icustay_id':str})
        #shouldnot be needed
        #events_df.columns = events_df.columns.str.upper()
        n_events += events_df.shape[0]

        #calculate empty hadm_id fields
        empty_hadm += events_df['hadm_id'].isnull().sum()
        #drop all events where no hadm_id is present
        events_df = events_df.dropna(subset=['hadm_id'])

        merged_df = events_df.merge(stays_df, left_on=['hadm_id'], right_on=['hadm_id'], how='left',
                                    suffixes=['', '_r'], indicator=True)

        #drop all events for which HADM_ID is not listed in stays.csv
        #since there is no way to know the targets of that stay (for example mortality)
        no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
        merged_df = merged_df[merged_df['_merge'] == 'both']

        #if icustay_id is empty in stays.csv, we try to recover it
        #events where it could not be recovered will be excluded
        cur_no_icustay = merged_df['icustay_id'].isnull().sum()
        no_icustay += cur_no_icustay
        merged_df.loc[:, 'icustay_id'] = merged_df['icustay_id'].fillna(merged_df['icustay_id_r'])
        recovered += cur_no_icustay - merged_df['icustay_id'].isnull().sum()
        could_not_recover += merged_df['icustay_id'].isnull().sum()
        merged_df = merged_df.dropna(subset=['icustay_id'])

        to_write = merged_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']]
        to_write.to_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index=False)

    assert(could_not_recover == 0)
    print('n_events: {}'.format(n_events))
    print('empty_hadm: {}'.format(empty_hadm))
    print('no_hadm_in_stay: {}'.format(no_hadm_in_stay))
    print('no_icustay: {}'.format(no_icustay))
    print('recovered: {}'.format(recovered))
    print('could_not_recover: {}'.format(could_not_recover))
    print('icustay_missing_in_stays: {}'.format(icustay_missing_in_stays))


if __name__ == "__main__":
    main()



