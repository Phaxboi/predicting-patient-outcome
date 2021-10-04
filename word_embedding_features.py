#this files takes a dataframe of features(numerical or categorical), concatenates their
#word vector at the end and return them

import pandas as pd
import os
import numpy as np
import argparse
from gensim.models import Word2Vec
import json
from tqdm import tqdm
import re

num_features = 672


#this will be run if you only want to generate the word embedding features
def to_word_embedding_features(subjects_root_path, we_model_path):

    wv = Word2Vec.load(we_model_path).wv
    word_embedding_features = []
    ids = []
    file = open("num_to_category.json")
    num_to_category = json.load(file)

    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading through folders'):
        for file_name in files:
            if (file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))

                if file_name == 'episode1_38761888_timeseries_48h.csv':
                    print('fak')
                adm_id = re.search('.*_(\d*)_.*', file_name).group(1)

                #for each episode, get its word vector
                word_vector = get_word_embedding_single_episode(episode, wv, num_to_category)

                word_embedding_features += [word_vector]
                ids += [adm_id]

    wv_features_df = pd.DataFrame(word_embedding_features)
    wv_features_df.to_csv((subjects_root_path + '/result/wv_feature_file.csv'), index=False)

    #NOTE: insert code to save new features as a seperate file, where each line is a feature
    wv_feature_ids = open(subjects_root_path + "/result/wv_feature_ids.csv", "w", encoding="utf16")
    for adm_id in ids:
        wv_feature_ids.writelines(str(adm_id) + "\n")
    wv_feature_ids.close()





    #iterate over all items in the list, we want to process every other triplets of numbers from the feature vector
    #assign a weight to each item we process, not sure if "first 10% hypotension" should be weighted same as
    #"last 10% hypotension" 
    #for index, row in df.iterrows():
        
    return ()

#detta borde funka: gå igenom varje kolumn per timeserie, ta length av uppmätta värden (ignorera tomma värden)
#gör om alla till ord och plocka fram word vectors, vikta dessa baserat på hur många uppmätta värden som fanns
#summera vektorn 


def get_word_embedding_single_episode(episode, wv, num_to_category):
    
    column_names = episode.columns.tolist()
    column_names.remove('hours')
    column_wvs = []

    #for each column, get its word vector and concat to list
    for column_name in column_names:

        column_info = num_to_category[column_name]
        categories_text = column_info["categories_text"]
        
        #for some columns there exist no good words to look at, so we skip those
        if(column_info["use_for_wv"] == 0):
            continue

        column = episode.get(column_name)        

        categories_thresholds = column_info["categories_thresholds"]

        #remove all empty cells
        cleaned_column = [val for val in column if not np.isnan(val)]
        num_of_values = len(cleaned_column)

        #discretize the numeric values  
        categories = np.searchsorted(categories_thresholds, cleaned_column)
        
        #translate discrete numbers into text categories
        column_text = [categories_text[val] for val in categories]
        
        #get the vector representation for each word and weight them according to how many values were measured
        column_wvs += [[wv[text]/num_of_values] for text in column_text if text != "normal"]

    #return a zero vector if no abnormal values are found NOTE: not sure if this is the best to do in thiis case
    #but it only seems to occur once so it shouldnt be a problem
    if column_wvs == []:
        return ([0]*100)

    #summarize all vectors into one
    wv_summarized = ([sum(x) for x in zip(*column_wvs)])[0].tolist()

    return wv_summarized




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects_root_path', type=str, help='Directory containing subject subdirectories')
    parser.add_argument('--we_model_path', type=str, help='Specify what Word2Vec model to use')
    args = parser.parse_args()

    to_word_embedding_features(args.subjects_root_path, args.we_model_path)


main()