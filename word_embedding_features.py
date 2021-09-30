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
    file = open("num_to_category.json")
    num_to_category = json.load(file)

    for root, dirs, files in tqdm(os.walk(subjects_root_path), desc='reading through folders'):
        for file_name in files:
            if (file_name.startswith('episode') & file_name.endswith('timeseries_48h.csv')):
                episode = pd.read_csv(os.path.join(root, file_name))

                if file_name == 'episode1_35962072_timeseries_48h.csv':
                    print('fak')
                subj_id = re.search('.*_(\d*)_.*', file_name).group(1)

                #for each episode, get its word vector
                word_vector = get_word_embedding_single_episode(episode, wv, num_to_category)

                word_embedding_features += [(subj_id, word_vector)]

    #NOTE: insert code to save new features as a seperate file, where each line is a feature
    wv_feature_file = open(subjects_root_path + "/result/wv_feature_file.csv", "w", encoding="utf16")
    for (subj_id,vector) in word_embedding_features:
        wv_feature_file.writelines(str(subj_id) + ',' + str(vector) + "\n")

    wv_feature_file.close()


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
        if(len(categories_text) == 0):
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