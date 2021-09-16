import argparse
import json
from smart_open import open
import gensim
from collections import defaultdict
import time


from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling

parser = argparse.ArgumentParser()
parser.add_argument('--target_file_name', help='what corpus file to preporcess')
parser.add_argument('--output_name', help='name of the output model file')
options = parser.parse_args()


class MySentences(object):
    def __init__(self, fname):
        self.fname = fname
 
    def __iter__(self):
        for line in open("corpus/" + self.fname + ".txt", encoding="utf16"):
            yield [line]
 
sentences = MySentences(options.target_file_name) # a memory-friendly iterator

# doc = open("corpus/corpus_processed.txt", encoding="utf16")
# lines = doc.readlines()
# lines = [line.split() for line in lines]word_embeddings/

start = time.time()
model = gensim.models.Word2Vec(sentences,vector_size=100, min_count=5, workers=4)
end = time.time()
print("time to train model: " + str((end - start)))

model.save("models/" + options.output_name + "_model")