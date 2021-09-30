import argparse
import json
from smart_open import open
import gensim
from collections import defaultdict
import time
from gensim.models import Phrases


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
        for line in open("word_embeddings/corpus/" + self.fname + ".txt", encoding="utf16"):
            yield line.split()
 
sentences = MySentences(options.target_file_name) # a memory-friendly iterator


start = time.time()
model = gensim.models.Word2Vec(sentences,vector_size=100, min_count=5, workers=4, window=10, epochs=20)
end = time.time()
print("time to train model: " + str((end - start)))

model.save("word_embeddings/models/" + options.output_name + "_model")




##Code to generate bi/tri-grams
# bigram = Phrases(sentences, min_count=5)
# trigram = Phrases(bigram[sentences], min_count=5)
# print(len(model.wv.index_to_key))
# print(len(bigram.vocab))
# print(len(trigram.vocab))


# start = time.time()
# model = gensim.models.Word2Vec(trigram[sentences],vector_size=100, min_count=5, workers=4)
# end = time.time()
# print("time to train trigram model: " + str((end - start)))
# print(len(model.wv.index_to_key))

# model.save("models/" + options.output_name + "trigram_model")
