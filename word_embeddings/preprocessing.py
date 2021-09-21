import argparse
import json
from smart_open import open
import gensim
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--target_file_name', help='what file in "data/" to preporcess')
options = parser.parse_args()

doc = open("word_embeddings/data/" + options.target_file_name, encoding="utf16")
lines = doc.readlines()

processed = [gensim.utils.simple_preprocess(line) for line in lines]

# # remove words that appear only twice
# frequency = defaultdict(int)
# for text in processed:
#     for token in text:
#         frequency[token] += 1

# processed = [[token for token in text if frequency[token] > 2]for text in processed]


# # Create a set of frequent words
# stoplist = set('for a of the and to in'.split(' '))
# # filter out stopwords
# processed = [[word for word in document if word not in stoplist] for document in processed]


#open file to write processed corpus to
corpus_file = open("word_embeddings/corpus/" + options.target_file_name, "w", encoding="utf16")
for article in processed:
    corpus_file.writelines(" ".join(article) + "\n")

corpus_file.close()









