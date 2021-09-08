import argparse
import json

import gensim

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="name of json file with pubmed articles to be used")
args = parser.parse_args()


json_file = open(args.file)
data = json.load(json_file)
all_texts = []
file = open("word_embeddings/corpus.txt", "w")

for article_id in data:
    text = data[article_id]["text"]
    #text = text.l
    file.writelines(text + "\n")
    #all_texts += [text + "\n"]


#map(lambda x : file.writelines(x + "\n"), all_words)
#lines = [x + "\n" for x in all_texts]

file.close()





