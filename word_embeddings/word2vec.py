import argparse
import json

import gensim

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="name of json file with pubmed articles to be used")
args = parser.parse_args()


json_file = open(args.file)
data = json.load(json_file)
all_words = []

for article_id in data:
    text = data[article_id]["text"]
    text_split = text.split()
    all_words += text_split




