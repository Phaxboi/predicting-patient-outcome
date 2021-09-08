import json
import argparse
from os import write


parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="name of json file to extract text from")
args = parser.parse_args()



json_file = open(args.file, encoding="utf16")
data = json.load(json_file)
dict_art = {}

for article_id in data:

    #id = data[article]["id"]
    passages = data[article_id]["passages"]
    title = passages[0]["text"]
    text = passages[1]["text"]

    dict_art.update({article_id:{"title":title, "text":text}})

#save only Atricle_id, title and text for each item
with open("word_embeddings/data/data_parsed.json", "w") as write_file:
    json.dump(dict_art, write_file, indent=2, ensure_ascii=False)

#save the list of all words
corpus_file = open("word_embeddings/data/words.txt","w")

