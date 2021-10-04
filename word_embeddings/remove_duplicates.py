import argparse
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument('--file_id', nargs='+', help='files with ids')
parser.add_argument('--file_art', nargs='+', help='files with article text')
parser.add_argument('--file_name', help='name of output files')
options = parser.parse_args()

unique_dict = {}
for i in range(len(options.file_id)):
    art = open('word_embeddings/data/%s' % options.file_art[i], 'r', encoding="utf16")
    id = open('word_embeddings/data/%s' % options.file_id[i], 'r')
    arts = [x.strip() for x in art]
    ids = [x.strip() for x in id]

    dic = dict(zip(ids, arts))
    unique_dict.update(dic)

ids = [id for id, art in unique_dict.items()]
arts = [art for id, art in unique_dict.items()]

file_ids = open('word_embeddings/data/%s_article_ids.txt' % options.file_name, 'w',encoding="utf16")
for ele in ids:
    file_ids.write(ele + "\n")
file_ids.close()
file_arts = open('word_embeddings/data/%s_article.txt' % options.file_name, 'w',encoding="utf16")
for ele in arts:
    file_arts.write(ele + "\n")
file_arts.close()
