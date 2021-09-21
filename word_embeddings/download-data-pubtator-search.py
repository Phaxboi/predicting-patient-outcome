import argparse
import requests
import json
from time import sleep, time
from tqdm import tqdm


pubmed_rate_limit_hz = 3 # without (free) api key, only 3/s allowed

parser = argparse.ArgumentParser()
parser.add_argument('--search', nargs='+', help='search term, e.g. "cardiac failure"')
parser.add_argument('--limit', default=1000, type=int, help='number or articles to fetch from pubmed')
parser.add_argument('--stride', default=1, type=int, help='sample every N article')
parser.add_argument('--file_name', help='name of output file, should preferably be search term with underscores')
options = parser.parse_args()


term = ' '.join(options.search).replace(' ', '+')
term = term.replace('_', ' ')
articles = []
idx = 0
while idx < options.limit:
    limit = 1000 if options.limit>=1000 else options.limit
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retstart=%i&retmax=%s&sort=date&term=%s' % (idx, limit, term)
    print(url, end='\r')
    t0 = time()
    info = requests.get(url).json()
    #stay within request frequency
    dt = time() - t0 - 1/pubmed_rate_limit_hz + 0.01 
    if dt > 0:
        sleep(dt)
    if 'esearchresult' not in info:
        print()
        print('rate limiting...')
        sleep(2)
        continue
    article_ids = info['esearchresult']['idlist']
    articles += article_ids
    idx += len(article_ids)
    if len(article_ids) < 990:
        break
print()

articles = articles[::options.stride]

corpus = open('word_embeddings/data/%s.txt' % options.file_name, 'w', encoding="utf16")
ids = open('word_embeddings/data/%s_article_ids.txt' % options.file_name, 'w')
chunk_cnt = 90
saved_cnt = 0
dict_art = {}
for i in tqdm(range(0, len(articles), chunk_cnt), desc="Processing atricle chunks"):
    article_term = ','.join(articles[i:i+chunk_cnt])
    url = 'https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?pmids=%s' % article_term
    r = requests.get(url)
    assert r.status_code == 200
    for line in r.text.splitlines():
        line = json.loads(line)

        id = line["id"]
        ids.writelines(id + "\n")

        text = line["passages"][1]["text"]
        corpus.writelines(text + "\n")

        saved_cnt += 1
print()
corpus.close()
ids.close()
print('saved pubtator articles:', saved_cnt)

