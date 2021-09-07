#!/usr/bin/env python3

import argparse
import requests
import json
from time import sleep, time


pubmed_rate_limit_hz = 3 # without (free) api key, only 3/s allowed

parser = argparse.ArgumentParser()
parser.add_argument('search', nargs='+', help='search term, e.g. "cardiac failure"')
parser.add_argument('--limit', default=1000, type=int, help='number or articles to fetch from pubmed')
parser.add_argument('--stride', default=1, type=int, help='sample every N article')
options = parser.parse_args()


term = ' '.join(options.search).replace(' ', '+')
articles = []
idx = 0
while idx < options.limit:
    limit = 1000 if options.limit>=1000 else options.limit
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retstart=%i&retmax=%s&sort=date&term=%s' % (idx, limit, term)
    print(url, end='\r')
    t0 = time()
    info = requests.get(url).json()
    dt = time() - t0 - 1/pubmed_rate_limit_hz + 0.01 # stay within request frequency
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

name = term.replace('"','')
name = '-'.join(name.split('+')[:5])
w = open('data/%s.json' % name, 'w')
chunk_cnt = 90
saved_cnt = 0
dict_art = {}
for i in range(0, len(articles), chunk_cnt):
    article_term = ','.join(articles[i:i+chunk_cnt])
    url = 'https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?pmids=%s' % article_term
    print('\r'+article_term[:60]+'... ', end='')
    r = requests.get(url)
    assert r.status_code == 200
    for line in r.text.splitlines():
        line = json.loads(line)
        dict_art.update({line["id"]:line})
        #print(','+line, file=w) # prefix with comma for ES loader
        saved_cnt += 1
json.dump(dict_art, w, indent=2)
print()
print('saved pubtator articles:', saved_cnt)
