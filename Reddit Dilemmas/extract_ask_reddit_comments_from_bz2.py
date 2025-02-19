import pandas as pd
import ujson as json
import gzip
from collections import *
import lzma
import bz2
from glob import glob
from tqdm.auto import tqdm
import array
import zstandard as zstd
from zstandard import ZstdError
import datetime
import gzip
import io
from nltk.tokenize import word_tokenize
from os.path import basename
import os
import csv
import re
from bz2 import BZ2File as bzopen
import joblib
import pickle
import sys

def main():
    fname = sys.argv[1]

    out_dir = '/shared/4/projects/reddit-morals/data/'

    bn = basename(fname).replace('.bz2', '.gz')
    bn = bn.replace('.xz', '.gz')

    outfile = out_dir + '/' + bn

    subs = set([
        'AskReddit', 'AITA', 'AITAFiltered', "AITAH", 'TwoHotTakes', 'AmITheAsshole', 'AmIOverreacting',
        'moraldilemmas', 'AmITheButtface', 'amiwrong',
        ])

    subs |= set([s.lower() for s in subs])

    fmt = bz2 if fname.endswith('bz2') else lzma

    with fmt.open(fname, 'rb') as f, gzip.open(outfile, 'w') as outf:

        try:
            for line in f:
                try:
                    j = json.loads(line)
                    if 'selftext' not in j:
                        continue
                    if 'author' not in j:
                        continue
                    if 'subreddit' not in j:
                        continue

                    selftext = j['selftext']
                    if selftext == '[removed]':
                        continue
                    if selftext == '[deleted]':
                        continue

                    sub = j['subreddit']
                    if sub.lower() not in subs:
                        continue

                    outf.write(line)
                except BaseException as e:
                    print(repr(e))
                    continue
        except Exception as ze:
            print(fname, repr(ze))


if __name__ == '__main__':
    main()
