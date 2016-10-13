import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

def load_data(filename):
    with open('new_train/{}'.format(filename)) as f:
        d = defaultdict(list)


        for line in f.readlines():
            m = re.search("<s>(.*)<\/s>", line)
            if m:
                m2 = re.search("<tag \"(.*)\">(.*)<\/>", line)
                if m2:
                    meaning = m2.group(1)
                    word = m2.group(2)
                    sentence = m.group(1)
                    sentence = sentence.replace('<s>', '')
                    sentence = sentence.replace('</s>', '')
                    sentence = sentence.replace('<p>', '')
                    sentence = sentence.replace('</p>', '')
                    sentence = sentence.replace('<@>', '')
                    sentence = sentence.replace(m2.group(0), '')
                    d[meaning].append(sentence)

    return d

def lem(documents):
    lem_documents = []
    for doc in documents:
        no_tag_words = [w[:-3] for w in gensim.utils.lemmatize(doc)]
        lem_documents.append(' '.join(no_tag_words))
    return lem_documents
