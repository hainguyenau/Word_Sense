import numpy as np
import pandas as pd
import re
import os
import gensim.utils
import mpld3
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

def load_data(filename):
    with open('data/{}'.format(filename)) as f:
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

def Tfidf(documents):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3)).fit(documents)
    vectors = vectorizer.transform(documents)
    return vectors, vectorizer
