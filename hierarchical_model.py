import numpy as np
import xml.etree.ElementTree as ET
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer
import pickle

# Read 1 file in train data
def read_train_data(word, pos):
    folder = None
    if pos == 'n':
        folder = 'nouns/'
    elif pos == 'v':
        folder = 'verbs/'
    path = 'training_data/' + folder+ word +'.'+ pos + '.xml'
    tree = ET.parse(path)
    root = tree.getroot()
    documents = list(elem.text for elem in list(root))
    # use a subset of training file for debugging. Will change back to full documents later.
    return documents[:1000]


# Get number of meanings of a word (helper)
def get_num_meanings(word, POS):
    length = 0
    for i in range(len(wordnet.synsets(word))):
        w = wordnet.synsets(word)[i]
        if w.pos() == POS and w.name().split('.')[0] == word:
            length += 1
    return length


# Get definitions of word (helper)
def get_def(word, POS):
    definitions = []
    for w in wordnet.synsets(word):
        if w.name().split('.')[0] == word and w.pos() == POS:
            definitions.append(w.definition())
    return definitions

# Tfidf Vectorizer documents
def vectorize(documents):
    vectorizer = TfidfVectorizer(stop_words='english').fit(documents)
    return vectorizer

# Transform vectorizer
def transform(vectorizer, documents):
    vectors = vectorizer.transform(documents)
    return vectors

# Fit KMeans model, return both km model and number of clusters
def model_fit(word, pos, vectors):
    hr = AgglomerativeClustering(n_clusters = get_num_meanings(word, pos), affinity = 'cosine', linkage = 'complete')
    hr.fit(vectors.toarray())
    return hr, get_num_meanings(word, pos)

# Get labels
def predict_labels(km):
    return km.labels_

# Get the list of definitions by order of labels range(0-n_cluster)
def labels_to_defs(km, word, pos):
    defs = []
    count = np.bincount(km.labels_)
    index =np.argsort(count)[::-1]
    for i in index:
         defs.append(get_def(word, pos)[i])
    return defs