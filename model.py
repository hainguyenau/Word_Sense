import numpy as np
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


# Read 1 file in train data
def read_data(filename):
    tree = ET.parse('training_data/nouns/access.n.xml')
    root = tree.getroot()
    documents = list(elem.text for elem in root.getchildren())
    return documents


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
    # lemmatizer = WordNetLemmatizer()
    # tokenizer = RegexpTokenizer(r'\w+')

    vectorizer = TfidfVectorizer(stop_words='english').fit(documents)
    vectors = vectorizer.transform(documents)
    return vectors

# Fit KMeans model
def kmean_fit(word, pos, vectors):
    km = KMeans(n_clusters = get_num_meanings(word, pos))
    km.fit(vectors)
    return km

# Get labels
def get_labels(km):
    defs = []
    count = np.bincount(km.labels_)
    index =np.argsort(count)[::-1]
    for i in index:
         defs.append(get_def('access','n')[i])
    return defs

# Predict a word in a new setences
# def predict(km, test_str, word):
#
#
#     if word in test_str:
#         v = vectorize([test_str])
#         return km.predict(v)
#     else:
#         return None


if __name__ == '__main__':
    documents = read_data('training_data/nouns/access.n.xml')
    print get_def('lie', 'v')
    # vectors = vectorize(documents)
    # km = kmean_fit('access','n',vectors)

    # Prediction (need clean up and create new predict function)
    # test = 'pricing behavior could be used to determine when to remove incumbent LEC access services from price cap regulation'
    # vectorizer = TfidfVectorizer(stop_words='english').fit(documents)
    # v = vectorizer.transform([test])
    # print km.predict(v)
