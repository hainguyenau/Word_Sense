import numpy as np
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
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
    return documents

# Read 1 file in test data
def read_test_data(filename):
    tree = ET.parse('test_data/nouns/access.n.xml')
    root = tree.getroot()
    test_documents = []
    for i in xrange(len(root.getchildren())):
        if root.getchildren()[i].text and root.getchildren()[i][0].text:
            s = root.getchildren()[i].text + root.getchildren()[i][0].text
            test_documents.append(s)
        else:
            s = root.getchildren()[i][0].text
    return test_documents

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
def kmean_fit(word, pos, vectors):
    km = KMeans(n_clusters = get_num_meanings(word, pos))
    km.fit(vectors)
    return km, get_num_meanings(word, pos)

# Get labels
def predict_labels(km):
    return km.labels_

# Get the list of definitions by order of labels range(0-n_cluster)
def labels_to_defs(km):
    defs = []
    count = np.bincount(km.labels_)
    index =np.argsort(count)[::-1]
    for i in index:
         defs.append(get_def('access','n')[i])
    return defs

# Predict a word in a new setences
# def predict(km, test_str, word):
#     if word in test_str:
#         v = vectorize([test_str])
#         return km.predict(v)
#     else:
#         return None


if __name__ == '__main__':
    train_documents = read_train_data('bow','v')

    # test_documents = read_test_data('test_data/nouns/access.n.xml')
    # vectorizer = vectorize(train_documents)
    # vectors = transform(vectorizer, train_documents)
    # km, num_clusters = kmean_fit('access','n',vectors)


    # Save model and vectorizer as pickle
    # with open('my_model.pkl', 'wb') as f:
    #     pickle.dump(km, f)
    #
    # with open('my_vectorizer.pkl', 'wb') as f:
    #     pickle.dump(vectorizer, f)


















    # Prediction (need clean up and create new predict function)
    # test = 'pricing behavior could be used to determine when to remove incumbent LEC access services from price cap regulation'
    # vectorizer = TfidfVectorizer(stop_words='english').fit(documents)
    # v = vectorizer.transform([test])
    # print km.predict(v)
