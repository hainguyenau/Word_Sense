from model import *
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import pickle

class Meaning:
    def __init__(self, word, pos):
        self.word = word
        self.pos = pos
        self.vectorizer = None
        self.km = None
        self.num_clusters = None

    def train(self):
        # train the meaning objec with proper train set
        train_documents = read_train_data(self.word, self.pos)
        self.vectorizer = vectorize(train_documents)
        vectors = transform(self.vectorizer, train_documents)
        self.km, self.num_clusters = kmean_fit(self.word, self.pos,vectors)

    def save():
        # save km model and vectorizer
        with open('my_km.pkl', 'wb') as f:
            pickle.dump(self.km, f)
        with open('my_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load():
        # create object and load pickle
        with open('data/my_km.pkl') as f:
            model = pickle.load(f)

        with open('data/my_vectorizer.pkl') as f:
            tfidf = pickle.load(f)

    def predict(sentence):
        # return meaning
        pass

if __name__ == '__main__':
    obj = Meaning('access', 'n')
    obj.train()
    obj.save()
