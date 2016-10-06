from model import *
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.wsd import lesk
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

        # scale the feature vectors
        scaler = MaxAbsScaler()
        vectors = scaler.fit(vectors)
        self.km, self.num_clusters = kmean_fit(self.word, self.pos,vectors)

    def save(self):
        # save km model and vectorizer
        with open('meaning_%s.pkl'.format(word), 'wb') as f:
            pickle.dump(self, f)


    # @static
    # def load(word):
    #     # create object and load pickle
    #     with open('meaning_%s.pkl'.format(word)) as f:
    #         meaning = pickle.load(f)
    #     return meaning

    def predict(self, sentence):
        # return meaning
        if self.word in sentence:
            v = self.vectorizer.transform([sentence])
            label =  self.km.predict(v)[0]
            return '\"' + self.word +'\"' + ": " + labels_to_defs(self.km, self.word, self.pos)[label]
        else:
            return 'The word '+ self.word + ' is not in the sentence!!!!'



if __name__ == '__main__':
    obj = Meaning('lie', 'v')
    obj.train()
    sentence = 'I always lie and never tell the truth'
    print obj.predict(sentence)
    print lesk(sentence.split(), 'lie','v').definition()
