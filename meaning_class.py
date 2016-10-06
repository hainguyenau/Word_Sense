from hierarchical_model import *
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
import os

class Meaning:
    def __init__(self, word, pos):
        self.word = word
        self.pos = pos
        self.vectorizer = None
        self.model = None
        self.num_clusters = None

    # Traing the class with the given model, word and pos
    def train(self):
        # train the meaning objec with proper train set
        train_documents = read_train_data(self.word, self.pos)
        self.vectorizer = vectorize(train_documents)
        vectors = transform(self.vectorizer, train_documents)

        # scale the feature vectors
        scaler = MaxAbsScaler(copy=False)
        vectors = scaler.fit_transform(vectors)
        self.model, self.num_clusters = model_fit(self.word, self.pos, vectors)

    # Load test data
    def load_test(self):
        folder = None
        if self.pos == 'n':
            folder = 'nouns'
        elif self.pos == 'v':
            folder = 'verbs'

        tree = ET.parse('test_data/{}/{}.{}.xml'.format(folder, self.word, self.pos))
        root = tree.getroot()
        test_documents = []

        for i in xrange(len(root.getchildren())):
            if list(root)[i].text and list(root)[i][0].text:
                s = list(root)[i].text + list(root)[i][0].text
                test_documents.append(s)
            else:
                s = list(root)[i][0].text
        return test_documents

    def save(self):
        # save model and vectorizer
        with open('pickles/meaning_{}.pkl'.format(self.word), 'wb') as f:
            pickle.dump(self, f)


    @staticmethod
    def load(word):
        # create object and load pickle
        with open('meaning_{}.pkl'.format(word)) as f:
            meaning = pickle.load(f)
        return meaning

    def predict(self, sentence):
        # return meaning
        if self.word in sentence:
            v = self.vectorizer.transform([sentence])
            label =  self.model.predict(v)[0]
            return labels_to_defs(self.model, self.word, self.pos)[label]
        else:
            return 'The word '+ self.word + ' is not in the sentence!!!!'

    #Predict with nltk
    def nltk_predict(self, sentence):
        tk = word_tokenize(sentence)
        return lesk(tk, self.word, self.pos).definition()



if __name__ == '__main__':
    obj = Meaning('access', 'n')
    obj.train()
    # test_documents = obj.load_test()
    print np.bincount(obj.model.labels_)


    # Assess score of ouro predictions compared to nltk
    # score = 0
    # for sent in test_documents:
    #     our_predictions = obj.predict(sent)
    #     nltk_prediction = obj.nltk_predict(sent)
    #
    #     if our_predictions == nltk_prediction:
    #         score += 1
    #     # print our_predictions
    #     # print nltk_prediction
    # print score, len(test_documents)














    # Iterate over the noun files
    # nouns = []
    # for filename in os.listdir('training_data/nouns'):
    #     if filename.endswith(".xml"):
    #         nouns.append(filename[:-6])
    #
    # # Iterate over the noun files
    # verbs = []
    # for filename in os.listdir('training_data/verbs'):
    #     if filename.endswith(".xml"):
    #         verbs.append(filename[:-6])
    #
    # # Traing all the nouns
    # for noun in nouns:
    #     obj = Meaning(noun, 'n')
    #     obj.train()
    #     obj.save()
