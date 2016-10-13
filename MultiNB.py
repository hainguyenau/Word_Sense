import numpy as np
import pandas as pd
import re
import gensim.utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict

class MultiNB:
    def __init__(self):
        self.lem_documents = None
        self.vectorizer = None
        self.vectors = None
        self.labels = None
        self.d = None
        self.NB = None

    # Function that takes in a filename and load that file into a self's dict
    def set_dict(self, filename):
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
        self.d = d

    # Function that takes in a documents (list) and return a lemmatized documents (list)
    def set_lem_documents(self):
        train_documents = [sentence for value in self.d.values() for sentence in value]

        lem_documents = []
        for doc in train_documents:
            no_tag_words = [w[:-3] for w in gensim.utils.lemmatize(doc)]
            lem_documents.append(' '.join(no_tag_words))
        self.lem_documents = lem_documents

    # Set main vectorizer for class
    def set_vectorizer(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3)).fit(self.lem_documents)


    # Function that takes in a (lemmatized) documents (list). Return vectors and vectorizer
    def Tfidf(self, lem_documents):
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3)).fit(lem_documents)
        vectors = vectorizer.transform(lem_documents)
        return vectors, vectorizer


    # Set the labels numpy array for the class
    def set_labels(self):
        ys = []
        for i in self.d.keys():
            yi = np.repeat(str(i), len(self.d[i]))
            ys.append(yi)
        self.labels = np.concatenate(ys)

    #
    def fit_NB(self, vectors):
        NB = MultinomialNB(alpha= 0.12)
        NB.fit(vectors,y)
        self.NB = NB




if __name__ == '__main__':
    # Create a MultiNB object
    obj = MultiNB()

    # Load data, set d
    obj.set_dict('hard.cor')

    # Set train_documents
    obj.set_lem_documents()

    # Set class labels
    obj.set_labels()

    # Set class vectorizer
    obj.set_vectorizer()

    # Fit NB model
    obj.fit_NB()

    # Predict a sentence
    s = "I work hard"
    vector = obj.vectorizer.transform([s])
    print obj.NB.predict(vector)
# # Scoring the train set
# pred_score_hard1, pred_score_hard2, pred_score_hard3 = 0, 0, 0
#
# for sent in obj.d['HARD1']:
#
#     if predict_NB(sent, vectorizer, NB) == 'HARD1':
#         pred_score_hard1 += 1
# print "HARD1 score: ", pred_score_hard1, '/', len(d['HARD1'])
#
# for sent in d['HARD2']:
#     if predict_NB(sent, vectorizer, NB) == 'HARD2':
#         pred_score_hard2 += 1
# print "HARD2 score: ", pred_score_hard2, '/', len(d['HARD2'])
#
# for sent in d['HARD3']:
#     if predict_NB(sent, vectorizer, NB) == 'HARD3':
#         pred_score_hard3 += 1
# print "HARD3 score: ", pred_score_hard3, '/', len(d['HARD3'])
