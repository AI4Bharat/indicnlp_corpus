
import os
import csv
import tempfile
import tarfile
import requests
import numpy as np

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def doc2vec(txt, emb):
    """
    a doc is represented as the mean of all the words vectors
    of its constituent words
    """
    words = txt.split()
    word_vecs = [emb.wv[word] for word in words]
    doc_vec = np.mean(np.array(word_vecs), axis=0)
    return doc_vec


class TxtCls:

    def __init__(self, *args, **kwargs):
        """
        Downloads and loads the evaluation dataset in memory

        The raw data is stored as an n x 2 array where the first columns
        holds the class label string and the second column holds the
        entire text of the document
        """
        self.data_dir = kwargs['data_dir']
        self.lang = kwargs['lang']
        self.task_dir = os.path.join(self.data_dir, 'indicnlp-txtcls')

        if not os.path.exists(self.task_dir):
            self.download(self.data_dir)

        lang_dir = os.path.join(self.task_dir, self.lang)
        train_fname = os.path.join(lang_dir, self.lang + '-train.csv')
        test_fname = os.path.join(lang_dir, self.lang + '-test.csv')

        self.raw_train = np.array(self.load_data(train_fname))
        self.raw_test = np.array(self.load_data(test_fname))

        labels = np.concatenate((self.raw_train[:, 0], self.raw_test[:, 0]))
        self.label_encoder = preprocessing.LabelEncoder(labels)
        self.label_encoder.fit(labels)

    def load_data(self, fname):
        with open(fname) as data_file:
            reader = csv.reader(data_file, delimiter=',')
            data = list(reader)
        return data

    def download(self, download_dir):
        fp = tempfile.NamedTemporaryFile()
        remote_url = 'https://storage.googleapis.com/nlp-corpora--ai4bharat/'\
                     'indicnlp-datasets/evaluation/indicnlp-txtcls.tar.xz'
        blob = requests.get(remote_url).content
        fp.write(blob)
        tar = tarfile.open(fp.name)
        tar.extractall(path=download_dir)
        tar.close()

    def evaluate(self, emb):
        self.train = self.process_dataset(self.raw_train, emb)
        self.test = self.process_dataset(self.raw_test, emb)

        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(self.train[:,1:], self.train[:,0])
        preds = knn.predict(self.test[:,1:])

        return accuracy_score(self.test[:,0], preds)

    def process_dataset(self, data, emb):
        label_ids = self.label_encoder.transform(data[:, 0])
        label_ids = np.expand_dims(label_ids, axis=1)
        doc_vecs = np.array([doc2vec(txt, emb) for txt in data[:, 1]])
        processed = np.hstack((label_ids, doc_vecs))
        return processed
