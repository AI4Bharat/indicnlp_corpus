
import os
import csv
import fasttext
import faiss
import tempfile
import tarfile
import requests
import numpy as np

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize
from indicnlp.tokenize import sentence_tokenize


normalizer_factory = indic_normalize.IndicNormalizerFactory()


def doc2vec(txt, lang, emb):
    """
    a doc is represented as the mean of all the words vectors
    of its constituent words
    """
    normalizer = normalizer_factory.get_normalizer(lang)
    normed_txt = normalizer.normalize(txt.replace('\n', ' '))
    words = indic_tokenize.trivial_tokenize(normed_txt, lang)
    word_vecs = [emb[word] for word in words if word in emb]
    if len(word_vecs) > 0:
        doc_vec = np.mean(np.array(word_vecs), axis=0)
    else:
        doc_vec = np.zeros(300)
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
        self.label_encoder = preprocessing.LabelEncoder()
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

        k = 5
        dim = 300
        database = self.train[:, 1:].astype('float32')
        queries = self.test[:, 1:].astype('float32')
        index = faiss.IndexFlatL2(dim)
        index.add(database)
        dist, idxs = index.search(queries, k)

        preds = []
        for neighbors in idxs:
           classes = [self.train[n, 0] for n in neighbors]
           preds.append(max(set(classes), key=classes.count))

        return accuracy_score(self.test[:,0], preds)

    def process_dataset(self, data, emb):
        print('Building document vectors...')
        label_ids = self.label_encoder.transform(data[:, 0])
        label_ids = np.expand_dims(label_ids, axis=1)
        doc_vecs = np.array([doc2vec(txt, self.lang, emb) for txt in tqdm(data[:, 1])])
        processed = np.hstack((label_ids, doc_vecs))
        return processed


if __name__ == '__main__':
    import argparse
    from gensim.models import KeyedVectors

    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', help='Path to embedding bin', type=str, required=True)
    parser.add_argument('--data_dir', help='Path to evaluation data directory', type=str, required=True)
    parser.add_argument('--lang', help='Language', type=str, required=True)
    args = parser.parse_args()
    emb = KeyedVectors.load_word2vec_format(args.emb_path, binary=False, encoding='utf8')
    print('Loaded embedding')
    
    txtcls = TxtCls(lang=args.lang, data_dir=args.data_dir)
    print('Accuracy: ', txtcls.evaluate(emb))
