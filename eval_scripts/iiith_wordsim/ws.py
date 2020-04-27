"""
Code derived from:
https://github.com/kudkudak/word-embeddings-benchmarks/blob/master/web/evaluate.py
"""

import numpy as np
import tempfile
import zipfile
import os
import requests

from scipy import stats


class WordSimilarity:

    def __init__(self, *args, **kwargs):
        self.data_dir = kwargs['data_dir']
        self.lang = kwargs['lang']
        self.task_dir = os.path.join(self.data_dir, 'iiith-wordsim')

        if not os.path.exists(self.task_dir):
            self.download(self.data_dir)

        self.fpath = os.path.join(self.task_dir, lang + '.txt')
        self.scores = []

        with open(self.fpath) as fp:
            content = fp.read()
        lines = content.split('\n')
        for line in lines:
            fields = line.split()
            if len(fields) < 3:
                continue
            self.scores.append((fields[0], fields[1], float(fields[2])))

    def download(self, download_dir):
        fp = tempfile.NamedTemporaryFile()
        remote_url = 'https://storage.googleapis.com/nlp-corpora--ai4bharat/'\
                     'indicnlp-datasets/evaluation/iiith-wordsim.tar.xz'
        blob = requests.get(remote_url).content
        fp.write(blob)
        tar = tarfile.open(fp.name)
        tar.extractall(path=download_dir)
        tar.close()

    def evaluate(self, emb):
        A = np.vstack([emb.wv[w1] for w1, w2, s in self.scores])
        B = np.vstack([emb.wv[w2] for w1, w2, s in self.scores])
        y = np.array([s for w1, w2, s in self.scores])

        pred_scores = [v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                       for v1, v2 in zip(A, B)]
        pred_scores = np.array(pred_scores)
        return stats.spearmanr(pred_scores, y).correlation


if __name__ == '__main__':
    import argparse
    from gensim.models.fasttext import load_facebook_model

    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', help='Path to embedding bin', type=str, required=True)
    parser.add_argument('--data_dir', help='Path to evaluation data directory', type=str, required=True)
    parser.add_argument('--lang', help='Language', type=str, required=True)
    args = parser.parse_args()
    emb = load_fasttext_model(args.emb_path)
    
    ws = WordSimilarity(lang=args.lang, data_dir=args.data_dir)
    ws.evaluate(emb)
