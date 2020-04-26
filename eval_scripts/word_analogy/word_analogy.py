
from src.utils import load_embeddings 
from torch import nn
from src.evaluation.wordsim import get_word_id
import os, io
import numpy as np
import torch
import argparse

def get_wordanalogy_scores_customfname(analogy_fname, language, word2id, embeddings, lower=True):
    """
    Return (english) word analogy score
        embeddings must be normalized 
    This is a modification of the MUSE analogy code
    """

    # normalize word embeddings
    embeddings = embeddings / np.sqrt((embeddings ** 2).sum(1))[:, None]

    # scores by category
    scores = {}
    n_found=0
    n_not_found=0

    word_ids = {}
    queries = {}

    with io.open(analogy_fname, 'r', encoding='utf-8') as f:
        for line in f:
            # new line
            line = line.rstrip()
            if lower:
                line = line.lower()

            # new category
            if ":" in line:
#                 assert line[1] == ' '
                category = line[2:]
#                 assert category not in scores
                scores[category] = {'n_found': 0, 'n_not_found': 0, 'n_correct': 0}
                word_ids[category] = []
                queries[category] = []
                continue

            # get word IDs
            assert len(line.split()) == 4, line
            word1, word2, word3, word4 = line.split()
            word_id1 = get_word_id(word1, word2id, lower)
            word_id2 = get_word_id(word2, word2id, lower)
            word_id3 = get_word_id(word3, word2id, lower)
            word_id4 = get_word_id(word4, word2id, lower)

            # if at least one word is not found
            if any(x is None for x in [word_id1, word_id2, word_id3, word_id4]):
                scores[category]['n_not_found'] += 1
                continue
            else:
                scores[category]['n_found'] += 1
                word_ids[category].append([word_id1, word_id2, word_id3, word_id4])
                # generate query vector and get nearest neighbors
                query = embeddings[word_id1] - embeddings[word_id2] + embeddings[word_id4]
                query = query / np.linalg.norm(query)

                queries[category].append(query)

    # Compute score for each category
    for cat in queries:
        qs = torch.from_numpy(np.vstack(queries[cat]))
        keys = torch.from_numpy(embeddings.T)
        values = qs.mm(keys).cpu().numpy()

    # be sure we do not select input words
        for i, ws in enumerate(word_ids[cat]):
            for wid in [ws[0], ws[1], ws[3]]:
                values[i, wid] = -1e9
        scores[cat]['n_correct'] = np.sum(values.argmax(axis=1) == [ws[2] for ws in word_ids[cat]])

    # pretty print
    separator = "=" * (30 + 1 + 10 + 1 + 13 + 1 + 12)
    pattern = "%30s %10s %13s %12s"
    print(separator)
    print(pattern % ("Category", "Found", "Not found", "Accuracy"))
    print(separator)

    # compute and log accuracies
    accuracies = {}
    for k in sorted(scores.keys()):
        v = scores[k]
        accuracies[k] = float(v['n_correct']) / max(v['n_found'], 1)
        print(pattern % (k, str(v['n_found']), str(v['n_not_found']), "%.4f" % accuracies[k]))
        n_found+=v['n_found']
        n_not_found+=v['n_not_found']
    print(separator)
    
    print()
    print('Coverage: {}'.format(n_found/(n_found+n_not_found)))

    return accuracies
        
        
class Params(object): 
    def __init__(self):
        pass 
        
def score_analogy(analogy_fname, embeddings_path, lang, emb_dim, max_vocab=200000, lower=True, cuda=True):

    # source embeddings
    params=Params()
    params.src_emb=embeddings_path
    params.tgt_emb=''
    params.max_vocab=max_vocab 
    params.emb_dim=emb_dim
    params.cuda=cuda
    params.src_lang=lang
    params.tgt_lang=''
        
    src_dico, _src_emb = load_embeddings(params, source=True)
    word2id = src_dico.word2id
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    if params.cuda:
        src_emb.cuda()
        
    embeddings=src_emb.weight.data.cpu().numpy()
    word2id=src_dico.word2id

    return get_wordanalogy_scores_customfname(analogy_fname, lang, word2id, embeddings, lower=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--analogy_fname', type=str, help='Analogy dataset file path', )
    parser.add_argument('--embeddings_path', type=str, help='Embeddings file path', )
    parser.add_argument('--lang', type=str, help='Language', )
    parser.add_argument('--emb_dim', type=int, help='Dimension of the embedding', default=300)
    parser.add_argument('--max_vocab', type=int, help='Number of vocab to use', default=200000)
    parser.add_argument('--cuda', action='store_true', help='Use CUDA?')
    parser.add_argument('--lower', action='store_true', help='Use lowercase?')

    args = parser.parse_args()
    print(args)
    
    analogy_scores=score_analogy(args.analogy_fname, args.embeddings_path, args.lang, 
                                 args.emb_dim, args.max_vocab, args.lower, args.cuda)
    
    print('Average score: ',end='')
    print(np.mean(list(analogy_scores.values())))
