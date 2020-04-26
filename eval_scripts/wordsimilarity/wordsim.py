import sys
import embeddings
import numpy as np
import scipy

def read_word_similarity(similarity_fname, delim='\t'):
    """read word similarity information file

    Args:
        similarity_fname (str): path word similarity database file
        delim (str): delimiter for each record in word similarity file 

    Returns:
        list of tuples of the form ('word0', 'word1', similarity)
    """
    sim_database=[]
    with open(similarity_fname,'r',encoding='utf-8') as similarity_file:
        for l in similarity_file:
            f = l.strip().split(delim)
            sim_database.append((f[0],f[1],float(f[2])))
    return sim_database

def compute_word_similarity(emb_info, sim_database):
    """Compute word similarity for the word pair dataset 

    Compute word similarity for the word pair dataset using 
    Spearman's correlation coefficient 

    Args:
        emb_info (tuple): tuple of 
        sim_database (list): See return of read_word_similarity 
            for format

    Returns: 
        tuple of (correlation,p_value,coverage)
        coverage refers to the fraction of the word pairs 
        in the similarity database covered by the embeddings 
        
    """
            
    emb_words, emb_vectors = emb_info
    w2i=build_w2i(emb_info[0])
    
    sim_words = set([ x[0] for x in sim_database ])
    sim_words.update([ x[1] for x in sim_database ])
    oov_words = sim_words.difference(emb_words)
    non_oov_words=sim_words.difference(oov_words)
    
    non_oov_sim_pairs = list(filter( lambda x: len(oov_words.intersection(x[:2]))==0 , sim_database))

    cos_sims=[]
    ref_sims=[]
    
    for w1, w2, ref_sim in non_oov_sim_pairs:
        v1=emb_vectors[w2i[w1]]
        v2=emb_vectors[w2i[w2]]
        cos_sim=np.dot(v1,v2)/np.sqrt(v1.dot(v1)*v2.dot(v2))
        
        cos_sims.append(cos_sim)
        ref_sims.append(ref_sim)
    
    corr=scipy.stats.spearmanr(np.array(cos_sims),np.array(ref_sims))
    return corr[0], corr[1], len(non_oov_sim_pairs)/len(sim_database)

if __name__ == '__main__':
    
    emb_fname=sys.argv[1]
    wsim_fname=sys.argv[2]
    max_voc=200000
    if len(sys.argv)>=4:
        max_voc = int(sys.argv[3])
    
    with open(emb_fname, 'r', encoding='utf-8') as emb_file:

        emb_info = embeddings.read(emb_file, max_voc=max_voc)
        wsim_db=geomm_utils.read_word_similarity(wsim_fname)
        correlation, pvalue, coverage = geomm_utils.compute_word_similarity(emb_info, wsim_db)

        print('Max Vocabulary: {}'.format(max_voc))
        print('Correlation: {}'.format(correlation))
        print('p-value: {}'.format(pvalue))
        print('Coverage: {}'.format(coverage))

