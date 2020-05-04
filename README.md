# <center>AI4Bharat-IndicNLP Corpus</center>

The AI4Bharat-IndicNLP corpus is an ongoing effort to create a large-scale, general-domain corpus for Indian languages. Currently, it contains 2.7 billion words for 10 Indian languages from two language families. We  share pre-trained word embeddings trained on these corpora. We create news article category classification datasets for 9 languages to evaluate the embeddings. We evaluate the IndicNLP embeddings on multiple evaluation tasks. You can read details regarding the corpus and other resources [HERE](ai4bharat-indicnlp-corpus-2020.pdf).

## Table of contents

* [Text Corpora](#text-corpora)
* [Word Embeddings](#word-embeddings)
* [IndicNLP News Article Classification Dataset](#indicnlp-news-article-classification-dataset)
* [Publicly available Classification Datasets](#publicly-available-classification-datasets)
* [Morphanalyzers](#morphanalyzers)
* [Other Indian language NLP Resources](#other-indian-language-nlp-resources)
* [Citing](#citing)
* [License](#license)
* [Contributors](#contributors)
* [Contact](#contact)

## Text Corpora

Following are the statistics of the text corpora used to build the embeddings and other resources. The text corpus will be available later, meanwhile you can download the embeddings, classification sets and other resources.

| Language | Sentences | Tokens  | Types | Vocab Frequency |
| -------- | --------- | ------- | ----- | --------------- |
| bn       | 7.2M      | 100.1M  | 1.5M  | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/corpus_stats/bn.vocabfreq.tsv.gz) |
| gu       | 7.8M      | 129.7M  | 2.4M  | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/corpus_stats/gu.vocabfreq.tsv.gz) |
| hi       | 62.9M     | 1199.8M | 5.3M  | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/corpus_stats/hi.vocabfreq.tsv.gz) |
| kn       | 14.7M     | 174.9M  | 3.0M  | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/corpus_stats/kn.vocabfreq.tsv.gz) |
| ml       | 11.6M     | 167.4M  | 8.8M  | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/corpus_stats/ml.vocabfreq.tsv.gz) |
| mr       | 9.9M      | 142.4M  | 2.6M  | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/corpus_stats/mr.vocabfreq.tsv.gz) |
| or       | 3.5M      | 51.5M   | 0.7M  | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/corpus_stats/or.vocabfreq.tsv.gz) |
| pa       | 6.5M      | 179.4M  | 0.5M  | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/corpus_stats/pa.vocabfreq.tsv.gz) |
| ta       | 20.9M     | 362.8M  | 9.4M  | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/corpus_stats/ta.vocabfreq.tsv.gz) |
| te       | 15.1M     | 190.2M  | 4.1M  | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/data/monolingual/indicnlp_v1/corpus_stats/te.vocabfreq.tsv.gz) |

**Note** 

- The vocabulary frequency files contain the frequency of all unique tokens in the corpus. Each line contains one word along with frequency delimited by tab.

## Pre-requisites 

To replicate the results reported in the paper, training and evaluation scripts are provided.

To run these scripts, the following tools/packages are required: 

- [FastText](https://github.com/facebookresearch/fastText)
- [MUSE](https://github.com/facebookresearch/MUSE)

For Python packages to install, see `requirements.txt`

## Word Embeddings

**DOWNLOAD**

_Version 1_ 

| language | pa | hi | bn | or | gu | mr | kn | te | ml | ta |
| -------- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| vectors | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.pa.vec.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.hi.vec.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.bn.vec.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.or.vec.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.gu.vec.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.mr.vec.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.kn.vec.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.te.vec.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.ml.vec.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.ta.vec.gz) |
| model | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.pa.bin.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.hi.bin.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.bn.bin.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.or.bin.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.gu.bin.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.mr.bin.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.kn.bin.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.te.bin.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.ml.bin.gz) | [link](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding/indicnlp.v1.ta.bin.gz) |

**Training word embeddings**

```bash
$FASTTEXT_HOME/build/fasttext skipgram \
	-epoch 10 -thread 30 -ws 5 -neg 10    -minCount 5 -dim 300 \
	-input $mono_path \
	-output $output_emb_prefix 
```

**Evaluation on word similarity task**

Evaluate on the IIIT-H Word Similarity Database: [**DOWNLOAD**](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/evaluations/word_similarity/iiith_wordsim.tgz)

The above mentioned link is a cleaned version of the same database found [HERE](https://github.com/syedsarfarazakhtar/Word-Similarity-Datasets-for-Indian-Languages).

_Evaluation Command_


```bash
python scripts/word_similarity/wordsim.py \
	<embedding_file_path> \
	<word_sim_db_path> \
	<max_vocab>
```

**Evaluation on word analogy task**

Evaluate on the [Facebook word analogy dataset](https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-hi.txt).

_Evaluation Command_

```bash
python  scripts/word_analogy/word_analogy.py \
    --analogy_fname <analogy_fname> \
    --embeddings_path <embedding_file_path> \
    --lang 'hi' \
    --emb_dim 300 \
    --cuda
```

## IndicNLP News Article Classification Dataset

We used the IndicNLP text corpora to create classification datasets comprising news articles and their categories for 9 languages. The dataset is balanced across classes.  The following table contains the statistics of our dataset:

| Language  | Classes                                     | Articles per Class |
| --------- | ------------------------------------------- | ------------------ |
| Bengali   | entertainment, sports                       | 7K                 |
| Gujarati  | business, entertainment, sports             | 680                |
| Kannada   | entertainment, lifestyle, sports            | 10K                |
| Malayalam | business, entertainment, sports, technology | 1.5K               |
| Marathi   | entertainment, lifestyle, sports            | 1.5K               |
| Oriya     | business, crime, entertainment, sports      | 7.5K               |
| Punjabi   | business, entertainment, sports, politics   | 780                |
| Tamil     | entertainment, politics, sport              | 3.9K               |
| Telugu    | entertainment, business, sports             | 8K                 |


[**DOWNLOAD**](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/evaluations/classification/indicnlp-news-articles.tgz)

**Evaluation Command** 

```bash
python3 scripts/txtcls.py --emb_path <path> --data_dir <path> --lang <lang code>
```

## Publicly available Classification Datasets

We also evaluated the IndicNLP embeddings on many publicly available classification datasets. 

* ACTSA Corpus: Sentiment analysis corpus for Telugu sentences. 
* BBC News Articles: Sentiment analysis corpus for Hindi documents extracted from BBC news website. 
* IIT Patna Product Reviews: Sentiment analysis corpus for product reviews posted in Hindi. 
* INLTK Headlines Corpus: Obtained from [inltk](https://github.com/goru001/inltk) project. The corpus is a collection of headlines tagged with their news category. Available for langauges: gu, ml, mr, ta and te. 
* IIT Patna Movie Reviews: Sentiment analysis corpus for movie reviews posted in Hindi. 
* Bengali News Articles: Contains Bengali news articles tagged with their news category.

We have created standard test, validation and test splits for the above mentioned datasets. You can download them to evaluate your embeddings.

[**DOWNLOAD**](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/evaluations/classification/classification_public_datasets.tgz)

**Evaluation Command** 

To evaluate your embeddings on the above mentioned datasets, first download them and then run the following command:

```bash
python3 scripts/txtcls.py --emb_path <path> --data_dir <path> --lang <lang code>
```

**License**

These datasets are available under original license for each public dataset. 

## Morphanalyzers

IndicNLP Morphanalyzers are unsupervised morphanalyzers trained with [morfessor](https://github.com/aalto-speech/morfessor).

**DOWNLOAD**

_Version 1_

| [pa](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/morph/morfessor/indicnlp.v1.pa.model.gz) | [hi](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/morph/morfessor/indicnlp.v1.hi.model.gz) | [bn](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/morph/morfessor/indicnlp.v1.bn.model.gz) | [or](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/morph/morfessor/indicnlp.v1.or.model.gz) | [gu](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/morph/morfessor/indicnlp.v1.gu.model.gz) | [mr](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/morph/morfessor/indicnlp.v1.mr.model.gz) | [kn](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/morph/morfessor/indicnlp.v1.kn.model.gz) | [te](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/morph/morfessor/indicnlp.v1.te.model.gz) | [ml](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/morph/morfessor/indicnlp.v1.ml.model.gz) | [ta](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/morph/morfessor/indicnlp.v1.ta.model.gz) |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

**Training Command**

```bash
## extract vocabulary from embedings file
zcat $embedding_vectors_path |  \
    tail -n +2 | \
    cut -f 1 -d ' '  > $vocab_file_path

## train morfessor 
morfessor-train -d ones \
        -S $model_file_path \
        --logfile  $log_file_path \
        --traindata-list $vocab_file_path \
        --max-epoch 10 
```

## Other Indian language NLP Resources

You can use IndicNLP embeddings for other Indian language tasks too. A comprehensive list of Indian language NLP resources can be found in the [IndicNLP Catalog](https://github.com/indicnlpweb/indicnlp_catalog). For processing the Indian language text, you can use the [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library).

## Citing

If you are using any of the resources, please cite the following article: 

```
@article{kunchukuttan2020indicnlpcorpus,
    title={IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
    author={Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    journal={arXiv preprint arXiv:2005.00085},
}
```

We would like to hear from you if: 

- You are using our resources. Please let us know how you are putting these resources to use. 
- You have any feedback on these resources. 

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Dataset" property="dct:title" rel="dct:type">IndicNLP Corpus</span>  is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.


## Contributors

* Anoop Kunchukuttan
* Divyanshu Kakwani
* Satish Golla
* Gokul NC
* Avik Bhattacharyya
* Mitesh Khapra
* Pratyush Kumar

## Contact

- Anoop Kunchukuttan (anoop.kunchukuttan@gmail.com)
- Mitesh Khapra (miteshk@cse.iitm.ac.in)
- Pratyush Kumar (pratyushk@cse.iitm.ac.in)
