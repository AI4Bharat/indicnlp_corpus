# <center>IndicNLP Corpora</center>





- currently many are stored on indicnlp azure blob
- All data maybe eventually stored on a dedicated ai4bharat bucket
  - Anoop, Mitesh and Gokul to discuss and decide
  - Note that network egress from cloud storage (i.e. people downloading our datasets) costs around $0.1/GB.
- Logo for IndicNLP Corpus (Avik?)





## Table of contents

* [Introduction](#introduction)
* [Text Corpora](#text-corpora)
* [Word Embeddings](#word-embeddings)
* [Morphanalyser](#morphanalyser)
* [License](#license)
* [Contributors](#contributors)
* [Contact](#contact)





## Introduction

(_Anoop, complete this section_)

- Brief description of the project
- Link to the paper on arxiv




## Text Corpora

(_Divyanshu, complete this section_)

- we should remove or cite the datasets that we obtained from other sources, like hindi's
- @Anoop: Note that the v1 datasets for kn, or, pa, ta, te are on google cloud at `gs://nlp-corpora--ai4bharat/indicnlp-datasets/monoling-v1/.` If we are not going with google cloud, we should copy it over to your Microsoft blob storage.


| Language | Sentences | Tokens  | Types | Download Link                                                |
| -------- | --------- | ------- | ----- | ------------------------------------------------------------ |
| bn       | 7.2M      | 100.1M  | 1.5M  | [v1](https://indicnlp.blob.core.windows.net/data/monolingual/sentence/bn.txt.gz) |
| gu       | 7.8M      | 129.7M  | 2.4M  | [v1](https://indicnlp.blob.core.windows.net/data/monolingual/sentence/gu.txt.gz) |
| hi       | 62.9M     | 1199.8M | 5.3M  | [v1](https://indicnlp.blob.core.windows.net/data/monolingual/sentence/hi.txt.gz) |
| kn       | 14.7M     | 174.9M  | 3.0M  | [v1](https://indicnlp.blob.core.windows.net/data/monolingual/sentence/kn.txt.gz) |
| ml       | 11.6M     | 167.4M  | 8.8M  | [v1](https://indicnlp.blob.core.windows.net/data/monolingual/sentence/ml.txt.gz) |
| mr       | 9.9M      | 142.4M  | 2.6M  | [v1](https://indicnlp.blob.core.windows.net/data/monolingual/sentence/mr.txt.gz) |
| or       | 3.5M      | 51.5M   | 0.7M  | [v1](https://indicnlp.blob.core.windows.net/data/monolingual/sentence/or.txt.gz) |
| pa       | 6.5M      | 179.4M  | 0.5M  | [v1](https://indicnlp.blob.core.windows.net/data/monolingual/sentence/pa.txt.gz) |
| ta       | 20.9M     | 362.8M  | 9.4M  | [v1](https://indicnlp.blob.core.windows.net/data/monolingual/sentence/ta.txt.gz) |
| te       | 15.1M     | 190.2M  | 4.1M  | [v1](https://indicnlp.blob.core.windows.net/data/monolingual/sentence/te.txt.gz) |

**Note**: We are working on releasing much larger dataset (version 2) soon.


## Pre-requisites 

To replicate the results reported in the paper, training and evaluation scripts are provided.
To run these scripts, the following tools/packages are required: 

- [FastText](https://github.com/facebookresearch/fastText)
- [MUSE](https://github.com/facebookresearch/MUSE)

## Word Embeddings

(_Anoop, complete this section_)

**Download Links**

_Version 1_ 

| language | pa | hi | bn | or | gu | mr | kn | te | ml | ta |
| -------- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| vectors | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.pa.vec.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.hi.vec.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.bn.vec.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.or.vec.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.gu.vec.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.mr.vec.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.kn.vec.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.te.vec.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.ml.vec.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.ta.vec.gz) |
| model | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.pa.bin.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.hi.bin.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.bn.bin.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.or.bin.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.gu.bin.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.mr.bin.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.kn.bin.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.te.bin.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.ml.bin.gz) | [link](https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.ta.bin.gz) |


**Training word embeddings**

```bash
$FASTTEXT_HOME/build/fasttext skipgram \
	-epoch 10 -thread 30 -ws 5 -neg 10    -minCount 5 -dim 300 \
	-input $mono_path \
	-output $output_emb_prefix 
```

**Evaluation on word similarity task**

Evaluate on the IIIT-H Word Similarity Database: `https://indicnlp.blob.core.windows.net/evaluations/word_similarity/iiith_wordsim.tgz`

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

## Morphanalyzers

IndicNLP Morphanalyzers are unsupervised morphanalyzers trained with [morfessor](https://github.com/aalto-speech/morfessor)

**Download Links**

_Version 1_

| [pa](https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.pa.model.gz) | [hi](https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.hi.model.gz) | [bn](https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.bn.model.gz) | [or](https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.or.model.gz) | [gu](https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.gu.model.gz) | [mr](https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.mr.model.gz) | [kn](https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.kn.model.gz) | [te](https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.te.model.gz) | [ml](https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.ml.model.gz) | [ta](https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.ta.model.gz) |
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

## IndicNLP News Article Classification Dataset

(_Divyanshu, complete this section_)

**Version 1.0**

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

The dataset can be downloaded from [here](). To evaluate your embeddings, first download the dataset and then run the following command:

```bash
python3 scripts/txtcls.py --emb_path <path> --data_dir <path> --lang <lang code>
```



- data format - described in the readme file

* raw article text - meaning?

- Evaluation scripts (TBD by Divyanshu): -- done
  - Please include the evaluation scripts in this repo
  - Ensure that the scripts are running

- Download links:
  - Divyanshu: prepare the data set and share with anoop
    - include a readme with a pointer to this section and mention of paper to cite
  - Anoop: upload to the above dataset to the  below locations
    - `https://indicnlp.blob.core.windows.net/data/classification/news-article/indicnlp-news-article-v1.0.tgz`



## Compiled Classification Datasets

(_Divyanshu, complete this section_)

To further enhance our evaluation benchmarks, we compiled the following publicly-available evaluation datasets.

* ACTSA Corpus: Sentiment analysis corpus for Telugu sentences. [download link]()
* BBC News Articles: Sentiment analysis corpus for Hindi documents extracted from BBC news website. [dowload link]()
* IIT Patna Product Reviews: Sentiment analysis corpus for product reviews posted in Hindi. [download link]()
* INLTK Headlines Corpus: Obtained from [inltk](https://github.com/goru001/inltk) project. The corpus is a collection of headlines tagged with their news category. Available for langauges: gu, ml, mr, ta and te. [download link]()
* IIT Patna Movie Reviews: Sentiment analysis corpus for movie reviews posted in Hindi. [download link]()
* Bengali News Articles: Contains Bengali news articles tagged with their news category. [download link]()



To evaluate your embeddings on any of the above dataset, first download them and then run the following command:

```bash
python3 scripts/txtcls.py --emb_path <path> --data_dir <path> --lang <lang code>
```



- Evaluation scripts (TBD by Divyanshu):
  - may need to rerun the evaluation for some languages since embeddings were updated
  - Anoop: check if Satish already trained with updated data
- Data Download links:
  - Divyanshu: Prepare the data set and share with anoop -- done
  - Anoop: Upload to the above dataset to the below locations
- `https://indicnlp.blob.core.windows.net/data/classification/public-eval_datasets-v1.0.tgz`
- License: available under original license



## Citing

If you are using any of the resources, please cite the following article: 

```
@article{kunchukuttan2020indicnlpcorpus,
    title={IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
    author={Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    journal={arXiv preprint arXiv:2004.YYYY},
}
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Dataset" property="dct:title" rel="dct:type">IndicNLP Corpus</span>  is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.


## Contributors

* Anoop Kunchukuttan
* Divyanshu Kakwani
* Satish Golla
* Gokul NC
* Avik Bhattacharyya

## Contact

- Anoop Kunchukuttan (anoop.kunchukuttan@gmail.com)
- Mitesh Khapra (miteshk@cse.iitm.ac.in)
- Pratyush Kumar (pratyushk@cse.iitm.ac.in)
