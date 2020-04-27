# IndicNLP Corpora





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




## Text Corpus

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

- [FastText]()
- [MUSE]()

## Word Embeddings

(_Anoop, complete this section_)

** Training word embedding **

```
$FASTTEXT_HOME/build/fasttext skipgram \
	-epoch 10 -thread 30 -ws 5 -neg 10    -minCount 5 -dim 300 \
	-input $mono_path \
	-output $output_emb_prefix 
```

**Computing word similarity**

IIIT-H Word Similarity Database: `https://indicnlp.blob.core.windows.net/evaluations/word_similarity/iiith_wordsim.tgz`

_Evaluation Command_

```
scripts/word_similarity/wordsim.sh \
	<embedding_file_path> \
	<word_sim_db_path> \
	200000
```


**Word analogy dataset**
TBD: (put) put link to Google analogy datasets

**Evaluation Script**


### Version 1.0
**Download Links**

- `https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.<langcode>.vec.gz`

- `https://indicnlp.blob.core.windows.net/embedding/indicnlp.v1.<langcode>.bin.gz`



## Morphanalyser

(_Anoop, complete this section_)
### Version 1.0

**Training Scripts**

**Download Links**

`https://indicnlp.blob.core.windows.net/morph/morfessor/indicnlp.v1.<laangcode>.model.gz`



## IndicNLP News Article Classification Dataset

(_Divyanshu, complete this section_)

**Version 1.0**

- data format

* raw article text

- Evaluation scripts (TBD by Divyanshu):
  - Please include the evaluation scripts in this repo
  - Ensure that the scripts are running

- Download links:
  - Divyanshu: prepare the data set and share with anoop
    - include a readme with a pointer to this section and mention of paper to cite
  - Anoop: upload to the above dataset to the  below locations
    - `https://indicnlp.blob.core.windows.net/data/classification/news-article/indicnlp-news-article-v1.0.tgz`



## Compiled Classification Datasets

(_Divyanshu, complete this section_)

- Satish's original compilation is available here:
  - `https://github.com/satti007/nltk`
- Evaluation scripts (TBD by Divyanshu):
  - Please include the evaluation scripts in this repo
  - Ensure that the scripts are running
  - may need to rerun the evaluation for some languages since embeddings were updated
  - Anoop: check if Satish already trained with updated data
- Data Download links:
  - Divyanshu: Prepare the data set and share with anoop
  - Anoop: Upload to the above dataset to the below locations
- `https://indicnlp.blob.core.windows.net/data/classification/public-eval_datasets-v1.0.tgz`
- License: available under original license

## Citing

(_Divyanshu, complete this section_)
- If you are using any of the resources , please cite the following

TBD



## License

(_Divyanshu, complete this section_)

Attribution-NonCommercial-ShareAlike
CC BY-NC-SA



## Contributors

(_Divyanshu, complete this section_)
* Anoop Kunchukuttan
* Divyanshu Kakwani
* Satish Golla
* Gokul NC
* Avik Bhattacharyya



## Contact

(_Anoop, complete this section_)
