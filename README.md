# ALA
Attention-based LSTM model with the Aspect information to solve financial opinion mining problem ( [WWW 2018 shared task1](https://sites.google.com/view/fiqa/home) )

## Requirements
- python >=3.5 
- tensorflow 
- numpy
- pickle
- nltk
- gensim

## Preprocessing
run the script to finish preprocessing:
```
sh preprocess.sh
```

## Training
1. aspect classification
```
python train_aspect.py
```
2. sentiment analysis
```
python train_senti.py model_type # eg. python train_senti.py DeepMem, options are DeepMem or AT_LSTM
```

## Paper

[Shijia E. et al. Aspect-based Financial Sentiment Analysis with Deep Neural Networks.](https://dl.acm.org/citation.cfm?id=3191825)

```
@inproceedings{E.:2018:AFS:3184558.3191825,
 author = {E., Shijia and Yang, Li and Zhang, Mohan and Xiang, Yang},
 title = {Aspect-based Financial Sentiment Analysis with Deep Neural Networks},
 booktitle = {Companion Proceedings of the The Web Conference 2018},
 series = {WWW '18},
 year = {2018},
 isbn = {978-1-4503-5640-4},
 location = {Lyon, France},
 pages = {1951--1954},
 numpages = {4},
 url = {https://doi.org/10.1145/3184558.3191825},
 doi = {10.1145/3184558.3191825},
 acmid = {3191825},
 publisher = {International World Wide Web Conferences Steering Committee},
 address = {Republic and Canton of Geneva, Switzerland},
 keywords = {long short-term memory network, representation learning, sentiment analysis},
}
```
