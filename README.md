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

[Aspect-based Financial Sentiment Analysis with Deep Neural Networks](https://dl.acm.org/citation.cfm?id=3191825)
