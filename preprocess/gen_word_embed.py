# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: gen_word_embed.py

@time: 2018/4/20 22:15

@desc: use gensim to train a word2vec model

"""

import os
import pickle
import codecs
import nltk
from gensim.models import Word2Vec


def load(path, name):
    return pickle.load(open(os.path.join(path, name), 'r'))


def word2vec(fname, save_file):
    sentences = []
    with codecs.open(fname, encoding='utf8') as sentence_file:
        lines = sentence_file.readlines()
        for line in lines:
            line_items = line.strip().split('\t')
            sentence = line_items[0]
            sentences.append(nltk.word_tokenize(sentence))

    model = Word2Vec(sentences, size=300, min_count=1, window=5, sg=1, iter=10)

    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    embeddings_index = {}

    for item in d:
        embeddings_index[item] = weights[d[item], :]

    pickle.dump(embeddings_index, open(save_file, 'wb'))


if __name__ == '__main__':
    word2vec('../data/train.tsv', '../data/embeddings_300_dim.pkl')
