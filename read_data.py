# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: read_data.py

@time: 2018/4/21 8:06

@desc:

"""

import numpy as np
import pandas as pd
import nltk
from utils import onehot_encoding


def isinteger(_str):
    return _str.strip().isdigit()


def isfloat(_str):
    return sum([n.isdigit() for n in _str.strip().split('.')]) == 2


def get_index(_list, item, start_index=0):
    for i in range(start_index, len(_list)):
        if item == _list[i]:
            return i
    raise ("can not find %s in %s" % (item, list))


def read_data_for_aspect(fname, pre_trained_vectors, embedding_size):

    data_csv = pd.read_csv(fname, sep='\t', header=None, index_col=None,
                           names=['text', 'aspect_l1', 'aspect_l2', 'score'])

    data_csv = data_csv.sample(frac=1).reset_index(drop=True)   # shuffle data

    stop_words = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '-', '/', '&', '``', "''"]

    max_context_len = 0
    word2idx = {}    # Index 0 represents words we haven't met before
    context = []
    context_len = []
    aspect_class = []

    for index, row in data_csv.iterrows():
        word_list = nltk.word_tokenize(row.text.strip())

        context_words = [word for word in word_list
                         if word not in stop_words and isinteger(word) is False and isfloat(word) is False
                         and word != 'T']

        words_have_vector = [word for word in context_words if word in pre_trained_vectors]

        # make sure most words can find their embedding vectors
        if len(words_have_vector) / float(len(context_words)) < 0.8:
            continue

        max_context_len = max(max_context_len, len(words_have_vector))

        idx = []
        for word in words_have_vector:
            if word not in word2idx:
                word2idx[word] = len(word2idx)+1    # Index 0 represents absent words, so start from 1
            idx.append(word2idx[word])

        context.append(idx)
        context_len.append(len(words_have_vector))

        aspect_class.append(row.aspect_l1 + '/' + row.aspect_l2)

    # convert to numpy format
    context_npy = np.zeros(shape=[len(context), max_context_len])
    for i in range(len(context)):
        context_npy[i, :len(context[i])] = context[i]

    aspect_class_npy, onehot_mapping = onehot_encoding(aspect_class)

    train_data = list()
    train_data.append(context_npy)          # [data_size, max_context_len]
    train_data.append(np.array(context_len))    # [data_size,]
    train_data.append(aspect_class_npy)     # [data_size, aspect_class]

    word_embeddings = np.zeros([len(word2idx)+1, embedding_size])
    for word in word2idx.keys():
        word_embeddings[word2idx[word]] = pre_trained_vectors[word]

    return train_data, word_embeddings, word2idx, max_context_len, onehot_mapping


def read_data_for_senti(fname, pre_trained_vectors, embedding_size, one_aspect):

    data_csv = pd.read_csv(fname, sep='\t', header=None, index_col=None,
                           names=['text', 'aspect_l1', 'aspect_l2', 'score'])

    data_csv = data_csv.sample(frac=1).reset_index(drop=True)   # shuffle data

    stop_words = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '-', '/', '&', '``', "''"]

    max_context_len = 0
    word2idx = {}    # Index 0 represents words we haven't met before
    aspect2idx = {}
    context = []
    context_len = []
    loc_info = []
    aspect = []
    score = []
    # aspect_class = []

    for index, row in data_csv.iterrows():
        word_list = nltk.word_tokenize(row.text.strip())

        context_words = [word for word in word_list
                         if word not in stop_words and isinteger(word) is False and isfloat(word) is False
                         and word != 'T']

        words_have_vector = [word for word in context_words if word in pre_trained_vectors]

        # make sure most words can find their embedding vectors
        if len(words_have_vector) / float(len(context_words)) < 0.8:
            continue

        max_context_len = max(max_context_len, len(words_have_vector))

        idx, distance = [], []
        start_index = 0
        stock_loc = get_index(word_list, 'T')
        for word in words_have_vector:
            if word not in word2idx:
                word2idx[word] = len(word2idx)+1    # Index 0 represents absent words, so start from 1
            idx.append(word2idx[word])
            word_loc = get_index(word_list, word, start_index=start_index)
            start_index = word_loc + 1
            distance.append(1 - abs(word_loc-stock_loc) / len(word_list))
        context.append(idx)
        context_len.append(len(words_have_vector))
        loc_info.append(distance)

        if one_aspect is True:      # consider there is only one abstract aspect
            aspect.append(0)
        else:
            if row.aspect_l2 not in aspect2idx:
                aspect2idx[row.aspect_l2] = len(aspect2idx)
            aspect.append(aspect2idx[row.aspect_l2])

        score.append([row.score])

    if one_aspect is True:
        aspect2idx['one_aspect'] = len(aspect2idx)

    # convert to numpy format
    context_npy = np.zeros(shape=[len(context), max_context_len])
    loc_info_npy = np.zeros(shape=[len(loc_info), max_context_len])

    for i in range(len(context)):
        context_npy[i, :len(context[i])] = context[i]
        loc_info_npy[i, :len(loc_info[i])] = loc_info[i]

    train_data = list()
    train_data.append(context_npy)          # [data_size, max_context_len]
    train_data.append(np.array(context_len))    # [data_size,]
    train_data.append(loc_info_npy)         # [data_size, max_context_len]
    train_data.append(np.array(aspect))     # [data_size]
    train_data.append(np.array(score))      # [data_size, 1]

    word_embeddings = np.zeros([len(word2idx)+1, embedding_size])
    for word in word2idx.keys():
        word_embeddings[word2idx[word]] = pre_trained_vectors[word]

    return train_data, word_embeddings, word2idx, aspect2idx, max_context_len
