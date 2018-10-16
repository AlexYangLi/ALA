# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: utils.py

@time: 2018/4/20 21:36

@desc:

"""

import os
import numpy as np
import codecs
import pickle


def get_glove_vectors(vector_file):
    pre_trained_vectors = {}
    with codecs.open(vector_file, 'r', encoding='utf8')as reader:
        for line in reader:
            content = line.strip().split()
            pre_trained_vectors[content[0]] = np.array(list(map(float, content[1:])))
    return pre_trained_vectors


def get_gensim_vectors(vector_file):
    with codecs.open(vector_file, 'rb')as reader:
        pre_train_vectors = pickle.load(reader)
    return pre_train_vectors


def split_train_valid(data, fold_index, fold_size):
    train = []
    valid = []
    for data_item in data:
        valid.append(data_item[(fold_index-1)*fold_size: fold_index*fold_size])
        train.append(np.concatenate([data_item[:(fold_index-1)*fold_size],
                                    data_item[fold_index*fold_size:]]))

    return train, valid


def batch_index(length, batch_size, n_iter=100):
    index = range(length)
    for j in range(n_iter):
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


def onehot_encoding(y):
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32), y_onehot_mapping


def only_save_best_epoch(model_path, model_name, best_epoch, train_time):
    data_suffix = '.data-00000-of-00001'
    data_name = model_name + '-' + str(best_epoch) + data_suffix
    new_data_name = str(train_time) + 'fold-model.data-00000-of-00001'
    new_data_path = os.path.join(model_path, new_data_name)
    if os.path.exists(new_data_path):
        os.remove(new_data_path)

    index_suffix = '.index'
    index_name = model_name + '-' + str(best_epoch) + index_suffix
    new_index_name = str(train_time) + 'fold-model.index'
    new_index_path = os.path.join(model_path, new_index_name)
    if os.path.exists(new_index_path):
        os.remove(new_index_path)

    meta_suffix = '.meta'
    meta_name = model_name + '-' + str(best_epoch) + meta_suffix
    new_meta_name = str(train_time) + 'fold-model.meta'
    new_meta_path = os.path.join(model_path, new_meta_name)
    if os.path.exists(new_meta_path):
        os.remove(new_meta_pacdth)

    for file in os.listdir(model_path):
        if file == data_name:
            os.rename(os.path.join(model_path, data_name), new_data_path)
        elif file == index_name:
            os.rename(os.path.join(model_path, index_name), new_index_path)
        elif file == meta_name:
            os.rename(os.path.join(model_path, meta_name), new_meta_path)
        elif file[:len(model_name)] == model_name \
                and (file[-1 * len(data_suffix):] == data_suffix or file[-1 * len(index_suffix):] == index_suffix or file[-1 * len(meta_suffix):] == meta_suffix):
            os.remove(os.path.join(model_path, file))
