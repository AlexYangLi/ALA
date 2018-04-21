# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train_senti.py

@time: 2018/4/21 0:03

@desc:

"""

import os
import sys
import logging
import numpy as np
import pickle
import tensorflow as tf
import utils
from read_data import read_data_for_senti
from model import DeepMem
from model import AT_LSTM


flags = tf.app.flags

# common hyper-parameter
flags.DEFINE_integer('n_word', 0, 'number of words')
flags.DEFINE_integer('n_aspect', 0, 'number of aspects')
flags.DEFINE_integer('max_len', 0, 'max length of one sentence')
flags.DEFINE_integer('embedding_dim', 300, 'word embedding dimension')
flags.DEFINE_integer('n_epoch', 50, 'max epoch to train')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('early_stopping_step', 3, "if loss doesn't descend in 3 epochs, stop training")
flags.DEFINE_integer('train_time', 0, 'train time')
flags.DEFINE_float('stddev', 0.01, 'weight initialization stddev')
flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
flags.DEFINE_boolean('show', True, 'print train progress')
flags.DEFINE_boolean('embed_trainable', True, 'whether word embeddings are trainable')
flags.DEFINE_boolean('one_aspect', True,
                     'whether consider all texts are related to one aspect (in an abstract level)')
flags.DEFINE_string('model_path', '.', 'path to save model')
flags.DEFINE_string('model_name', 'm', 'model_name')
flags.DEFINE_string('data', './data/train.csv', 'data set file path')
flags.DEFINE_string("vector_file", "./data/embeddings_300_dim.pkl", "pre-trained word vectors file path")

# hyper-parameter for deep memory model
flags.DEFINE_integer('n_hop', 5, 'number of hops')
flags.DEFINE_boolean('use_loc_info', True, 'whether to add location attention')

# hyper-parameter for attention-based & target-dependent lstm model
flags.DEFINE_integer('hidden_size', 300, "lstm's hidden units")
flags.DEFINE_integer('n_layer', 3, 'number of lstm layers')
flags.DEFINE_boolean('is_multi', True, 'whether to use multi lstm')
flags.DEFINE_string('lstm_type', 'at', 'type of lstm model')

FLAGS = flags.FLAGS

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename='./log/train.log', filemode='a')


def train_model(data, word_embeddings, model_type=""):
    # 10-fold cross validation
    n_fold = 10
    fold_size = int(len(data[0]) / n_fold)
    mse_list, r2_list = [], []
    for i in range(1, n_fold + 1):
        FLAGS.train_time = i
        train_data, valid_data = utils.split_train_valid(data, i, fold_size)
        graph = tf.Graph()
        with tf.Session(graph=graph)as sess:
            if model_type == 'DeepMem':
                model = DeepMem(FLAGS, sess)
            elif model_type == 'AT_LSTM':
                model = AT_LSTM(FLAGS, sess)
            else:
                return
            model.build_model()
            mse, r2 = model.run(train_data, valid_data, word_embeddings)
            mse_list.append(mse)
            r2_list.append(r2)

    avg_mse = np.mean(mse_list)
    avg_r2 = np.mean(r2_list)
    print(model_type, "10fold_mse&r2:", avg_mse, avg_r2)
    print(model_type, '10fold_std_mse&r2:', np.std(mse_list), np.std(r2_list))
    logging.debug(model_type + ' 10fold_mse: ' + str(avg_mse) + '\t10fold_r2 :' + str(avg_r2))
    logging.debug(model_type + ' 10fold_mse_std: ' + str(np.std(mse_list)) + '\t10fold_r2_std :' + str(np.std(r2_list)))


def main(model_type):
    pre_trained_vectors = utils.get_gensim_vectors(FLAGS.vector_file)

    data, word_embeddings, word2idx, aspect2idx, max_context_len = read_data(FLAGS.data,
                                                                             pre_trained_vectors,
                                                                             FLAGS.embedding_dim,
                                                                             FLAGS.one_aspect)

    FLAGS.max_len = max_context_len
    FLAGS.n_aspect = len(aspect2idx)
    FLAGS.n_word = word_embeddings.shape[0]
    FLAGS.model_path = os.path.join('./save_model', model_type)
    FLAGS.model_name = 'm'

    if not os.path.exists(FLAGS.model_path):
        os.mkdir(FLAGS.model_path)

    print('unique words embedding: ', word_embeddings.shape)
    print('unique aspect: ', len(aspect2idx))
    print('max sentence len: ', max_context_len)

    train_model(data, word_embeddings, model_type)

    save_data = {'aspect2idx': aspect2idx, 'word2idx': word2idx, 'max_context_len': max_context_len}
    with open(os.path.join(FLAGS.model_path, 'save_data'), 'wb')as writer:
        pickle.dump(save_data, writer)


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'argument error!'
    main(sys.argv[1])
