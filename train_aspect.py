# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train_aspect.py

@time: 2018/4/20 22:12

@desc: train aspect classification models

"""

import os
import logging
import numpy as np
import pickle
import tensorflow as tf
import utils
from read_data import read_data_for_aspect
from models import Classifier


flags = tf.app.flags

# common hyper-parameter
flags.DEFINE_integer('embedding_dim', 300, 'word embedding dimension')
flags.DEFINE_integer('n_epoch', 50, 'max epoch to train')
flags.DEFINE_integer('max_len', 0, 'max length of one sentence')
flags.DEFINE_integer('n_word', 0, 'number of words')
flags.DEFINE_integer('n_class', 0, 'how many classes to predict')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('early_stopping_step', 3, "if loss doesn't descend in 3 epochs, stop training")
flags.DEFINE_integer('train_time', 0, 'train time')
flags.DEFINE_float('stddev', 0.01, 'weight initialization stddev')
flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
flags.DEFINE_boolean('show', True, 'print train progress')
flags.DEFINE_boolean('embed_trainable', True, 'whether word embeddings are trainable')
flags.DEFINE_string('data', './data/train.csv', 'data set file path')
flags.DEFINE_string("vector_file", "./data/embeddings_300_dim.pkl", "pre-trained word vectors file path")
flags.DEFINE_string('model_path', '.', 'path to save model')
flags.DEFINE_string('model_name', 'm', 'model_name')

# hyper-parameter for aspect classification model
flags.DEFINE_string('classifier_type', 'lstm', 'type of classification model: lstm')
flags.DEFINE_integer('hidden_size', 300, "lstm's hidden units")
flags.DEFINE_integer('n_layer', 3, 'number of lstm layers')
flags.DEFINE_boolean('is_multi', True, 'whether to use multi lstm')

FLAGS = flags.FLAGS

if not os.path.exists('log'):
    os.makedirs('log')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename='./log/train.log', filemode='a')


def train_model(data, word_embeddings):
    # 10-fold cross validation
    n_fold = 10
    fold_size = int(len(data[0]) / n_fold)
    
    loss_list, acc_list = [], []
    for i in range(1, n_fold + 1):
        FLAGS.train_time = i
        train_data, valid_data = utils.split_train_valid(data, i, fold_size)
        graph = tf.Graph()
        with tf.Session(graph=graph)as sess:
            model = Classifier(FLAGS, sess)
            model.build_model()
            loss, acc = model.run(train_data, valid_data, word_embeddings)
            loss_list.append(loss)
            acc_list.append(acc)

    avg_loss = np.mean(loss_list)
    avg_acc = np.mean(acc_list)
    print("10fold_loss&acc:", avg_loss, avg_acc)
    print('10fold_std_loss&acc:', np.std(loss_list), np.std(acc_list))
    logging.debug('10fold_loss: ' + str(avg_loss) + '\t10fold_acc :' + str(avg_acc))
    logging.debug('10fold_loss_std: ' + str(np.std(loss_list)) + '\t10fold_acc_std :' + str(np.std(acc_list)))


def main(_):
    pre_trained_vectors = utils.get_gensim_vectors(FLAGS.vector_file)

    data, word_embeddings, word2idx, max_context_len, onehot_mapping = read_data_for_aspect(FLAGS.data,
                                                                                            pre_trained_vectors,
                                                                                            FLAGS.embedding_dim)

    FLAGS.max_len = max_context_len
    FLAGS.n_word = word_embeddings.shape[0]
    FLAGS.model_path = './save_model/classifier/'
    FLAGS.model_name = 'm'
    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)

    FLAGS.n_class = data[-1].shape[1]

    if not os.path.exists(FLAGS.model_path):
        os.mkdir(FLAGS.model_path)

    print('unique words embedding: ', word_embeddings.shape)
    print('max sentence len: ', max_context_len)
    print('n_class : ', FLAGS.n_class)

    train_model(data, word_embeddings)

    save_data = {'onehot_mapping': onehot_mapping, 'word2idx': word2idx, 'max_context_len': max_context_len}
    with open(os.path.join(FLAGS.model_path, 'save_data'), 'wb')as writer:
        pickle.dump(save_data, writer)


if __name__ == '__main__':
    tf.app.run()

