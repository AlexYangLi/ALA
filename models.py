# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: models.py

@time: 2018/4/20 21:44

@desc: neural models for aspect classification, sentiment analysis

"""

import os
import logging
import tensorflow as tf
from utils import only_save_best_epoch


# aspect classification model
class Classifier(object):
    def __init__(self, config, sess):
        self.edim = config.embedding_dim
        self.n_epoch = config.n_epoch
        self.batch_size = config.batch_size
        self.n_class = config.n_class
        self.n_layer = config.n_layer
        self.early_stopping_step = config.early_stopping_step
        self.stddev = config.stddev
        self.hidden_size = config.hidden_size
        self.n_word = config.n_word
        self.max_len = config.max_len
        self.classifier_type = config.classifier_type
        self.l2_reg = config.l2_reg
        self.show = config.show
        self.embed_trainable = config.embed_trainable
        self.is_multi = config.is_multi
        self.train_time = config.train_time
        self.save_model_path = config.model_path
        self.save_model_name = config.model_name

        self.sess = sess

        # input
        self.context = tf.placeholder(shape=[None, self.max_len], dtype=tf.int32, name='context')
        self.aspect_class = tf.placeholder(shape=[None, self.n_class], dtype=tf.int32, name='aspect_class')
        self.context_len = tf.placeholder(dtype=tf.int32, name='context_len')
        self.keep_prob = tf.placeholder(dtype=tf.float32)

        # word embeddings will be optimized if trainable == True
        self.word_embeddings = tf.Variable(tf.random_uniform([self.n_word, self.edim], -1.0, 1.0),
                                           name='word_embeddings', trainable=self.embed_trainable)

        # batch context embedding & aspect embedding
        self.context_embed = tf.nn.embedding_lookup(self.word_embeddings, self.context,
                                                    name='context_embed')  # [batch_size, max_len, edim]

        # Weights & Bias
        self.W_hidden = tf.Variable(tf.truncated_normal([self.hidden_size, 128], stddev=self.stddev), name='W_hidden')
        self.B_hidden = tf.Variable(tf.constant(0.1, shape=[128]), name='B_hidden')

        self.W_final = tf.Variable(tf.truncated_normal([128, self.n_class], stddev=self.stddev), name='W_final')

        self.B_final = tf.Variable(tf.constant(0.1, shape=[self.n_class]), name='B_final')

        self.predict = None
        self.loss = None
        self.accuracy = None
        self.train_step = None

        self.best_loss = 128.0
        self.best_epoch = 0
        self.best_accuracy = 0.0
        self.stopping_step = 0

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S', filename='./data/LSTM_Classifierwatch.log', filemode='a')

    def multi_lstm(self):
        batch_size = tf.shape(self.context_embed)[0]
        if self.is_multi:
            stacked_lstm = []
            for i in range(self.n_layer):
                stacked_lstm.append(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size,
                                                                                          state_is_tuple=True),
                                                                  output_keep_prob=self.keep_prob))
            multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_lstm, state_is_tuple=True)
        else:
            multi_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size,
                                                                                    state_is_tuple=True),
                                                            output_keep_prob=self.keep_prob)
        initial_state = multi_lstm_cell.zero_state(batch_size, dtype=tf.float32)
        return multi_lstm_cell, initial_state

    def dynamic_rnn(self, lstm_cell, inputs, context_len, initial_state, max_len, out_type='all'):
        outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                           inputs=inputs,
                                           sequence_length=context_len,
                                           initial_state=initial_state,
                                           dtype=tf.float32)  # [batch_size, max_len, hidden_size]
        batch_size = tf.shape(outputs)[0]
        if out_type == 'last':
            indexs = tf.range(0, batch_size) * max_len + (context_len - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_size]), indexs)  # [batch_size, hidden_size]
        elif outputs == 'all_avg':
            outputs = AT_LSTM.reduce_mean(outputs, context_len)  # [batch_size, hidden_size]
        return outputs

    def lstm_model(self):
        multi_lstm_cell, initial_state = self.multi_lstm()
        lstm_output = self.dynamic_rnn(multi_lstm_cell, self.context_embed, self.context_len, initial_state,
                                       self.max_len, out_type='last')  # [batch_size, hidden]
        hidden_output = tf.matmul(lstm_output, self.W_hidden) + self.B_hidden
        predict = tf.nn.softmax(tf.matmul(hidden_output, self.W_final) + self.B_final)
        return predict

    def cnn_model(self):
        conv_filter_w = tf.Variable(tf.truncated_normal([3, self.edim, 64],
                                                        stddev=self.stddev), name='W_hidden')
        conv_filter_b = tf.Variable(tf.constant(0.1, shape=[64]))
        conv_output = tf.nn.conv1d(self.context_embed, conv_filter_w, stride=2, padding='SAME')
        relu_output = tf.nn.relu(conv_output + conv_filter_b)

        batch_size = tf.shape(relu_output)[0]
        flatten = tf.reshape(relu_output, [batch_size, -1])

        W_hidden = tf.Variable(tf.truncated_normal([14 * 64, 128], stddev=self.stddev), name='W_hidden_cnn')
        B_hidden = tf.Variable(tf.constant(0.1, shape=[128]), name='B_hidden_cnn')
        hidden_output = tf.matmul(flatten, W_hidden) + B_hidden
        predict = tf.nn.softmax(tf.matmul(hidden_output, self.W_final) + self.B_final)
        return predict

    def build_model(self):
        if self.classifier_type == 'lstm':
            self.predict = self.lstm_model()
        elif self.classifier_type == 'cnn':
            self.predict = self.cnn_model()
        else:
            self.predict = self.lstm_model()
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.aspect_class, logits=self.predict))
        # self.loss = - tf.reduce_mean(tf.cast(self.polarity, tf.float32) * tf.log(self.predict))
        correct_prediction = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.aspect_class, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, train_data, valid_data, word_embeddings):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.word_embeddings.assign(word_embeddings))
        saver = tf.train.Saver(max_to_keep=self.n_epoch)

        context, context_len, aspect_class = train_data
        loops = int(len(context) / self.batch_size)
        for epoch in range(self.n_epoch):
            avg_loss, avg_accuracy = 0.0, 0.0
            for i in range(loops):
                feed_dict = {self.context: context[i * self.batch_size: (i + 1) * self.batch_size],
                             self.context_len: context_len[i * self.batch_size: (i + 1) * self.batch_size],
                             self.aspect_class: aspect_class[i * self.batch_size: (i + 1) * self.batch_size],
                             self.keep_prob: 0.5}
                _, loss, accuracy = self.sess.run([self.train_step, self.loss, self.accuracy], feed_dict=feed_dict)
                avg_loss += loss
                avg_accuracy += accuracy

            avg_loss /= loops
            avg_accuracy /= loops

            logging.debug(self.classifier_type + ' ' + str(epoch) + ' train_loss: ' +
                          str(avg_loss) + '\ttrain_accuracy :' + str(avg_accuracy))
            if self.show:
                print(self.classifier_type, ' epoch:', epoch, 'train_loss:', avg_loss, 'train_accuracy:', avg_accuracy)

            saver.save(self.sess, os.path.join(self.save_model_path + self.save_model_name), global_step=epoch)

            valid_loss, valid_accuracy = self.valid(valid_data)

            if valid_accuracy > self.best_accuracy:
                self.best_loss = valid_loss
                self.best_accuracy = valid_accuracy
                self.best_epoch = epoch
                self.stopping_step = 0
            else:
                self.stopping_step += 1
            if self.stopping_step >= self.early_stopping_step:
                logging.debug(self.classifier_type + ' early stopping is trigger at epoch: ' + str(epoch))
                if self.show:
                    print(self.classifier_type, 'early stopping is trigger at epoch: ', epoch)
                break

        if self.show:
            print(self.classifier_type, 'best epoch:', self.best_epoch, 'best loss:', self.best_loss,
                  'best accuracy:', self.best_accuracy)
        logging.debug(self.classifier_type + ' best epoch: ' + str(self.best_epoch) +
                      '\tbest loss: ' + str(self.best_loss) + '\tbest accuracy :' + str(self.best_accuracy))

        only_save_best_epoch(self.save_model_path, self.save_model_name, self.best_epoch, self.train_time)

        return self.best_loss, self.best_accuracy

    def valid(self, valid_data):
        context, context_len, aspect_class = valid_data
        feed_dict = {self.context: context,
                     self.context_len: context_len,
                     self.aspect_class: aspect_class,
                     self.keep_prob: 1.0}
        loss, accuracy = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        logging.debug(self.classifier_type + ' valid_loss: ' + str(loss) + '\tvalid_accuracy :' + str(accuracy))
        if self.show:
            print(self.classifier_type, ' valid_loss:', loss, 'valid_accuracy:', accuracy)
        return loss, accuracy

    def run(self, train_data, valid_data, word_embeddings):
        loss, accuracy = self.train(train_data, valid_data, word_embeddings)
        return loss, accuracy


# deep memory network
class DeepMem(object):
    def __init__(self, config, sess):
        self.edim = config.embedding_dim
        self.n_epoch = config.n_epoch
        self.early_stopping_step = config.early_stopping_step
        self.batch_size = config.batch_size
        self.n_hop = config.n_hop
        self.n_word = config.n_word
        self.n_aspect = config.n_aspect
        self.stddev = config.stddev
        self.l2_reg = config.l2_reg
        self.max_len = config.max_len
        self.show = config.show
        self.embed_trainable = config.embed_trainable
        self.use_loc_info = config.use_loc_info
        self.train_time = config.train_time
        self.save_model_path = config.model_path
        self.save_model_name = config.model_name

        self.sess = sess

        self.context = tf.placeholder(shape=[None, self.max_len], dtype=tf.int32, name='context')
        self.aspect = tf.placeholder(dtype=tf.int32, name='aspect')
        self.loc_info = tf.placeholder(shape=[None, self.max_len], dtype=tf.float32, name='loc_info')
        self.score = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='score')

        self.word_embeddings = tf.Variable(tf.random_uniform([self.n_word, self.edim], -1.0, 1.0),
                                           name='word_embeddings', trainable=self.embed_trainable)
        self.aspect_embeddings = tf.Variable(tf.random_uniform([self.n_aspect, self.edim], -1.0, 1.0),
                                             name='aspect_embedding')

        self.context_embed = tf.nn.embedding_lookup(self.word_embeddings, self.context,
                                                    name='context_embed')
        self.aspect_embed = tf.nn.embedding_lookup(self.aspect_embeddings, self.aspect,
                                                   name='aspect_embed')

        # weight responding to attention model, which is shared among different layers
        self.w_att = tf.Variable(tf.truncated_normal([2 * self.edim, 1], stddev=self.stddev), name='w_att')

        # bias responding to attention model, which is shared among different layers
        self.b_att = tf.Variable(tf.constant(0.1, shape=[1, 1], name='b_att'))

        # weight to linear transformation aspect, which is shared among different layers
        self.w_linear = tf.Variable(tf.truncated_normal([self.edim, self.edim], stddev=self.stddev), name='w_linear')

        self.w_final = tf.Variable(tf.truncated_normal([self.edim, 1], stddev=self.stddev), name='w_final')

        self.layer = []  # computational layers' output
        self.mse = None
        self.r2 = None
        self.train_step = None

        self.best_mse = 4.0
        self.best_epoch = 0
        self.best_r2 = 0.0
        self.stopping_step = 0

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S', filename='./data/DeepMem_watch.log', filemode='a')

    def build_memory(self):

        # For every given sentences and aspect word, we map each word into its embedding vector.
        # These word vectors are separated into tow parts, aspect representation and context representation.
        # As for context representation, we use pre-trained word vectors.
        # As for aspect representation, we initialized it randomly and optimized along with network training.
        # Noted that all aspects share one embedding vector.

        batch_size = tf.shape(self.context_embed)[0]

        if self.use_loc_info:
            loc_attention = tf.tile(tf.expand_dims(self.loc_info, 2),
                                    [1, 1, self.edim], name='loc_attention')  # [batch_size, max_len, edim]
            self.context_embed = self.context_embed * loc_attention  # [batch_size, max_len, edim]

        # Multiple hops to learn representation of text with multiple levels of abstraction
        self.layer.append(self.aspect_embed)
        for h in range(self.n_hop):
            aspect_embed_tile = tf.reshape(tf.tile(self.layer[-1], [1, self.max_len]), [-1, self.max_len, self.edim],
                                           name='aspect_embed_tile')  # [batch_size, max_len, edim]
            w_att_tile = tf.reshape(tf.tile(self.w_att, [batch_size, 1]), [batch_size, 2 * self.edim, 1],
                                    name='W_att_tile')  # [batch_size, 2*edim, 1]
            b_att_tile = tf.reshape(tf.tile(self.b_att, [batch_size, self.max_len]), [batch_size, self.max_len, 1],
                                    name='b_att_tile')  # [batch_size, max_len. 1]
            att = tf.matmul(tf.concat([self.context_embed, aspect_embed_tile], axis=2), w_att_tile,
                            name='W_mul_mem')  # [batch_size, max_len, 1]

            att_score = tf.nn.tanh(tf.add(att, b_att_tile), name='attention_score')  # [batch_size, max_len, 1]
            att_score2dim = tf.reshape(att_score, shape=[-1, self.max_len])  # [batch_size, mem_sie]

            prob = tf.nn.softmax(att_score2dim)  # [batch_size, max_len]
            prob_2dim = tf.reshape(prob, [-1, 1, self.max_len])  # [batch_size, 1, max_len]

            # apply attention to context
            weighted_sum = tf.matmul(prob_2dim, self.context_embed, name='weighted_sum')  # [batch_size, 1, edim]
            context_att = tf.reshape(weighted_sum, [-1, self.edim], name='context_att')  # [batch_size, edim]

            # apply linear transformation to aspect
            aspect_transformed = tf.matmul(self.layer[-1], self.w_linear,
                                           name='aspect_transformed')  # [batch_size, edim]

            # generate a new representation of text
            text = tf.add(context_att, aspect_transformed)
            self.layer.append(text)

    def build_model(self):
        self.build_memory()

        # We regard the output vector in last hop as text final representation(feature)
        # Feed it into a dense layer for aspect level sentiment prediction

        predict = tf.tanh(tf.matmul(self.layer[-1], self.w_final), name='predict')  # [batch_size, 1]

        self.mse = tf.reduce_mean(tf.square(self.score - predict))

        score_mean = tf.reduce_mean(self.score)
        self.r2 = 1 - tf.reduce_sum(tf.square(self.score - predict)) / tf.reduce_sum(tf.square(self.score - score_mean))

        self.train_step = tf.train.AdamOptimizer().minimize(self.mse)

    def train(self, train_data, valid_data, word_embeddings):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.word_embeddings.assign(word_embeddings))
        saver = tf.train.Saver(max_to_keep=self.n_epoch)
        context, context_len, loc_info, aspect, score = train_data

        loops = int(len(context) / self.batch_size)
        for epoch in range(self.n_epoch):
            avg_mse, avg_r2 = 0.0, 0.0
            for i in range(loops):
                feed_dict = {self.context: context[i * self.batch_size: (i + 1) * self.batch_size],
                             self.loc_info: loc_info[i * self.batch_size: (i + 1) * self.batch_size],
                             self.aspect: aspect[i * self.batch_size: (i + 1) * self.batch_size],
                             self.score: score[i * self.batch_size: (i + 1) * self.batch_size]}
                _, mse, r2 = self.sess.run([self.train_step, self.mse, self.r2], feed_dict=feed_dict)

                avg_mse += mse
                avg_r2 += r2

            avg_mse /= loops
            avg_r2 /= loops
            logging.debug('deepmem ' + str(epoch) + ' train_mse: ' + str(avg_mse) + '\ttrain_r2 :' + str(avg_r2))

            if self.show:
                print('deepmem epoch:', epoch, 'train_mse:', avg_mse, 'train_r2:', avg_r2)

            saver.save(self.sess, os.path.join(self.save_model_path + self.save_model_name), global_step=epoch)

            valid_mse, valid_r2 = self.valid(valid_data)
            if valid_mse < self.best_mse:
                self.best_mse = valid_mse
                self.best_r2 = valid_r2
                self.best_epoch = epoch
                self.stopping_step = 0
            else:
                self.stopping_step += 1
            if self.stopping_step >= self.early_stopping_step:
                logging.debug('deepmem early stopping is trigger at epoch: ' + str(epoch))
                if self.show:
                    print('deepmem early stopping is trigger at epoch: ', epoch)
                break

        only_save_best_epoch(self.save_model_path, self.save_model_name, self.best_epoch, self.train_time)

        if self.show:
            print('deepmem best epoch:', self.best_epoch, 'best mse:', self.best_mse, 'best r2:', self.best_r2)
        logging.debug('deepmem ' + 'best epoch: ' + str(self.best_epoch) +
                      '\tbest mse: ' + str(self.best_mse) + '\tbest r2 :' + str(self.best_r2))

        return self.best_mse, self.best_r2

    def valid(self, valid_data):
        context, context_len, loc_info, aspect, score = valid_data

        feed_dict = {self.context: context,
                     self.loc_info: loc_info,
                     self.aspect: aspect,
                     self.score: score}

        mse, r2 = self.sess.run([self.mse, self.r2], feed_dict=feed_dict)

        logging.debug('deepmem valid_mse: ' + str(mse) + '\tvalid_r2 :' + str(r2))

        if self.show:
            print('deepmem valid_mse:', mse, 'valid_r2:', r2)

        return mse, r2

    def run(self, train_data, valid_data, word_embeddings):
        mse, r2 = self.train(train_data, valid_data, word_embeddings)
        return mse, r2


# attention-based lstm
class AT_LSTM(object):
    def __init__(self, config, sess):
        self.edim = config.embedding_dim
        self.n_epoch = config.n_epoch
        self.batch_size = config.batch_size
        self.n_aspect = config.n_aspect
        self.n_layer = config.n_layer
        self.early_stopping_step = config.early_stopping_step
        self.stddev = config.stddev
        self.hidden_size = config.hidden_size
        self.n_word = config.n_word
        self.max_len = config.max_len
        self.lstm_type = config.lstm_type
        self.l2_reg = config.l2_reg
        self.show = config.show
        self.embed_trainable = config.embed_trainable
        self.is_multi = config.is_multi
        self.save_model_path = config.model_path
        self.save_model_name = config.model_name
        self.train_time = config.train_time

        self.sess = sess

        # input
        self.context = tf.placeholder(shape=[None, self.max_len], dtype=tf.int32, name='context')
        self.aspect = tf.placeholder(dtype=tf.int32, name='aspect')
        self.score = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='score')
        self.context_len = tf.placeholder(dtype=tf.int32, name='context_len')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        # aspect embeddings will be optimized along with training process
        # word embeddings will be optimized if trainable == True
        self.word_embeddings = tf.Variable(tf.random_uniform([self.n_word, self.edim], -1.0, 1.0),
                                           name='word_embeddings', trainable=self.embed_trainable)
        self.aspect_embeddings = tf.Variable(tf.random_uniform([self.n_aspect, self.edim], -1.0, 1.0),
                                             name='aspect_embeddings')

        # batch context embedding & aspect embedding
        self.context_embed = tf.nn.embedding_lookup(self.word_embeddings, self.context,
                                                    name='context_embed')  # [batch_size, max_len, edim]
        self.aspect_embed = tf.nn.embedding_lookup(self.aspect_embeddings, self.aspect,
                                                   name='aspect_embed')  # [batch_size, edim]

        # Weights
        self.W_concat = tf.Variable(tf.truncated_normal([self.edim + self.hidden_size, self.edim + self.hidden_size],
                                                        stddev=self.stddev), name='W_concat')
        self.W_softmax = tf.Variable(tf.truncated_normal([self.edim + self.hidden_size, 1], stddev=self.stddev),
                                     name='W_softmax')
        self.Wp = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size], stddev=self.stddev),
                              name='Wp')
        self.Wx = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size], stddev=self.stddev),
                              name='Wx')
        self.W_final = tf.Variable(tf.truncated_normal([self.hidden_size, 1], stddev=self.stddev), name='W_final')

        self.predict = None
        self.mse = None
        self.r2 = None
        self.train_step = None

        self.best_mse = 4.0
        self.best_epoch = 0
        self.best_r2 = 0.0
        self.stopping_step = 0

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S', filename='./data/AT_LSTM_watch.log', filemode='a')

    def multi_lstm(self):
        batch_size = tf.shape(self.context_embed)[0]

        if self.is_multi:
            stacked_lstm = []
            for i in range(self.n_layer):
                stacked_lstm.append(tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size,
                                                                                          state_is_tuple=True),
                                                                  output_keep_prob=self.keep_prob))
            multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_lstm, state_is_tuple=True)
        else:
            multi_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.hidden_size,
                                                                                    state_is_tuple=True),
                                                            output_keep_prob=self.keep_prob)

        initial_state = multi_lstm_cell.zero_state(batch_size, dtype=tf.float32)

        return multi_lstm_cell, initial_state

    def dynamic_rnn(self, lstm_cell, inputs, context_len, initial_state, max_len, out_type='all'):
        outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                           inputs=inputs,
                                           sequence_length=context_len,
                                           initial_state=initial_state,
                                           dtype=tf.float32)  # [batch_size, max_len, hidden_size]

        batch_size = tf.shape(outputs)[0]
        if out_type == 'last':
            indexs = tf.range(0, batch_size) * max_len + (context_len - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_size]), indexs)  # [batch_size, hidden_size]
        elif outputs == 'all_avg':
            outputs = AT_LSTM.reduce_mean(outputs, context_len)  # [batch_size, hidden_size]
        return outputs

    def cnn_model(self):
        conv_filter_w = tf.Variable(tf.truncated_normal([3, self.edim, 64],
                                                        stddev=self.stddev), name='W_hidden')
        conv_filter_b = tf.Variable(tf.constant(0.1, shape=[64]))
        conv_output = tf.nn.conv1d(self.context_embed, conv_filter_w, stride=1, padding='SAME')
        relu_output = tf.nn.relu(conv_output + conv_filter_b)

        relu_output = tf.expand_dims(relu_output, 1)
        max_pool = tf.nn.max_pool(relu_output, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        batch_size = tf.shape(max_pool)[0]
        flatten = tf.reshape(max_pool, [batch_size, -1])

        W_hidden = tf.Variable(tf.truncated_normal([9 * 64, 128], stddev=self.stddev), name='W_hidden_cnn')
        B_hidden = tf.Variable(tf.constant(0.1, shape=[128]), name='B_hidden_cnn')
        hidden_output = tf.matmul(flatten, W_hidden) + B_hidden

        W_final = tf.Variable(tf.truncated_normal([128, 1], stddev=self.stddev), name='W_final')
        predict = tf.matmul(hidden_output, W_final, name='predict')
        return predict

    # orignal lstm model
    def original_model(self):
        multi_lstm_cell, initial_state = self.multi_lstm()
        lstm_output = self.dynamic_rnn(multi_lstm_cell, self.context_embed, self.context_len, initial_state,
                                       self.max_len, out_type='last')  # [batch_size, hidden]

        predict = tf.matmul(lstm_output, self.W_final, name='predict')

        return predict

    # at_model: use attention mechanism
    # concatenate the aspect vector into the sentence hidden representations for computing attention weights
    def at_model(self):
        batch_size = tf.shape(self.context_embed)[0]
        aspect_expand = tf.tile(self.aspect_embed, [1, self.max_len])
        aspect_expand = tf.reshape(aspect_expand, [-1, self.max_len, self.edim],
                                   name='aspect_expand')  # [batch_size, max_len, edim]

        multi_lstm_cell, initial_state = self.multi_lstm()
        lstm_output = self.dynamic_rnn(multi_lstm_cell, self.context_embed, self.context_len, initial_state,
                                       self.max_len, out_type='all')  # [batch_size, max_len, hidden]

        w_concat_tile = tf.reshape(tf.tile(self.W_concat, [batch_size, 1]),
                                   [batch_size, self.edim + self.hidden_size, -1],
                                   name='w_concat_title')  # [batch_size, edim+hidden_size, edim+hidden_size]
        M = tf.nn.tanh(tf.matmul(tf.concat([lstm_output, aspect_expand], axis=2), w_concat_tile),
                       name='M')  # [batch_size, max_len, edim+hidden_size]
        w_softmax_tile = tf.reshape(tf.tile(self.W_softmax, [batch_size, 1]),
                                    [batch_size, -1, 1])  # [batch_size, edim+hidden_size, 1]
        w_mul_M = tf.reshape(tf.matmul(M, w_softmax_tile), [batch_size, 1, -1],
                             name='w_mul_M')  # [batch_size, 1, max_len]

        alpha = AT_LSTM.softmax(w_mul_M, self.context_len, self.max_len)  # [batch_size, 1, max_len)

        r = tf.reshape(tf.matmul(alpha, lstm_output), [batch_size, self.hidden_size],
                       name='r')  # weighted representation of sentence with given aspect: [batch_size, hidden_size]

        # get last output of lstm
        index = tf.range(batch_size) * self.max_len + (self.context_len - 1)
        lstm_last = tf.gather(tf.reshape(lstm_output, [-1, self.hidden_size]), index)  # [batch_size, hidden_size]

        h = tf.tanh(tf.matmul(r, self.Wp) + tf.matmul(lstm_last, self.Wx))  # final sentence representation

        predict = tf.matmul(h, self.W_final, name='predict')

        return predict

    # ae_model: additionally append the aspect vector into the input word vectors
    def ae_model(self):
        aspect_expand = tf.tile(self.aspect_embed, [1, self.max_len])
        aspect_expand = tf.reshape(aspect_expand, [-1, self.max_len, self.edim],
                                   name='aspect_expand')  # [batch_size, max_len, edim]
        inputs = tf.concat([self.context_embed, aspect_expand], axis=2, name='inputs')

        multi_lstm_cell, initial_state = self.multi_lstm()
        lstm_output = self.dynamic_rnn(multi_lstm_cell, inputs, self.context_len, initial_state, self.max_len,
                                       out_type='last')  # [batch_size, hidden]

        predict = tf.matmul(lstm_output, self.W_final, name='predict')

        return predict

    # atae_model: attention mechanism & aspect vector as input
    def atae_model(self):
        batch_size = tf.shape(self.context_embed)[0]
        aspect_expand = tf.tile(self.aspect_embed, [1, self.max_len])
        aspect_expand = tf.reshape(aspect_expand, [-1, self.max_len, self.edim],
                                   name='aspect_expand')  # [batch_size, max_len, edim]
        inputs = tf.concat([self.context_embed, aspect_expand], axis=2, name='inputs')

        multi_lstm_cell, initial_state = self.multi_lstm()
        lstm_output = self.dynamic_rnn(multi_lstm_cell, inputs, self.context_len, initial_state, self.max_len,
                                       out_type='all')  # [batch_size, max_len, hidden]

        w_concat_tile = tf.reshape(tf.tile(self.W_concat, [batch_size, 1]),
                                   [batch_size, self.edim + self.hidden_size, -1],
                                   name='w_concat_title')  # [batch_size, edim+hidden_size, edim+hidden_size]
        M = tf.nn.tanh(tf.matmul(tf.concat([lstm_output, aspect_expand], axis=2), w_concat_tile),
                       name='M')  # [batch_size, max_len, edim+hidden_size]
        w_softmax_tile = tf.reshape(tf.tile(self.W_softmax, [batch_size, 1]),
                                    [batch_size, -1, 1])  # [batch_size, edim+hidden_size, 1]
        w_mul_M = tf.reshape(tf.matmul(M, w_softmax_tile), [batch_size, 1, -1],
                             name='w_mul_M')  # [batch_size, 1, max_len]

        alpha = AT_LSTM.softmax(w_mul_M, self.context_len, self.max_len)  # [batch_size, 1, max_len)

        r = tf.reshape(tf.matmul(alpha, lstm_output), [batch_size, self.hidden_size],
                       name='r')  # weighted representation of sentence with given aspect: [batch_size, hidden_size]

        # get last output of lstm
        index = tf.range(batch_size) * self.max_len + (self.context_len - 1)
        lstm_last = tf.gather(tf.reshape(lstm_output, [-1, self.hidden_size]), index)  # [batch_size, hidden_size]

        h = tf.tanh(tf.matmul(r, self.Wp) + tf.matmul(lstm_last, self.Wx))  # final sentence representation

        predict = tf.matmul(h, self.W_final, name='predict')

        return predict

    def build_model(self):
        if self.lstm_type == 'at':
            self.predict = self.at_model()
        elif self.lstm_type == 'ae':
            self.predict = self.ae_model()
        elif self.lstm_type == 'atae':
            self.predict = self.atae_model()
        elif self.lstm_type == 'cnn':
            self.predict = self.cnn_model()
        else:
            self.predict = self.original_model()

        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.mse = tf.reduce_mean(tf.square(self.score - self.predict)) + sum(reg_loss)

        score_mean = tf.reduce_mean(self.score)
        self.r2 = 1 - tf.reduce_sum(tf.square(self.score - self.predict)) / tf.reduce_sum(
            tf.square(self.score - score_mean))

        self.train_step = tf.train.AdamOptimizer().minimize(self.mse)

    def train(self, train_data, valid_data, word_embeddings):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.word_embeddings.assign(word_embeddings))
        saver = tf.train.Saver(max_to_keep=self.n_epoch)

        context, context_len, loc_info, aspect, score = train_data
        loops = int(len(context) / self.batch_size)
        for epoch in range(self.n_epoch):
            avg_mse, avg_r2 = 0.0, 0.0
            for i in range(loops):
                feed_dict = {self.context: context[i * self.batch_size: (i + 1) * self.batch_size],
                             self.context_len: context_len[i * self.batch_size: (i + 1) * self.batch_size],
                             self.aspect: aspect[i * self.batch_size: (i + 1) * self.batch_size],
                             self.score: score[i * self.batch_size: (i + 1) * self.batch_size],
                             self.keep_prob: 0.5}
                # print(self.sess.run(self.predict, feed_dict=feed_dict))
                _, mse, r2 = self.sess.run([self.train_step, self.mse, self.r2], feed_dict=feed_dict)

                avg_mse += mse
                avg_r2 += r2

            avg_mse /= loops
            avg_r2 /= loops
            logging.debug(self.lstm_type + ' ' + str(epoch) + ' train_mse: ' +
                          str(avg_mse) + '\ttrain_r2 :' + str(avg_r2))

            if self.show:
                print(self.lstm_type, ' epoch:', epoch, 'train_mse:', avg_mse, 'train_r2:', avg_r2)

            saver.save(self.sess, os.path.join(self.save_model_path, self.save_model_name), global_step=epoch)

            valid_mse, valid_r2 = self.valid(valid_data)

            if valid_mse < self.best_mse:
                self.best_mse = valid_mse
                self.best_r2 = valid_r2
                self.best_epoch = epoch
                self.stopping_step = 0
            else:
                self.stopping_step += 1
            if self.stopping_step >= self.early_stopping_step:
                logging.debug(self.lstm_type + ' early stopping is trigger at epoch: ' + str(epoch))
                if self.show:
                    print(self.lstm_type + ' early stopping is trigger at epoch: ', epoch)
                break

        if self.show:
            print(self.lstm_type + ' best epoch:', self.best_epoch, 'best mse:', self.best_mse, 'best r2:', self.best_r2)
        logging.debug(self.lstm_type + ' best epoch: ' + str(self.best_epoch) +
                      '\tbest mse: ' + str(self.best_mse) + '\tbest r2 :' + str(self.best_r2))

        only_save_best_epoch(self.save_model_path, self.save_model_name, self.best_epoch, self.train_time)

        return self.best_mse, self.best_r2

    def valid(self, valid_data):
        context, context_len, loc_info, aspect, score = valid_data

        feed_dict = {self.context: context,
                     self.context_len: context_len,
                     self.aspect: aspect,
                     self.score: score,
                     self.keep_prob: 1.0}

        mse, r2 = self.sess.run([self.mse, self.r2], feed_dict=feed_dict)

        logging.debug(self.lstm_type + ' valid_mse: ' + str(mse) + '\tvalid_r2 :' + str(r2))

        if self.show:
            print(self.lstm_type, ' valid_mse:', mse, 'valid_r2:', r2)

        return mse, r2

    def run(self, train_data, valid_data, word_embeddings):
        mse, r2 = self.train(train_data, valid_data, word_embeddings)
        return mse, r2

    @staticmethod
    def reduce_mean(inputs, length):
        """

        :param inputs: 3D tensor->[batch_size, sequence_len, embedding_size]
        :param length: 1D tensor->[batch_size], represent every sample's len, not bigger than sequence_len
        :return: 2D tensor->[batch_size, embedding_size]
        """
        length = tf.reshape(length, [-1, 1])
        outputs = tf.reduce_sum(inputs, axis=1) / length
        return outputs

    @staticmethod
    def softmax(inputs, length, max_length):
        """

        :param inputs: 3D tensor->[batch_size, 1, n_class]
        :param length: 1D tensor->[batch_size], represent every sample's len, not bigger than sequence_len
        :param max_length:
        :return: 3D tensor->[batch_size, 1, n_class]
        """
        length = tf.reshape(length, [-1])
        inputs = tf.exp(inputs)
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True)
        return inputs / _sum
