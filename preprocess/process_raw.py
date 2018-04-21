# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: process_raw.py

@time: 2018/4/20 21:36

@desc: pre-processing raw data, including:
       1. change stock text into lower cases
       2. extract aspect level1 & aspect level2
       3. replace stock name in stock text with token 'T'

"""

import os
import codecs
import json


def parse_json(json_files, save_file):
    writer = codecs.open(save_file, 'w', encoding='utf8')

    for json_file in json_files:
        with codecs.open(json_file, 'r', encoding='utf8')as reader:
            data_json = json.load(reader)
        for sample in data_json.values():
            stock_text = sample['sentence'].lower()
            for target in sample['info']:
                stock = target['target'].lower()
                aspects = target['aspects'].split(',')[0]
                aspects = aspects.replace('[', '').replace(']', '').replace('\'', '').replace('\"', '').split('/')
                aspect_l1 = '_'.join(aspects[0].split(' '))
                aspect_l2 = '_'.join(aspects[1].split(' '))
                score = target['sentiment_score']

                if stock not in stock_text:
                    sentence = ' T ' + stock_text
                else:
                    start_pos = stock_text.index(stock)
                    end_pos = start_pos + len(stock)
                    sentence = stock_text[:start_pos] + ' T ' + stock_text[end_pos:]    # replace stock with 'T'
                writer.write(sentence + '\t' + aspect_l1 + '\t' + aspect_l2 + '\t' + score + '\n')
    writer.close()


if __name__ == '__main__':
    save_dir = '../data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    parse_json(json_files=['../raw_data/task1_headline_ABSA_train.json', '../raw_data/task1_post_ABSA_train.json'],
               save_file=os.path.join(save_dir, 'train.tsv'))

