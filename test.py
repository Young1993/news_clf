# import jieba.analyse

import jieba
from torchtext import data
from torchtext import vocab

import pandas as pd
import numpy as np
import torch
import codecs
import torch.nn.functional as F

# load stop words
def load_stopwords():
    stopwords = []
    with codecs.open('tmp/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # print(line.strip())
            stopwords.append(line.strip())
    return stopwords


def extract_keywords(s):
    # jieba.analyse.textrank
    tags = jieba.analyse.extract_tags(s, topK=35, withWeight=True, allowPOS=(['n', 'v', 'nt', 'vn']))
    res = ''
    for item in tags:
        res += ' ' + item[0]
    print(res)
    return res


class Data():
    def __init__(self):
        self.data = []
        self.target = []
        self.target_names = []
        self.class_name = ['社会', '时政', '健康', '科技', '教育']

    def load_data(self, path):
        df = pd.read_csv(path)  # , nrows=30
        for i in range(len(df)):
            tmp = extract_keywords(df['title'][i] + '  ' + df['content'][i])
            self.data.append(tmp)
            self.target.append(self.class_name.index(df['category'][i]))
            self.target_names.append(df['category'][i])
        self.target = np.asarray(self.target)

import dill

embedding = './data/sgns.sogou.word'

def tokenizer(s):
    return jieba.lcut(s)

# load data
def load_news(config, text_field, band_field):
    fields = {
        'text': ('text', text_field),
        'label': ('label', band_field)
    }

    word_vectors = vocab.Vectors(config.embedding_file)

    train, val, test = data.TabularDataset.splits(
        path=config.data_path, train='train.csv', validation='val.csv',
        test='test.csv', format='csv', fields=fields)

    print("the size of train: {}, dev:{}, test:{}".format(
        len(train.examples), len(val.examples), len(test.examples)))

    text_field.build_vocab(train, val, test, max_size=config.n_vocab, vectors=word_vectors,
                           unk_init=torch.Tensor.normal_)
    band_field.build_vocab(train, val, test)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(config.batch_size, config.batch_size, config.batch_size), sort=False,
        device=config.device, sort_within_batch=False, shuffle=False)



# data loader and split Chinese
text_field = data.Field(tokenize=tokenizer, include_lengths=True, fix_length=512)

band_field = data.Field(sequential=False, use_vocab=False, batch_first=True,
                        dtype=torch.int64, preprocessing=data.Pipeline(lambda x: int(x)))

train_iterator, val_iterator, test_iterator = load_news(config, text_field, band_field)