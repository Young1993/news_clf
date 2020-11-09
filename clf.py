# -!- coding: utf-8 -!-
import json
import codecs
import pandas as pd
import re
import jieba
import logging
import numpy as np
import utils
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from joblib import dump, load
from sklearn import metrics
from sklearn.utils.extmath import density


def read_json():
    f = open('./data/样本-健康')
    content = f.read()
    o = json.loads(content)
    # print(o)
    return o


def splice(s):
    return s[:2]

# extract sample
def select_sample():
    df = pd.read_csv('./data/fold/sample.csv')
    # df['category'] = df['category'].apply(splice)
    # df.to_csv('./data/fold/sample.csv', index=False)
    df = df.sample(frac=1)
    active_sample = df[15:]
    train_sample = df[:15]
    train_sample.to_csv('./data/fold/train.csv', index=False)
    active_sample.to_csv('./data/fold/active.csv', index=False)


def extract_keywords(s):
    tags = jieba.analyse.extract_tags(s, topK=30, withWeight=True, allowPOS=())
    res = ''
    for item in tags:
        res += ' ' + item[0]
    return res


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 100 else s[:97] + "..."


def benchmark(clf, name, X_train, y_train, X_active, y_test, feature_names, class_name):
    logging.info('_' * 80)
    logging.info("Training: ")
    logging.info(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    logging.info("train time: %0.3fs" % train_time)

    dump(clf, './models/' + name + '.joblib')

    t0 = time()
    pred = clf.predict(X_active)
    test_time = time() - t0
    logging.info("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred, average='macro')
    precision = metrics.precision_score(y_test, pred, average='macro')
    recall = metrics.recall_score(y_test, pred, average='macro')
    logging.info("accuracy:   %0.3f" % score)
    logging.info("f1:   %0.3f" % f1)
    logging.info("precision:   %0.3f" % precision)
    logging.info("recall:   %0.3f" % recall)

    if hasattr(clf, 'coef_'):
        logging.info("dimensionality: %d" % clf.coef_.shape[1])
        logging.info("density: %f" % density(clf.coef_))

        if feature_names is not None:
            logging.info("top 10 keywords per class:")
            for i, label in enumerate(class_name):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))

    logging.info("classification report:")
    logging.info(metrics.classification_report(y_test, pred, target_names=class_name))

    logging.info("confusion matrix:")
    logging.info(metrics.confusion_matrix(y_test, pred))

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


# train classifier
def train():
    # load data
    data_train = utils.Data()
    data_train.load_data('./data/fold/train.csv')
    data_active = utils.Data()
    data_active.load_data('./data/fold/active.csv')
    logging.info('data loaded')
    print("%d categories" % len(data_train.target_names))

    stopwords = load_stopwords()

    # text to vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, token_pattern=r"(?u)\b\w+\b",
                                 ngram_range=(1,2), stop_words=stopwords, preprocessor=extract_keywords)
    # analyzer='char_wb',
    X_train = vectorizer.fit_transform(data_train.data)
    logging.info("finished train sample transform")

    X_active = vectorizer.transform(data_active.data)
    logging.info("finished active sample transform")

    y_train, y_test = data_train.target, data_active.target

    logging.info("train n_samples: %d, n_features: %d" % X_train.shape)
    logging.info("active n_samples: %d, n_features: %d" % X_active.shape)
    feature_names = vectorizer.get_feature_names()
    feature_names = np.asarray(feature_names)

    # Train sparse Naive Bayes classifiers
    logging.info('=' * 80)
    logging.info("Naive Bayes")

    benchmark(MultinomialNB(alpha=.01), 'MultinomialNB', X_train, y_train, X_active, y_test, feature_names,
              data_train.class_name)
    benchmark(BernoulliNB(alpha=.01), 'BernoulliNB', X_train, y_train, X_active, y_test, feature_names,
              data_train.class_name)
    benchmark(ComplementNB(alpha=.1), 'ComplementNB', X_train, y_train, X_active, y_test, feature_names,
              data_train.class_name)


# load stop words
def load_stopwords():
    stopwords = []
    with codecs.open('./stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # print(line.strip())
            stopwords.append(line.strip())
    return stopwords


# classfier for news
def main():
    news_list = []
    re_comment = re.compile('<!--[^>]*-->')
    re_html = re.compile('<[^>]*>')
    for o in ['样本-健康', '样本-教育', '样本-社会', '样本-科技', '样本-要闻']:
        print(o)
        with codecs.open('./data/' + o, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                try:
                    line = json.loads(line)
                    line['content'] = line['content'].replace('> 下需要用 p 标签分段，不能直接就放文字或图片标签 -->', '')
                    line['content'] = re_comment.sub('', line['content'])
                    line['content'] = re_html.sub('', line['content'])
                    print(i, line['info_id'])
                    news_list.append(line)
                except:
                    continue
    df = pd.DataFrame(news_list)
    df.to_csv('./data/fold/data.csv', index=False)


if __name__ == '__main__':
    logging.basicConfig(filemode='w', filename="./logs/log.txt", level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    # select_sample()
    train()
    # process_data()
    # main()
