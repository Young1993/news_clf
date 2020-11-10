import jieba.analyse
import pandas as pd
import numpy as np
import codecs

# load stop words
def load_stopwords():
    stopwords = []
    with codecs.open('./stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # print(line.strip())
            stopwords.append(line.strip())
    return stopwords

def extract_keywords(s):
    tags = jieba.analyse.extract_tags(s, topK=30, withWeight=True, allowPOS=())
    res = ''
    for item in tags:
        res += ' ' + item[0]
    return res

class Data():
    def __init__(self):
        self.data = []
        self.target = []
        self.target_names = []
        self.class_name = ['社会', '要闻', '健康', '科技', '教育']

    def load_data(self, path):
        df = pd.read_csv(path) #, nrows=30
        for i in range(len(df)):
            self.data.append(df['title'][i] + '  ' + df['content'][i])
            self.target.append(self.class_name.index(df['category'][i]))
            self.target_names.append(df['category'][i])
        self.target = np.asarray(self.target)