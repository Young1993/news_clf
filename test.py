# import jieba.analyse
import pandas as pd
import numpy as np
import codecs
import torch.nn.functional as F

a = np.array([0.55389804, -6.314381, 2.2801042, -0.46613902, -6.0917926])
print(F.softmax(a, dim=-1))


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
