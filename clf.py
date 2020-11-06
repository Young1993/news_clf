# -!- coding: utf-8 -!-
import json
import codecs
import pandas as pd
import re
import jieba
import jieba.analyse
import gensim
from collections import Counter


def read_json():
    f = open('./data/样本-健康')
    content = f.read()
    o = json.loads(content)
    # print(o)
    return o


def statistics():
    df = pd.read_csv('./data/fold/data.csv')
    # print(df.info())
    # print(len(df))
    # raise

    # add stop words
    stopwords = []
    with codecs.open('./stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            # print(line.strip())
            stopwords.append(line.strip())

    # analyse
    tags = jieba.analyse.extract_tags(df['content'][0], topK=20, withWeight=True, allowPOS=())
    print(tags)

    # count the words
    # c = Counter()
    # for i in range(len(df)):
    #     words = jieba.lcut(df['content'][i].strip()) # 分词工具后面需要比较
    #     for w in words:
    #         if w not in stopwords:
    #             c[w] += 1
    #             # print(w)
    # print(c)


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
    statistics()
    # main()
