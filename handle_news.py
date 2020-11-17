import pandas as pd
import os
import re
import codecs
import jieba


# 教育 41936
# 财经 37098
# 时政 63086
# 科技 162929
# 社会
# 健康 5k

def handle_text():
    path = './data/THUCNews/科技'
    files = os.listdir(path)
    print(len(files))

    news_list = []

    for block in files:
        if re.search('.txt', block):
            with codecs.open(path + '/' + block, 'r', 'utf-8') as f:
                # title, content = f.readlines()[0], " ".join(f.readlines()[1:])

                lines = f.readlines()
                title = lines[0]
                content = ''.join(lines[1:])
                # print(title, content)
                news_list.append({
                    'title': title,
                    'content': content,
                    'label': '科技'
                })
    df = pd.DataFrame(news_list)
    df.to_csv('./data/news/tech.csv', index=False)


def merge_data():
    # merge data
    path = './data/news/'
    files = os.listdir(path)
    arr = []
    for block in files:
        # print(block)
        df = pd.read_csv(path + block)
        if len(df) < 10000:
            arr.append(df)
        else:
            arr.append(df.sample(n=10000, random_state=1423, axis=0))

    df1 = pd.concat(arr, ignore_index=True, sort=False)
    df1.to_csv('./data/news/raw.csv', index=False)


def split_data():
    df = pd.read_csv('./data/news/data.csv')
    # print(df.head())
    # print(len(df))
    df = df.sample(frac=1)
    split_1 = int(0.8 * len(df))
    split_2 = int(0.9 * len(df))
    train_data = df[:split_1]
    val_data = df[split_1:split_2]
    test_data = df[split_2:]
    train_data.to_csv('./data/train/train.csv', index=False)
    val_data.to_csv('./data/train/val.csv', index=False)
    test_data.to_csv('./data/train/test.csv', index=False)


def replace_name(name):
    return name if name != '时政' else '要闻'


def replace_label():
    df = pd.read_csv('./data/news/raw.csv')

    df['label'] = df['label'].apply(replace_name)
    df.to_csv('./data/news/data.csv', index=False)


def process():
    class_name = ['教育', '财经', '要闻', '科技', '社会', '健康']
    df = pd.read_csv('./data/train/train.csv')
    arr = []
    for i in range(len(df)):
        s = handle(df['title'][i]) + '' + handle(df['content'][i])
        tmp = jieba.lcut(s)
        tmp = tmp[:512]
        tmp = ''.join(tmp)
        arr.append({
            'text': tmp,
            'label': class_name.index(df['label'][i])
        })
    df1 = pd.DataFrame(arr)
    df1.to_csv('./fold/train.csv', index=False)


def statistics():
    df = pd.read_csv('./data/news/data.csv')
    print(df.groupby('label'))
    for g in df.groupby('label'):
        print(g)


def handle(s):
    try:
        if type(s) == str:
            return s.strip().replace('\u3000', '').replace('\n', '')
        else:
            return ''
    except:
        print(s)
        return ''


if __name__ == '__main__':
    # merge_data()
    # replace_label()
    # statistics()
    # split_data()
    process()
    # df = pd.read_csv('./data/train/train.csv')
    # # print(df.iloc[37980])
    # for i in range(len(df)):
    #     s = handle(df['title'][i]) + '' + handle(df['content'][i])

    # df['title'] = df['title'].apply(handle)
    # df['content'] = df['content'].apply(handle)
    #
    # df.to_csv('./fold/test.csv', index=False)
    # val_data.to_csv('./data/train/val.csv', index=False)
    # test_data.to_csv('./data/train/test.csv', index=False)
    # print(df.groupby('label'))
    # print(len(df))
