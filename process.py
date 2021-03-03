import pandas as pd
import os
import re
import codecs
import jieba
import json
# import tqdm


def filter_tags(htmlstr):
    # 先过滤CDATA
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)  # 匹配CDATA
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # Script
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # style
    re_br = re.compile('<br\s*?/?>')  # 处理换行
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    re_comment = re.compile('<!--[^>]*-->')  # HTML注释
    s = re_cdata.sub('', htmlstr)  # 去掉CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)  # 去掉style
    s = re_br.sub('', s)  # 将br转换为 空字符串
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_comment.sub('', s)  # 去掉HTML注释
    # 去掉多余的空行
    blank_line = re.compile('\n+')
    s = blank_line.sub('\n', s)
    return s


def process_news():
    df_label = pd.read_csv('./dict.csv', usecols=['label'])
    label_list = df_label.label.tolist()
    news = []
    with codecs.open('./article_for_cat', 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp = json.loads(line)
            tmp['category'] = tmp['category'].split('-')[0]
            tmp['category'] = label_list.index(tmp['category'])
            tmp['content'] = filter_tags(tmp['content'])
            if len(tmp['content']) > 20:
                print(tmp['info_id'])
                news.append(tmp)
    df = pd.DataFrame(news)
    df.to_csv('./data/news.csv', index=False)


# 教育 41936
# 财经 37098
# 时政 63086
# 科技 162929
# 社会
# 健康 5k

def handle_wrong_data():
    df = pd.read_csv('./data/news/health.csv')
    c = 0
    for i in range(len(df)):
        # if re.search('例确诊病例|输入性病例|阳性！安徽籍！|确诊轨迹公布|全球疫情动态|境外输入|国内新冠肺炎疫情|无症状感染者|多名被感染者被|新增确诊|新*最新情况|本土确诊，新增', df.title[i]):
        # if re.search('培训会|.*义诊活动|知识讲座|世卫：|爱心公示：|疾控重要提醒|贵州：|新冠肺炎疫情|新冠肺炎确诊|新增.*例', df.title[i]):
        if re.search('Science子刊|Hepatol Int|一致性评价|显出顶级疗效|修美乐|赛诺菲|阿斯利康|新药|老药', df.title[i]):
            print(df.title[i])
            c += 1
            df.label[i] = '其他'
    print(c)
    df.to_csv('./data/news/health.csv', index=False)


def handle_news():
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


# 合并 原始数据
def merge_data():
    # merge data
    path = './data/news/'
    files = os.listdir(path)
    arr = []
    for block in files:
        if re.search('.csv', block):
            df = pd.read_csv(path + block)
            if len(df) < 10000:
                arr.append(df)
            else:
                arr.append(df.sample(n=10000, random_state=1423, axis=0))

    df1 = pd.concat(arr, ignore_index=True, sort=False)
    df1.to_csv('./data/raw.csv', index=False)


# 分割 data/ 数据
def split_data(file='./data/raw.csv', data_dir='./data/train'):
    df = pd.read_csv(file)
    print(df.info())
    df = df.sample(frac=1)
    split_1 = int(0.82 * len(df))
    split_2 = int(0.91 * len(df))
    train_data = df[:split_1]
    val_data = df[split_1:split_2]
    test_data = df[split_2:]
    train_data.to_csv(data_dir + '/train.csv', index=False)
    val_data.to_csv(data_dir + './val.csv', index=False)
    test_data.to_csv(data_dir + './test.csv', index=False)


'''
 data 中数据 导出到 fold，生成正式可训练数据
 利用结巴 Jie ba 截取 512 个词
'''


def export():
    class_name = ['教育', '财经', '时政', '科技', '社会', '健康', '其他']
    for block in ['train', 'val', 'test']:
        df = pd.read_csv('./data/train/' + block + '.csv')
        arr = []
        for i in range(len(df)):
            s = handle_str(df['title'][i]) + handle_str(df['content'][i])
            tmp = jieba.lcut(s)
            tmp = tmp[:512]
            tmp = ''.join(tmp)
            arr.append({
                'text': tmp,
                'label': class_name.index(df['label'][i])
            })
        df1 = pd.DataFrame(arr)
        df1.to_csv('./fold/' + block + '.csv', index=False)


def labeled_data():
    d = []
    re_comment = re.compile('<!--[^>]*-->')
    re_html = re.compile('<[^>]*>')
    with codecs.open('./data/tmp/评测数据', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            line['content'] = line['content'].replace('> 下需要用 p 标签分段，不能直接就放文字或图片标签 -->', '')
            line['content'] = re_comment.sub('', line['content'])
            line['content'] = re_html.sub('', line['content'])
            d.append({
                'title': line['title'],
                'content': line['content'].strip()
            })
    df = pd.DataFrame(d)
    df.to_csv('./data/sample.csv', index=False)


def statistics(filename='./fold/train.csv'):
    df = pd.read_csv(filename)
    # print(df.info())
    # print(df.groupby('label'))
    for g in df.groupby('label'):
        print(g[0], len(g[1]))


# handle string
def handle_str(s):
    try:
        if type(s) == str:
            return s.strip().replace(' ', '').replace('\u3000', '').replace('\n', '')
        else:
            return ''
    except:
        print(s)
        return ''


def handle_predict():
    df1 = pd.read_csv('./tmp/sample_predict.csv')
    df2 = pd.read_csv('./tmp/predict_modify.csv')
    info = df2.info_id.tolist()
    c = 0

    for o in range(len(df1)):
        if df1['info_id'][o] in info:
            c += 1
            index = info.index(df1['info_id'][o])
            print(df1.title[o], df1['predict'][o], df2.predict[index])
            df1['predict'][o] = df2.predict[index]

    df1 = df1.drop(labels=['info_id', "category", "tag", "poi"], axis=1)
    df1 = df1.rename(columns={'predict': 'label'})
    df1.to_csv('./data/news/labeled.csv', index=False)


def handle_label():
    df = pd.read_csv('./data/raw.csv')

    for i in range(len(df)):
        if df.label[i] in ['其他', '广告', '职场']:
            df.label[i] = '其他'
        elif df.label[i] == '要闻':
            df.label[i] = '时政'

    df.to_csv('./data/raw.csv', index=False)


if __name__ == '__main__':
    # handle the whole classes of news
    # process_news()
    split_data(file='./data/news.csv', data_dir='./data')
# 处理错误标签的标记数据
# handle_wrong_data()
# 处理预测的数据
# handle_predict()
# 合并数据
# merge_data()
# 统计分析数据
# statistics('./data/raw.csv')
# 处理标签
# handle_label()
# 分割出数据成 train、val、test
# split_data()
# 处理导出到 fold
# export()
# 处理网易新闻给过来的数据
# labeled_data()
