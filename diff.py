import pandas as pd
import jieba


# df = pd.read_csv('./data/tmp/diff.csv')
# df =df.sample(frac=0.1)
# df.to_csv('./data/tmp/diff_sm.csv', index=False)

# df = pd.read_csv('predict.csv', usecols=['title', 'category', 'predict', 'content'])
# total = len(df)
# d = 0
# c = 0
# index = []
# for i in range(total):
#     if df['category'][i] == '健康':
#         d += 1
#         continue
#     elif df['category'][i] == df['predict'][i]:
#         c += 1
#     else:
#         index.append(i)
# df2 = df.iloc[index]
# df2.to_csv('./data/tmp/diff.csv', index=False)
# print(c, d, len(df2))
# df_active = pd.read_csv('./data/fold/health_confirm.csv', usecols=['title', 'content', 'category'])
# df_train = pd.read_csv('./data/fold/health.csv', usecols=['title', 'content', 'category'])
#

# df = pd.concat([df_active, df_train])
# df = df.rename(columns={'category':'label'})
# df = df[['title','content','label']]
# df.to_csv('./data/news/health.csv', index=False)

# print(df_train[df_train['category'] == '健康'])
# df_health = df_train[df_train['category'] == '健康']
# df_health.to_csv('./data/fold/health.csv', index=False)

# df_health = df_active[df_active['category'] == '健康']
# df_health.to_csv('./data/fold/health_to_confirm.csv', index=False)

# df_active = pd.read_csv('./data/fold/health_confirm.csv', usecols=['title', 'content', 'category'])
# df_train = pd.read_csv('./data/fold/train.csv', usecols=['title', 'content', 'category'])
# df = pd.concat([df_active, df_train])
# df = df.rename(columns={'category':'label'})
# df = df[['title','content','label']]
# df.to_csv('./data/news/health.csv', index=False)

def tokenizer(s):
    return jieba.lcut(s)


# df = pd.read_csv('./data/sample.csv')
# for i in range(len(df)):
#     try:
#         t = df['title'][i] if type(df['title'][i]) == str else ''
#         c = df['content'][i] if type(df['content'][i]) == str else ''
#         input_doc = tokenizer(t + c)
#     except:
#         print(i, df['title'][i], df['content'][i])
#
# for i in range(len(df)):
#     t = df['title'][i] if type(df['title'][i]) == str else ''
#     c = df['content'][i] if type(df['content'][i]) == str else ''
#     print(i)

df = pd.read_csv('./predict.csv')
df1 = df[:200]
df2 = df[1000: 1200]
df3 = df[2000: 2200]
df4 = df[3000: 3200]
df5 = df[4000: 4200]
df6 = pd.concat([df1,df2,df3,df4,df5], ignore_index=True, sort=False)
df6.to_csv('./sample_predict.csv', index=False)