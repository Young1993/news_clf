import pandas as pd

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

df_active = pd.read_csv('./data/fold/health_confirm.csv', usecols=['title', 'content', 'category'])
df_train = pd.read_csv('./data/fold/train.csv', usecols=['title', 'content', 'category'])
df = pd.concat([df_active, df_train])
df = df.rename(columns={'category':'label'})
df = df[['title','content','label']]
df.to_csv('./data/news/health.csv', index=False)