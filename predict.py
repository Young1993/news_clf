from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import utils

stopwords = utils.load_stopwords()

# text to vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, token_pattern=r"(?u)\b\w+\b",
                             ngram_range=(1, 2), stop_words=stopwords, preprocessor=utils.extract_keywords)
data_train = utils.Data()
data_train.load_data('./data/fold/train.csv')
X_train = vectorizer.fit_transform(data_train.data)
print("finished train sample transform")
be = load('./models/BernoulliNB.joblib')
mu = load('./models/MultinomialNB.joblib')
co = load('./models/ComplementNB.joblib')


def predict_label(s):
    s = utils.extract_keywords(s)
    v = vectorizer.transform([s])
    # print(be.predict_proba(v))
    res = (be.predict_proba(v) +  mu.predict_proba(v) + co.predict_proba(v)) / 3
    return res[0]

print(predict_label('“阅读让城市更有温度！”第二十一届深圳读书月盐田区活动启动'))