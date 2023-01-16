import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('/Users/yinghongliu/Downloads/loans50k.csv')
num = pd.DataFrame()# store numerical features
df = df.dropna()
for e in list(df.columns):
    if df[e].dtype == 'float64' : # drop the load ID feature not including int64
        num[e] = df[e]
df['status'] = df['status'].replace(['Fully Paid', 'In Grace Period', 'Late (16-30 days)','Late (31-120 days)','Current'], '0')
df['status'] = df['status'].replace(['Charged Off', 'Default' ], '1')
df['status'] = df['status'].astype('int64')
X = df['amount'] # will add all features after completion
y = df['status']  # 1: default, 0: not default

model = LogisticRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
