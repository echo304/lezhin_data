import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 500)

df = pd.read_table("./lezhin_dataset_v2_training.tsv", header=None)
df_test = pd.read_table("./lezhin_dataset_v2_test_without_label.tsv", header=None)

def preprocess_dataset(dataset):
  # convert dtype of the values
  dataset = dataset.astype(np.float64)

  # fill missing values with mean
  dataset = dataset.fillna(dataset.mean())

  # Numpy values
  return dataset.values

# Drop hashed value
# 해쉬값을 임의의 값으로 변환하는 방식을 고려했으나 새로운 데이터에는 새로운 해쉬값 
# 올 수도 있으므로 삭제하기로 결정 
df = df.drop([6, 7, 9], axis=1)
df_test = df_test.drop([5, 6, 8], axis=1)

# extract label
y = df.pop(0).values

# preprocess
X = preprocess_dataset(df)
X_test = preprocess_dataset(df_test)

# split training set and test set by 9:1
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

# Standardization with X_train dataset
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train) 
X_valid = scaler.transform(X_valid) 

# print(pd.DataFrame(X_train))
# no. of input feature
input = X.shape[1]

# no. of hidden layer feature
hidden = input

# no. of output
output = 2

# selected alpha value with cross validation test
alpha = 0.0005

# selected learning rate value
rate = 0.001

# Prediction model

clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(hidden, 2), 
                  learning_rate_init=rate, max_iter=2000, verbose=True)
clf.fit(X_train, y_train)
test_score = clf.score(X_valid, y_valid)
print('Test score : ', test_score)

# Result with Test dataset
result = clf.predict(X_test)
