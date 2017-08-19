import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def replace_hash_with_int(hash, hash_list):
  if hash == 0:
    hash = '0'
  index = np.where(hash_list == hash)[0][0]
  return index

df = pd.read_table("./lezhin_public_dataset_training.tsv", header=None)

# # randomly shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Process hashed value 
unique_value_7 = df[7].unique()
unique_value_8 = df[8].unique()
unique_value_16 = df[16].unique()
unique_value_18 = df[18].unique()
df[7] = df[7].map(lambda x: replace_hash_with_int(x, unique_value_7))
df[8] = df[8].map(lambda x: replace_hash_with_int(x, unique_value_8))
df[16] = df[16].map(lambda x: replace_hash_with_int(x, unique_value_16))
df[18] = df[18].map(lambda x: replace_hash_with_int(x, unique_value_18))

# convert dtype of the values
df = df.astype(np.float64)

# extract label
y = df.pop(0).values

# example
X = df.values

scaler = preprocessing.StandardScaler().fit(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


X_train, y_train = scaler.transform(X_train), y_train
X_test, y_test = scaler.transform(X_test), y_test

print(X_test)
print(X_train.shape)
print(X_test.shape)

# no. of input feature
input = X.shape[1]

# no. of hidden layer feature
hidden = input

# no. of output
output = 2

# selected alpha value with cross validation test
alpha = 0.1

# selected learning rate value
rate = 0.005

# Prediction model
clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=(hidden,), 
                   learning_rate_init=rate, max_iter=2000, verbose=True)

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
