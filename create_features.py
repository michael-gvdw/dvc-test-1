import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('./assets/data/car_eval_encoded.csv')
column_names = list(df.columns)

print(df.head())

y = np.array(df.pop('class/state'))
X = np.array(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(column_names[:-1])
print(column_names[-1])

X_train_df = pd.DataFrame(X_train, columns=column_names[:-1])
X_test_df = pd.DataFrame(X_test, columns=column_names[:-1])

y_train_df = pd.DataFrame(y_train, columns=[column_names[-1]])
y_test_df = pd.DataFrame(y_test, columns=[column_names[-1]])


X_train_df.to_csv('./assets/features/X_train.csv', index=False, mode='w')
X_test_df.to_csv('./assets/features/X_test.csv', index=False, mode='w')

y_train_df.to_csv('./assets/features/y_train.csv', index=False, mode='w')
y_test_df.to_csv('./assets/features/y_test.csv', index=False, mode='w')
