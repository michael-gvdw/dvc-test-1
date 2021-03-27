import yaml

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

params = yaml.safe_load(open('params.yaml'))['featurize']

# load in raw dataset from local dir
df = pd.read_csv(params['source'])
column_names = list(df.columns)


# encode categorical values
label_encoder = LabelEncoder()
for column, _ in df.iteritems():
    df[column] = label_encoder.fit_transform(np.array(df[column]))
print(df.head())

# save encoded values
df.to_csv(params['source'], index=False, mode='w')


# split features and labels
y = np.array(df.pop('class/state'))
X = np.array(df)

# split into train and test sub datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=params['random_seed'])

print(f'features: {column_names[:-1]}')
print(f'label: {column_names[-1]}')

# create dataframes from numpy arrays
X_train_df = pd.DataFrame(X_train, columns=column_names[:-1])
X_test_df = pd.DataFrame(X_test, columns=column_names[:-1])

y_train_df = pd.DataFrame(y_train, columns=[column_names[-1]])
y_test_df = pd.DataFrame(y_test, columns=[column_names[-1]])


# save features and labels
X_train_df.to_csv(f'{params["to"]}X_train.csv', index=False, mode='w')
X_test_df.to_csv(f'{params["to"]}X_test.csv', index=False, mode='w')

y_train_df.to_csv(f'{params["to"]}y_train.csv', index=False, mode='w')
y_test_df.to_csv(f'{params["to"]}y_test.csv', index=False, mode='w')
