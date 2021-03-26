import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class/state']
classes = ['unacc', 'acc', 'good', 'vgood']


# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', 
#                 header=None, 
#                 names=column_names)
# df.to_csv('./assets/original_dataset/car_eval_raw.csv', index=False, mode='w')


# load in the original dataset
df = pd.read_csv('./assets/original_dataset/car_eval_raw.csv')
print(df.head())

print()

# encode categorical values
label_encoder = LabelEncoder()
for column, _ in df.iteritems():
    df[column] = label_encoder.fit_transform(np.array(df[column]))
print(df.head())

df.to_csv('./assets/data/car_eval_encoded.csv', index=False, mode='w')
