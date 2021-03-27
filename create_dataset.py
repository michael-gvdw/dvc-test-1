import numpy as np
import pandas as pd


column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class/state']
classes = ['unacc', 'acc', 'good', 'vgood']

# read file from uci repository
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', 
                header=None, 
                names=column_names)

# show info about dataset
print(df.shape)
print(df.head())

# save raw dataset
df.to_csv('./assets/original_dataset/car_eval_raw.csv', index=False, mode='w')

