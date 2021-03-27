import os
import yaml

import numpy as np
import pandas as pd

# create directories if not existing
os.makedirs('./assets/pipeline/original_dataset/', exist_ok=True)
os.makedirs('./assets/pipeline/data/', exist_ok=True)

params = yaml.safe_load(open('params.yaml'))['load']


# read file from uci repository
df = pd.read_csv(params['source'], header=params['header'])

# save raw dataset
df.to_csv(params['to'], index=False, mode='w')

# show info about dataset
print(df.shape)
print(df.head())

print(df.columns)
# add column names
df.columns = params['column_names']
print(df.columns)

# show the columns
print(df.head())

df.to_csv('./assets/pipeline/data/car_eval.csv', index=False, mode='w')


