import yaml

import numpy as np
import pandas as pd


params = yaml.safe_load(open('params.yaml'))['load']


# read file from uci repository
df = pd.read_csv(params['source'], 
                header=params['header'], 
                names=params['column_names'])

# show info about dataset
print(df.shape)
print(df.head())

# save raw dataset
df.to_csv(params['to'], index=False, mode='w')

