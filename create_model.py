import os
import yaml
import pickle

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

# create directories if not existing
os.makedirs('./assets/pipeline/models/', exist_ok=True)

params = yaml.safe_load(open('params.yaml'))['train']

X_train = np.array(pd.read_csv(f'{params["source"]}X_train.csv'))
y_train = np.array(pd.read_csv(f'{params["source"]}y_train.csv'))

X_test = np.array(pd.read_csv(f'{params["source"]}X_test.csv'))
y_test = np.array(pd.read_csv(f'{params["source"]}y_test.csv'))


clf = DecisionTreeClassifier(criterion=params['tree']['criterion'], 
                            splitter=params['tree']['splitter'], 
                            max_depth=params['tree']['max_depth'], 
                            max_features=params['tree']['max_features'])
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

pickle.dump(clf, open(f'{params["to"]}{params["tree"]["file_name"]}', mode='wb'))



