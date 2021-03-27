import os
import json
import yaml
import pickle

import numpy as np
import pandas as pd

from sklearn import metrics

# create directories if not existing
os.makedirs('./assets/pipeline/metrics/', exist_ok=True)

params = yaml.safe_load(open('params.yaml'))['evaluate']
 
clf = pickle.load(open('./assets/pipeline/models/model.pickle', mode='rb'))

X_test = np.array(pd.read_csv('./assets/pipeline/features/X_test.csv'))
y_test = np.array(pd.read_csv('./assets/pipeline/features/y_test.csv'))

y_pred = clf.predict(X_test)
    
accuracy_score = metrics.accuracy_score(y_test, y_pred)
print(accuracy_score)

precission_score = metrics.precision_score(y_test, y_pred, average=None)
print(precission_score)

recall_score = metrics.recall_score(y_test, y_pred, average=None)
print(recall_score)

f1_score = metrics.f1_score(y_test, y_pred, average=None)
print(f1_score)


with open('./assets/pipeline/metrics/metrics.json', mode='w') as output_file:
    json.dump(dict(accuracy_score=accuracy_score, 
                    precision_score=list(precission_score), 
                    recall_score=list(recall_score), 
                    f1_score=list(f1_score)), 
                output_file
            ) 