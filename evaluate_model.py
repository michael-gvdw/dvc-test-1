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
    

metrics_list = []
accuracy_score = metrics.accuracy_score(y_test, y_pred)
print(accuracy_score)

precission_score = list(metrics.precision_score(y_test, y_pred, average=None))
metrics_list.append(precission_score)
print(precission_score)

recall_score = list(metrics.recall_score(y_test, y_pred, average=None))
metrics_list.append(recall_score)
print(recall_score)

f1_score = list(metrics.f1_score(y_test, y_pred, average=None))
metrics_list.append(f1_score)
print(f1_score)

classes= ['unacc', 'acc', 'good', 'vgood']

temp = []
for metrics in metrics_list:
    temp.append(dict(zip(classes, metrics)))

print(temp[0])
precission_score_dict = temp[0]
recall_score_dict = temp[1]
f1_score_dict = temp[2]

with open('./assets/pipeline/metrics/metrics.json', mode='w') as output_file:
    json.dump(dict(accuracy_score=accuracy_score, 
                    precision_score=precission_score_dict, 
                    recall_score=recall_score_dict, 
                    f1_score=f1_score_dict
                ), 
                    output_file
                ) 