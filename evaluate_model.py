import json
import pickle

import numpy as np
import pandas as pd

from sklearn import metrics

clf = pickle.load(open('./assets/models/model.pickle', mode='rb'))

X_test = np.array(pd.read_csv('./assets/features/X_test.csv'))
y_test = np.array(pd.read_csv('./assets/features/y_test.csv'))

y_pred = clf.predict(X_test)

accuracy_score = metrics.accuracy_score(y_test, y_pred)
print(accuracy_score)

precission_score = metrics.precision_score(y_test, y_pred, average=None)
print(precission_score)

recall_score = metrics.recall_score(y_test, y_pred, average=None)
print(recall_score)

f1_score = metrics.f1_score(y_test, y_pred, average=None)
print(f1_score)


with open('./assets/metrics/metrics.json', mode='w') as output_file:
    json.dump(dict(accuracy_score=accuracy_score, 
                    precision_score=list(precission_score), 
                    recall_score=list(recall_score), 
                    f1_score=list(f1_score)), 
                output_file
            ) 