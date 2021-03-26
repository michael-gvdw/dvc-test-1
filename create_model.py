import pickle

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier


X_train = np.array(pd.read_csv('./assets/features/X_train.csv'))
y_train = np.array(pd.read_csv('./assets/features/y_train.csv'))

X_test = np.array(pd.read_csv('./assets/features/X_test.csv'))
y_test = np.array(pd.read_csv('./assets/features/y_test.csv'))


clf = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=10, max_features='sqrt')
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

pickle.dump(clf, open('./assets/models/model.pickle', mode='wb'))



