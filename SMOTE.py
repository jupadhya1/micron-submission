#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

"""
Synthetic Minority Oversampling Technique
Imbalanced-Learn Library
SMOTE for Balancing Data
SMOTE for Classification
SMOTE With Selective Synthetic Sample Generation
Borderline-SMOTE
Borderline-SMOTE SVM
Adaptive Synthetic Sampling (ADASYN)
"""

"""
 problem with imbalanced classification is that there are too few examples of the minority class for a model to effectively learn the decision boundary.

One way to solve this problem is to oversample the examples in the minority class. This can be achieved by simply duplicating examples from the minority class in the training dataset prior to fitting a model. This can balance the class distribution but does not provide any additional information to the model.

An improvement on duplicating examples from the minority class is to synthesize new examples from the minority class. This is a type of data augmentation for tabular data and can be very effective.


SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.

Specifically, a random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5). A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two examples in feature space.
"""
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# summarize class distribution
counter = Counter(y)
print(counter)
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

# In[2]:


X = pd.read_csv("../assets/data/data_new.csv", header=None, index_col=None)
X.head(5)


# In[4]:


if X.columns.to_list().count('target') > 0:
    X = X.drop('target', axis=1)


# In[5]:


Y = pd.read_csv("../assets/data/targets.csv", index_col=0)
Y.head(5)


# In[6]:


XY = pd.concat([X,Y], axis=1)


# In[7]:


XY


# In[8]:


XY = XY.loc[(XY.target > 0.0)]


# Lets check their frequencies

# In[9]:


Y = XY.target.astype('int').astype('category').reset_index().drop('index', axis=1)
X = XY.iloc[:,:-1].reset_index().drop('index', axis=1)


# In[10]:


Y.shape, X.shape


# In[11]:


Y.target.value_counts()



from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[13]:


X_train_, X_test, y_train_, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# In[14]:


y_test.target.value_counts()


# In[15]:


y_train_.target.value_counts()


# In[74]:


svc = LinearSVC(random_state=42, C=10).fit(X_train_, y_train_)
confusion_matrix(y_test, svc.predict(X_test))


# In[75]:


dtr = DecisionTreeClassifier(random_state=42).fit(X_train_, y_train_)
confusion_matrix(y_test, dtr.predict(X_test))


# In[78]:


lgr = LogisticRegression(random_state=42).fit(X_train_, y_train_)
confusion_matrix(y_test, lgr.predict(X_test))




y_train_.target.value_counts()



from imblearn.over_sampling import SVMSMOTE, BorderlineSMOTE, KMeansSMOTE, ADASYN, SMOTE


# In[21]:


for smote in [SVMSMOTE(random_state=42), BorderlineSMOTE(random_state=42), SMOTE(random_state=42)]:
    X_train, y_train = smote.fit_resample(X_train_, y_train_)
    svc = LinearSVC(random_state=42, C=10).fit(X_train_, y_train_)
    print(confusion_matrix(y_test, svc.predict(X_test)))
    dtr = DecisionTreeClassifier(random_state=42).fit(X_train_, y_train_)
    print(confusion_matrix(y_test, dtr.predict(X_test)))
    lgr = LogisticRegression(random_state=42, penalty='none').fit(X_train_, y_train_)
    print(confusion_matrix(y_test, lgr.predict(X_test)))
    print("\n\n\n")





svc = LinearSVC(random_state=42, C=10).fit(X_train_, y_train_)
dtr = DecisionTreeClassifier(random_state=42).fit(X_train_, y_train_)
lgr = LogisticRegression(random_state=42, penalty='none').fit(X_train_, y_train_)




# In[39]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [1, 10, 100, 1000] }
svc = LinearSVC(random_state=42) 
svc_clf = GridSearchCV(svc, param_grid)
svc_clf.fit(X_train_, y_train_)
print(svc_clf.best_params_)


# In[28]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [1, 10, 100, 1000] }
lgr = LogisticRegression(random_state=42)
lgr_clf = GridSearchCV(svc, param_grid)
lgr_clf.fit(X_train_, y_train_)
print(lgr_clf.best_params_)


# In[89]:


from sklearn.model_selection import GridSearchCV
param_grid = {"criterion": ['gini', 'entropy'],
             "max_features": ["auto","sqrt", "log2"]}
dtr = DecisionTreeClassifier(random_state=42)
dtr_clf = GridSearchCV(dtr, param_grid)
dtr_clf.fit(X_train_, y_train_)
print(dtr_clf.best_params_)




# In[132]:


svc = LinearSVC(random_state=42, C=10).fit(X_train_, y_train_)
confusion_matrix(y_test, svc.predict(X_test))


# In[93]:


dtr = DecisionTreeClassifier(random_state=42, criterion="entropy").fit(X_train_, y_train_)
confusion_matrix(y_test, dtr.predict(X_test))


# In[94]:


lgr = LogisticRegression(random_state=42, C=10).fit(X_train_, y_train_)
confusion_matrix(y_test, lgr.predict(X_test))


# In[133]:


from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
eclf = VotingClassifier(estimators=[('lr', lgr), ('sv', svc)], 
                        voting='hard',
                        flatten_transform=True)


# In[134]:


eclf.fit(X_train_, y_train_)
confusion_matrix(y_test, eclf.predict(X_test))


# In[135]:


print(classification_report(y_test, eclf.predict(X_test)))


# In[136]:


from joblib import dump
dump(eclf, "../assets/models/ClassImbalance.pkl")

