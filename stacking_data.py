import pandas as pd

import numpy

import numpy as np

import pandas as pd

import  sklearn as sk

import sklearn.tree

import sklearn.linear_model

import sklearn.datasets

from sklearn.datasets import load_breast_cancer

r1 = sk.linear_model.LogisticRegression(random_state=0)

r2 = sk.tree.DecisionTreeClassifier(random_state=0, min_samples_split=8, min_samples_leaf=4)

r3 = sk.tree.DecisionTreeClassifier(random_state=0, min_samples_split=7, min_samples_leaf=4)

r4 = sk.tree.DecisionTreeClassifier(random_state=0, min_samples_split=12, min_samples_leaf=4)

model_list = [r1,r2,r3,r4]



X_full, y_full = sklearn.datasets.load_breast_cancer(return_X_y=True)

X, Xt, y, yt = sklearn.model_selection.train_test_split(X_full, y_full, test_size=0.3, random_state=0)

# X, Xt and y for stacking
import random as rd
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = rd.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
            fold = np.array(fold)
        dataset_split.append(fold)
    return dataset_split

#estimator = list of models
#



def stack_pred(estimator, X, y, Xt, k=3, method='predict'):
# записываем предсказания для test для обученных данных
    models = list()
    count = 0
    sX = numpy.zeros((len(X),))
    models_mean = list()
    index_t = list()
    sXt = list()

    if len(estimator) < k:
        k = len(estimator)

    kf = sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=0)
    # берем для каждой модели только часть данных от train
    for index_list, model in enumerate(estimator):
        # на всех кроме одной обучаем модель
        for train_index, test_index in kf.split(X):
            if index_list != count:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                #обучаем на тестовых данных
                model.fit(X_train, y_train)
            else:
            #на одной деаем предсказание
                index_t = train_index
            count +=1
        # деаем предсказание на test
        pred = np.array(model.predict(Xt))
        if (index_list ==0):
            sXt = pred
        else:
            sXt = np.append(sXt,pred)
            # на одной деаем предсказание
            sX[index_t] = model.predict(X[index_t])
        count = 0
    #усредняем предсказание на test
    sXt =  sXt/len(model_list)
    sX = np.array(sX)
    return (sX,sXt)



stack_pred(model_list,X,y, Xt)