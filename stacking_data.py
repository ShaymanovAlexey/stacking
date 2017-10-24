import pandas as pd

import numpy

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

kf = sklearn.model_selection.KFold(n_splits=4, shuffle=True, random_state=0)

X_full, y_full = sklearn.datasets.load_breast_cancer(return_X_y=True)

X, Xt, y, yt = sklearn.model_selection.train_test_split(X_full, y_full, test_size=0.3, random_state=0)

def stacking(model_list, train, test):
    # записываем предсказания для test для обученных данных
    models = list()
    count = 0
    sX = numpy.zeros((len(train),2))
    models_mean = list()
    index_t = list()
    sXt = pd.DataFrame([])
    # берем для каждой модели только часть данных от train
    for index_list, model in enumerate(model_list):
        # на всех кроме одной обучаем модель
        for train_index, test_index in kf.split(train):
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
        pred = pd.DataFrame(model.predict_proba(test))
        if (index_list ==0):
            sXt = pred
        else:
            sXt = sXt.add(pred,fill_value=0)
            # на одной деаем предсказание
            sX[index_t] = model.predict_proba(X[index_t])
        count = 0
    #усредняем предсказание на test
    sXt =  sXt/len(model_list)
    sX = pd.DataFrame(sX)
    return (sX,sXt)



stacking(model_list,X, Xt)