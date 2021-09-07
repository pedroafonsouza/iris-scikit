#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Atividade Prática 01
=========================================================
Professor: Carlos Severiano
Aluno: Pedro Afonso Ramos de Souza
Curso: BSI
Disciplina: Inteligência Artificial

"""


print(__doc__)

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
from sklearn.datasets import load_iris


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Tamanho sépala')
plt.ylabel('Largura sépala')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

resultDecisionTreeTotal = 0
for i in range(30):
    X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'],
                                                        train_size=0.6, test_size=0.4)

    clf = DecisionTreeClassifier(max_depth=2,
                                 random_state=0)
    clf.fit(X_train, Y_train)
    decisionTreePredict = clf.predict(X_test)
    resultDecisionTree = accuracy_score(Y_test, decisionTreePredict)
    resultDecisionTreeTotal += resultDecisionTree

    print('DecisionTree - ', i, ' - ', resultDecisionTree)
    tree.plot_tree(clf);

    # fn = ['Tamanho sépala(cm)', 'Largura sépala (cm)', 'Tamanho pétala (cm)', 'Largura pétala (cm)']
    # cn = ['Setosa', 'Versicolor', 'Virginica']
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    # tree.plot_tree(clf,
    #                feature_names=fn,
    #                class_names=cn,
    #                filled=True);
    # tree.export_graphviz(clf,
    #                      out_file="tree.dot",
    #                      feature_names=fn,
    #                      class_names=cn,
    #                      filled=True)

    # !dot -Tpng -Gdpi=300 tree.dot -o tree.png

print('Result Decision Tree: ', resultDecisionTreeTotal / 30)

print(sep='---------------------------')

resultGaussianTotal = 0
for i in range(30):
    X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'],
                                                        train_size=0.6, test_size=0.4)
    model = GaussianNB()
    model.fit(X_train, Y_train)
    gaussianPredict = model.predict(X_test)
    resultGaussian = accuracy_score(Y_test, gaussianPredict)
    resultGaussianTotal += resultGaussian
    print('GaussianNB - ', i, ' - ', resultGaussian)

print('Result GaussianNB: ', resultGaussianTotal / 30)



