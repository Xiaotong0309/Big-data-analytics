import pandas as pd
import numpy as np
import io
import requests
import math
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def model_select(model, dataset, index, split_size, delete_feature):
    #devide data into features and labels
    X = dataset.iloc[:, 0:index].values
    y = dataset.iloc[:, index].values
    #split datasset

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=0)
    #feature scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #training for classification
    if model == "random_forest":
        clf=RandomForestClassifier(n_estimators=200)
    elif model == "boosting":
        clf=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
    elif model == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model == "bagging":
        clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    elif model == "svm":
        clf = SVC(gamma='auto')
    elif model == "gpc":
        kernel = 1.0 * RBF(1.0)
        clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
    elif model == "nb":
        clf = GaussianNB()
    elif model == "quadratic_discriminant":
        clf = QuadraticDiscriminantAnalysis()

    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    #evaluating for classification
    #print(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
    #print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)



iter = 50


def init(length):
    data = {}
    data['all_user'] = [0.0]*length
    data['public_user'] = [0.0]*length
    data['user_with_tweet'] = [0.0]*length
    return data
def plot_acc(data, xlabel, ylabel, title, x_axis):
    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e']
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    labels = list(data.keys())
    for i in range(len(labels)):
        name = labels[i].replace('_', ' ')
        plt.plot(x_axis, data[labels[i]], color=color_sequence[i], label = name)
        plt.legend(loc=2, ncol=2)

    fig.suptitle(title, fontsize=18, ha='center')
    plt.show()
#plot split_size
def plot_test_size():
    x_axis = list(range(10, 70, 5))
    data = init(len(x_axis))
    for t in range(iter):
        for i in range(len(x_axis)):
            x_axis[i] = float(x_axis[i])/100
            acc = model_select("random_forest", pd.read_csv("clean_user_all.csv"), 6, x_axis[i], -1)
            data['all_user'][i] += acc
            acc = model_select("random_forest", pd.read_csv("clean_public_user.csv"), 6, x_axis[i], -1)
            data['public_user'][i] += acc
            acc = model_select("random_forest", pd.read_csv("clean_balance.csv"), 11, x_axis[i], -1)
            data['user_with_tweet'][i] += acc
    for i in range(len(x_axis)):
        data['all_user'][i] /= iter
        data['public_user'][i] /= iter
        data['user_with_tweet'][i] /= iter
    plot_acc(data, "test size", "accuracy", "Accuracy score under different test size", x_axis)
#plot model
def plot_model():
    model_list = ["random_forest", "bagging", "knn", "boosting", "nb", "svm", "quadratic_discriminant"]
    x_axis = model_list
    data = init(len(x_axis))
    for t in range(iter):
        for i in range(len(x_axis)):
            acc = model_select(x_axis[i], pd.read_csv("clean_user_all.csv"), 6, 0.3, -1)
            data['all_user'][i] += acc
            acc = model_select(x_axis[i], pd.read_csv("clean_public_user.csv"), 6, 0.3, -1)
            data['public_user'][i] += acc
            acc = model_select(x_axis[i], pd.read_csv("clean_balance.csv"), 11, 0.3, -1)
            data['user_with_tweet'][i] += acc
            print(x_axis[i])
    for i in range(len(x_axis)):
        data['all_user'][i] /= iter
        data['public_user'][i] /= iter
        data['user_with_tweet'][i] /= iter
    x_list = list(range(len(x_axis)))
    plot_acc(data, "model", "accuracy", "Accuracy score under different model", x_axis)
plot_split_size()
plot_model()
