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


def model_select(model, train, predict, index):

    X_train = train.iloc[:, 0:index].values
    y_train = train.iloc[:, index].values
    X_test = predict.iloc[:, 0:index].values

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
    return y_pred

p1 = "data/Patrick_Shyu/private.csv"
p2 = "data/Patrick_Shyu/public.csv"
t1 = pd.read_csv("clean/user_all.csv")
t2 = pd.read_csv("clean/tweet_public.csv")
d1 = pd.read_csv(p1)
d2 = pd.read_csv(p2)

pred_pri = model_select("random_forest", t1, d1, 6)
pred_pub = model_select("random_forest", t2, d2, 11)

df = pd.read_csv(p1)
df['predict'] = pred_pri
df.to_csv(p1, index=False)
df = pd.read_csv(p2)
df['predict'] = pred_pub
df.to_csv(p2, index=False)
