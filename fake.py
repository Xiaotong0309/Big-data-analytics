import pandas as pd
import numpy as np
import io
import requests
def train_random_forest(dataset):
    #devide data into features and labels
    X = dataset.iloc[:, 0:4].values
    y = dataset.iloc[:, 4].values
    #split datasset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    #feature scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    '''
    #training for regression
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    #evaluating for regression
    from sklearn import metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    '''

    #training for classification
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=200)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    #evaluating for classification
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)

def train_boosting(dataset):
    #devide data into features and labels
    X = dataset.iloc[:, 0:4].values
    y = dataset.iloc[:, 4].values
    #split datasset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    #feature scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #training for classification
    from sklearn.ensemble import GradientBoostingClassifier
    clf=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    #evaluating for classification
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)

def train_knn(dataset):
    #devide data into features and labels
    X = dataset.iloc[:, 0:4].values
    y = dataset.iloc[:, 4].values
    #split datasset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    #feature scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #training for classification
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    #evaluating for classification
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)

def train_bagging(dataset):
    #devide data into features and labels
    X = dataset.iloc[:, 0:4].values
    y = dataset.iloc[:, 4].values
    #split datasset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    #feature scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #training for classification
    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    #evaluating for classification
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)
'''
#get data from url
url="https://drive.google.com/file/d/1mVmGNx6cbfvRHCDvF12ZL3wGLSHD9f/view"
s=requests.get(url).content
dataset=pd.read_csv(io.StringIO(s.decode('utf-8')))
'''
#get data from local file
path_regress = "/Users/xiaotongdiao/Desktop/fake/petrol_consumption.csv"
path_classify = "/Users/xiaotongdiao/Desktop/fake/bill_authentication.csv"
dataset = pd.read_csv(path_classify)
#print(dataset.head())
train_random_forest(dataset)
train_boosting(dataset)
train_knn(dataset)
train_bagging(dataset)
