import numpy as np
import pandas


train=pandas.read_csv("cleaned2.csv")
#dataset = np.loadtxt(train, delimiter=",", dtype='str')
# separate the data from the target attributes
#print (len(dataset))
#print (len(train))
X = train[1:][1:-1]
Y = train[-1:]
#print(len(X))
#from sklearn.naive_bayes import GaussianNB
#gnb = GaussianNB()
#y_pred = gnb.fit(X, Y)#.predict(X)
#print(y_pred)

from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
#standardized_X = preprocessing.scale(X)

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, Y)
print(model)
# make predictions
expected = Y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))