import numpy as np
import pandas


train=pandas.read_csv("cleaned.csv")
dataset = np.loadtxt(train, delimiter=",", dtype='str')
# separate the data from the target attributes
X = dataset[1:-1]
Y = dataset[-1:]


from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X, axis=1)
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