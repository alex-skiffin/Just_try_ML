import numpy
import pandas

print("load data")
trainNP=numpy.genfromtxt("train.csv", delimiter=",", dtype="U75", skip_header=1)
testNP=numpy.genfromtxt("test.csv", delimiter=",", dtype="U75", skip_header=1)
testNP=testNP[:,1:]
testNP[testNP == '']='0'
print(len(testNP[0,:]))
print(len(trainNP[0,:-1]))
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#le.fit(trainNP[1])
#print(le.transform(trainNP[1]))43973

allDataForNormalize=trainNP[:,:-1]

#numpy.savetxt("allDataForNormalize.csv", allDataForNormalize, delimiter=",")
#numpy.savetxt("testNP.csv", testNP, delimiter=",")
#temp=numpy.append(allDataForNormalize[:,0],testNP[:,0])
#print(len(temp))
#le.fit(temp)
#train=pandas.read_csv("cleaned2.csv")
#dataset = np.loadtxt(train, delimiter=",", dtype='str')
# separate the data from the target attributes
#print (len(dataset))
#print (len(train))
print("normalize data")
i=0
while i <= (len(allDataForNormalize[0,:])-1):
	temp=numpy.append(allDataForNormalize[:,i],testNP[:,i])
	le.fit(temp)
	print(le.transform(temp))
	allDataForNormalize[:,i]=le.transform(temp[:50000])
	testNP[:,i]=le.transform(temp[50000:])
	print (testNP[:,i])
	print (allDataForNormalize[:,i])
	i=i+1

	
	
print (str(len(allDataForNormalize[0,:])) +'		'+str(len(allDataForNormalize[:,0])))
print (str(len(testNP[0,:])) +'		'+str(len(testNP[:,0])))
print ("save to file")
numpy.savetxt("testNP.csv", testNP.astype(float), delimiter=",")
numpy.savetxt("allDataForNormalize.csv", allDataForNormalize.astype(float), delimiter=",")
XData = allDataForNormalize[:50000,:]
#numpy.nan_to_num(testNP)#.fillna(0)
YTarget = trainNP[:,-1]
#allDataForNormalize[:50000,-1]
print (str(len(allDataForNormalize[0,:])) +'		'+str(len(allDataForNormalize[:,0])))
print (str(len(XData[0,:])) +'		'+str(len(XData[:,0])))
print (str(len(YTarget)) +'		'+str(len(YTarget)))
print (str(len(testNP[0,:])) +'		'+str(len(testNP[:,0])))
print (testNP[-1,:])

print("fit clf data")
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(XData, YTarget)
print("predict clf data")
testCLFResult=clf.predict(testNP)
result=[[0 for x in range(2)] for x in range(63)] 
result[0][0]="ID"
result[0][1]="y"
i=1
for el in testCLFResult:
	result[i][1]=el
	result[i][0]=i-1
	i=i+1

numpy.savetxt("CLFresult.csv", testCLFResult.astype(float), delimiter=",")
numpy.savetxt("CLFresult2.csv", result.astype(float), delimiter=",")
#print(len(X))
#from sklearn.naive_bayes import GaussianNB
#gnb = GaussianNB()
#y_pred = gnb.fit(X, Y)#.predict(X)
#print(y_pred)

print("normalized_X")
from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(XData.astype(float))
# standardize the data attributes
#standardized_X = preprocessing.scale(X)

print("GaussianNB")
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(XData, YTarget)
print(model)
# make predictions
expected = YTarget
predicted = model.predict(testNP)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

numpy.savetxt("GaussianNBresult.csv", predicted.astype(float), delimiter=",")
