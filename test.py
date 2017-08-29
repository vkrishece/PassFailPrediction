from __future__ import division
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib

modelName="predictPassOrNot.pkl"
#LOADING DATA
student_data = pd.read_csv("student-data.csv")
print "Student data loaded successfully!"



#Getting knowledge on data
n_students, n_features = student_data.shape
n_passed, _ = student_data[student_data['passed'] == 'yes'].shape
n_failed, _ = student_data[student_data['passed'] == 'no'].shape
grad_rate = (n_passed / n_students) * 100
#print "\n\n\n------------------"
#print "Understanding Data"
#print "------------------\n\n\n"
#print "Total number of students in class: {}".format(n_students)
#print "Number of students passed: {}".format(n_passed)
#print "Number of students failed: {}".format(n_failed)
#print "Graduation rate of the class: {:.2f}%".format(grad_rate)
# Check for data imbalance, will deal with this by statifying the data when splitting
y_pred = ['yes']*n_students
y_true = ['yes']*n_passed + ['no']*n_failed
score = f1_score(y_true, y_pred, pos_label='yes', average='binary')
#print "F1 score for all 'yes' on students: {:.4f}".format(score)


# Splitting to X and Y
feature_cols = list(student_data.columns[:-1]) 
target_col = student_data.columns[-1]
X_all = student_data[feature_cols]
y_all = student_data[target_col]
#Preprocess_features
def preprocess_features(X):
    output = pd.DataFrame(index = X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)  
        output = output.join(col_data)
    return output
X_all = preprocess_features(X_all)



X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify=y_all,test_size=0.24, random_state=42)
#print "Training set has {} samples.".format(X_train.shape[0])
#print "Testing set has {} samples.".format(X_test.shape[0])
#print "Grad rate of the training set: {:.2f}%".format(100 * (y_train == 'yes').mean())
#print "Grad rate of the testing set: {:.2f}%".format(100 * (y_test == 'yes').mean())

#print "\n\n\n------------------"



def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    #print "Predicted as :",y_pred[0]
    target=target.values
    #print "TARGET",target[0]
    #print "----------"
    #print "Made predictions in {:.4f} seconds.".format(end - start)
    return y_pred[0],target[0]



clf = joblib.load('predictPassOrNot.pkl')
print "Loaded Model predictPassOrNot.pkl"
#train_predict(clf, X_train, y_train, X_test, y_test)
#print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
#print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))

print "Hint : Ignore the Header and give the input."
print "       1 is the first row"
rowTobeChecked=raw_input('Enter the row in CSV file to be checked  : ')
print "Predicting for ",rowTobeChecked," th row"
rowTobeChecked=int(rowTobeChecked)-1
xData=X_all.iloc[[rowTobeChecked]]
yData=y_all.iloc[[rowTobeChecked]]
predictedData,actualData=predict_labels(clf, xData,yData)
pred,act="",""
print "------------------------------------------------"
if(predictedData in actualData ):
	if("yes" in predictedData):
		pred="passed"
	else:
		pred="didn't passed"
	if("yes" in actualData):
		act="passed"
	else:
		act="didn't passed"
	print "Correctly Identified the Result for the given input"
	#print "Actually ",pred," Predicted As ",act

else:
	if("yes" in predictedData):
		pred="passed"
	elif("no" in predictedData):
		pred="didn't passed"
	if("yes" in actualData):
		act="passed"
	elif("no" in actualData):
		act="didn't passed"
	print "Sorry!!Wrongly Identified the Result for the given input"
print "\n--------------------------------------"
print "Actually |",act," Predicted As |",pred
print "--------------------------------------\n\n"
