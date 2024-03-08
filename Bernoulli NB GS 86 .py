# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:21:49 2020

@author: AbdelAzeem
"""
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#----------------------------------------------------
#reading data

data = pd.read_csv('heart.csv')
#data.describe()

#X Data
X = data.iloc[:,:-1]
#y Data
y = data.iloc[:,-1]
print('X Data is \n' , X.head())
print('X shape is ' , X.shape)

# -------------- MinMaxScaler for Data --------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)

#---------Feature Selection = Logistic Regression 13=>7 -------------------

from sklearn.linear_model import  LogisticRegression

thismodel = LogisticRegression()


FeatureSelection = SelectFromModel(estimator = thismodel, max_features = None) # make sure that thismodel is well-defined
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
print('X Shape is ' , X.shape)
print('Selected Features are : ' , FeatureSelection.get_support())

#--------------------- Normalizing Data -------------------------------
#Normalizing Data

scaler = Normalizer(copy=True, norm='max') # you can change the norm to 'l1' or 'max' 
X = scaler.fit_transform(X)

#showing data
#print('X \n' , X[:10])
#print('y \n' , y[:10])


#------------ Splitting data ---33% Test  67% Training --------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)

#--------------------------------- Bernoulli NB classifier  54 % --------

#Import Libraries
from sklearn.naive_bayes import BernoulliNB


#Applying BernoulliNB Model 

'''
sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True,class_prior=None)
'''
import time
t0 = time.clock()
BernoulliNBModel = BernoulliNB(alpha=1.0,binarize=1)
BernoulliNBModel.fit(X_train, y_train)
tr = (time.clock()-t0)
#Calculating Details
print('BernoulliNBModel Train Score is : ' , BernoulliNBModel.score(X_train, y_train))
print('BernoulliNBModel Test Score is : ' , BernoulliNBModel.score(X_test, y_test))
print('Training Time is: ', tr*1000)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = BernoulliNBModel.predict(X_test)
y_pred_prob = BernoulliNBModel.predict_proba(X_test)
#print('Predicted Value for BernoulliNBModel is : ' , y_pred[:10])
#print('Prediction Probabilities Value for BernoulliNBModel is : ' , y_pred_prob[:10])

# ---------- confusion_matrix ----------
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)
TP = 54 # 11
TN = 0 # 00
FN = 0  # 01
FP = 46 # 10
accuracy_score = ((TP + TN) / float(TP + TN + FP + FN))*100
precision_score = (TP /float(TP + FP ))*100
recall_score = (TP / float(TP + FN))*100
f1_score = (2 * (precision_score * recall_score) / (precision_score + recall_score))
print('accuracy_score is :' , accuracy_score)
print('Precision Score is : ', precision_score)
print('recall_score is : ', recall_score)
print('f1_score is :' , f1_score)

from sklearn.metrics import roc_auc_score
#5 roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)

ROCAUCScore = roc_auc_score(y_test,y_pred, average='micro') #it can be : macro,weighted,samples
print('ROCAUC Score : ', ROCAUCScore*100)


#            ################# ((Grid Search)) ##############


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
import pandas as pd

SelectedModel = BernoulliNB()

'''
sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True,class_prior=None)

BernoulliNBModel = BernoulliNB(alpha=1.0,binarize=1)
'''


SelectedParameters = {'binarize':[0.0,0.3,0.4,1.1,0.5,0.6,0.7,0.8,0.9,0.10],
                       'alpha' : [0.3,1,10,20,30,60,100,200,300] }
import time
t0 = time.clock()
GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 10,return_train_score=True)

GridSearchModel.fit(X_train, y_train)
tr = (time.clock()-t0)
sorted(GridSearchModel.cv_results_.keys())

GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

# Showing Results
print('All Results are :\n', GridSearchResults )
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)
print('Best Estimator is :', GridSearchModel.best_estimator_)
print('Time in sec :',tr)





#--------- try 2 ----- 86 % ----- alpha=30, binarize=0.3, class_prior=None, fit_prior=True

from sklearn.naive_bayes import BernoulliNB


#Applying BernoulliNB Model 

'''
sklearn.naive_bayes.BernoulliNB(alpha=30, binarize=0.3, fit_prior=True,class_prior=None)
'''

BernoulliNBModel = BernoulliNB(alpha=30, binarize=0.3, class_prior=None, fit_prior=True)
BernoulliNBModel.fit(X_train, y_train)

#Calculating Details
print('BernoulliNBModel Train Score is : ' , BernoulliNBModel.score(X_train, y_train))
print('BernoulliNBModel Test Score is : ' , BernoulliNBModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = BernoulliNBModel.predict(X_test)
y_pred_prob = BernoulliNBModel.predict_proba(X_test)


# ------------------ Metrics ----------------------
# ---------- confusion_matrix ----------
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)



from sklearn.metrics import roc_auc_score
#5 roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)
TP = 53
TN = 25
FN = 1
FP = 21
accuracy_score = ((TP + TN) / float(TP + TN + FP + FN))*100
precision_score = (TP /float(TP + FP ))*100
recall_score = (TP / float(TP + FN))*100
f1_score = (2 * (precision_score * recall_score) / (precision_score + recall_score))
print('accuracy_score is :' , accuracy_score)
print('Precision Score is : ', precision_score)
print('recall_score is : ', recall_score)
print('f1_score is :' , f1_score)
ROCAUCScore = roc_auc_score(y_test,y_pred, average='micro') #it can be : macro,weighted,samples
print('ROCAUC Score : ', ROCAUCScore*100)
