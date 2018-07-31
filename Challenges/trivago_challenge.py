# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:54:31 2018

@author: geoadom

Exploratory Data Analysis
   Asessing data quality
   Determining the ratio of label classes
Building model
  CrossValidation (CV): training & validation sets partion. 5-fold CV.
  ML algoritms: Random Forest, SVM & Logistic Regression
  Optimal Model: GridSearch, penalty L1 & L2, different Matthews Correlation Coeffient (MCC)
  
  
The MCC is used in machine learning as a measure of the quality of binary classifications. 
The statistic is also known as the phi coefficient.
"""

import pandas as pd
from sklearn.metrics import matthews_corrcoef
import os
import pandas_profiling

from sklearn.model_selection import train_test_split


#Reading training and test data
train_actions = pd.read_csv("trivago_dataset_actions_bookings/case_study_actions_train.csv",sep="\t")
train_bookings = pd.read_csv("trivago_dataset_actions_bookings/case_study_bookings_train.csv",sep="\t")

test_actions = pd.read_csv("trivago_dataset_actions_bookings/case_study_actions_target.csv",sep="\t")
test_bookings = pd.read_csv("trivago_dataset_actions_bookings/case_study_bookings_target.csv",sep="\t")

#Exploratory data analysis (EDA)
#Descriptive statistics
train_actions.describe()
train_bookings.describe()

#Data Profiling using pandas_profiling package
###Actions profile
profile = pandas_profiling.ProfileReport(train_actions)
#list of variables rejected due to high correlation
rejected_variables = profile.get_rejected_variables(threshold=0.9)

#generate a report
profile.to_file(outputfile="actionsProfile.html")

###Bookings profile
profile = pandas_profiling.ProfileReport(train_bookings)
#list of variables rejected due to high correlation
rejected_variables = profile.get_rejected_variables(threshold=0.9)

#generate a report
profile.to_file(outputfile="bookingsProfile.html")

##########################################################################################################
##Preparing files to work with
##Create one files with all features/variables from users' action & users' booking
##########################################################################################################
#count times user make an action
actionsByUser = train_actions.groupby([ 'user_id', 'session_id', 'ymd','action_id', 'reference', 'step']).size().reset_index()

#Merging data to crea one for training and one for test
#Traing data
#Merge actions and booking (both datsets) based on user_id, 'session_id', 'ymd' (merge keys)
train = pd.merge(train_actions,train_bookings, how='left', on=['user_id', 'session_id', 'ymd'], suffixes=('_action', '_booking'))
#Saving training data set 
train.to_csv("train.csv")

#Test data
#merge both datset based on user_id, 'session_id', 'ymd'
train = pd.merge(test_actions,test_bookings, how='left', on=['user_id', 'session_id', 'ymd'], suffixes=('_action', '_booking'))

train.to_csv("test.csv")

##########################################################################################################
# Determine ratio of classes' label
##########################################################################################################
os.chdir("M:/TrivagoCallange")
data = pd.read_csv("trivago_dataset_actions_bookings/train.csv",dtype = str)

#Get a copy of training data on X and labels on y
X = data[['ymd', 'user_id', 'session_id', 'action_id', 'reference',
       'step', 'referer_code', 'is_app', 'agent_id', 'traffic_type']].copy()
#Labels
y= data['has_booking'].copy()

#Checking for imbalance labels
#Counting  instances of each class
# Some basic stats on the target variable
#values count
target_count = y.value_counts()

print ('# booking = {}'.format(target_count[1])  )
# booking = 762309

print ('# no booking = {}'.format(target_count[0]))
# no booking = 5100554

print ('% no booking = {}%'.format(round(float(target_count[0]) / len(y) * 100), 3) ) 
# % no booking = 87%

print ('% booking = {}%'.format(round(float(target_count[1]) / len(y) * 100), 3) ) 
# % b   ooking = 13%

print ('% ratio = {}'.format(round(target_count[0] /target_count[1] , 2) ) )

########################################################################
#Traing process
#######################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report

#1) Use crossvalidation to divide data into training set and validation set
#2) Use 3 ML algorithm (SVM, logistic regression and random forest) to create a classifier for each
#3) Use Grid Search to find the optimal parameters/hyer-parameters for each algorithm
#Note: Regularisation penalty were used for SVM and logistic regression which can deal with imbalance classes 
#Random forest algoritm is robust to imbalance classes

#Cross-validadtion
#Take 70% of the data as training data, and the remainder 30% as validation dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

#Create the pipeline
pipe = Pipeline( steps=[('svm', svm.SVC() ), 
                        ('log', LogisticRegression()), 
                        ('rf',RandomForestClassifier()) ] )


#Set the set of parameter for Grid search
param_options = [
    {
        'svm__C': [0.01, 0.1, 1.0],
        'svm__kernel': ['linear','rbf', 'poly'],
        'svm__gamma': [0.01, 0.1, 1.0],
        'svm__class_weight': ['balanced','{1: 10}',None]

    },
    {
        'log__C': [0.01, 0.1, 1.0],
        'log_penalty': ['l1','l2'],
        'log__class_weight ': ['balanced','{1: 10}',None],
        'log__solver':['liblinear', 'sag', 'saga','newton-cg', 'lbfgs']
    },
    {
      'rf__n_estimators': [10, 100,500,1000],
      'rf__criterion': ['gini', 'entropy'],
      'rf__max_features': ['auto', 'log2', 'sqrt', None],
      'r__max_depth': [2,5,8,0.25,0.5,0.75]

    }
]

#Grid serach return the optimal classifier based on different parameters
#using 5-fold cross validation 
#This is a time consuming task
grid = GridSearchCV(pipe, cv=5, n_jobs=2, param_grid=param_options)
#create the model using training set
grid.fit(X_train,y_train)

#predict using the validation set
y_pred = grid.predict(X_val)
#Create a performance report
report = classification_report( y_val, y_pred )
#Calculate matthews correlation coefficient
mc = matthews_corrcoef(y_val, y_pred)

############################################
##Using imblance package
from sklearn.utils import resample
from imblearn import over_sampling as os
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

#Logistic Regression
#Notes: C : Inverse of regularization strength; must be a positive float. 
#           Like in support vector machines, smaller values specify stronger regularization.
#class_weight : dict or ‘balanced’, default: None
#Weights associated with classes in the form {class_label: weight}. 
#If not given, all classes are supposed to have weight one.

pipe = Pipeline( steps=[('sampling', RandomUnderSampler() ), 
                        ('log', LogisticRegression())
                        ] )

params =[{
        'C': [0.01, 0.1, 1.0],
        'penalty': ['l1','l2'],
        'class_weight': ['balanced',None],
        'solver':['liblinear', 'sag', 'saga','newton-cg', 'lbfgs']
    }]
grid = GridSearchCV(LogisticRegression(), cv=5, n_jobs=2, param_grid=params)
grid.fit(X_train,y_train)

y_pred = grid.predict(X_val)
report = classification_report( y_val, y_pred )
mc = matthews_corrcoef(y_val, y_pred)

#Random forests
params =[{
      'n_estimators': [10, 100,500,1000],
      'criterion': ['gini', 'entropy'],
      'max_features': ['auto', 'log2', 'sqrt', None],
      'max_depth': [2,5,8,0.25,0.5,0.75]

    }]
grid = GridSearchCV(RandomForestClassifier(), cv=5, n_jobs=2, param_grid=params)
grid.fit(X_train,y_train)

y_pred = grid.predict(X_val)
report = classification_report( y_val, y_pred )
mc = matthews_corrcoef(y_val, y_pred)

#SVM
# fit the model and get the separating hyperplane
params =[{
        'C': [0.01, 0.1, 1.0],
        'kernel': ['linear','rbf', 'poly'],
        'gamma': [0.01, 0.1, 1.0],
        'class_weight': ['balanced',None]

    }]
grid = GridSearchCV(svm.SVC(), cv=5, n_jobs=2, param_grid=params)
grid.fit(X_train,y_train)

y_pred = grid.predict(X_val)
report = classification_report( y_val, y_pred )
mc = matthews_corrcoef(y_val, y_pred)

############################################################################
## Testing process
## Build a model on the whole training set using the optimal model based on performance on the validation set 
## Random Forest got the higest MCC 
## Predict classes on unseen data set: the given test set
############################################################################
data_test = pd.read_csv("trivago_dataset_actions_bookings/test.csv")

X_test = data_test[['ymd', 'user_id', 'session_id', 'action_id', 'reference',
       'step', 'referer_code', 'is_app', 'agent_id', 'traffic_type']].copy()

data = pd.read_csv("trivago_dataset_actions_bookings/train.csv",dtype = str)

X = data[['ymd', 'user_id', 'session_id', 'action_id', 'reference',
       'step', 'referer_code', 'is_app', 'agent_id', 'traffic_type']].copy()
y= data['has_booking'].copy()

#Random forest with optimal parameters
clf_rf = RandomForestClassifier(n_estimators=50,
                                bootstrap=True, 
                                class_weight=None, 
                                criterion='gini',
                                oob_score=True,
                                min_impurity_split=1e-07
                                )
clf_rf.fit(X,y)
pred=clf_rf.predict(X_test)

##########################################################################
#Submission file
############################################
pred = pd.DataFrame(data=pred)
pred = pred.rename(columns={ 0:'has_booking'})
session_id = X_test.session_id

submission = pd.concat([session_id,pred],axis=1)
submission.to_csv('prediction.csv', index=False)
