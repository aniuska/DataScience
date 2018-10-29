
# Trivago Challenge


### Summary

The task is to build a machine learning model to estimate if a booking occurred. The acuracy of the prediction will evaluated by Matthews Correlation Coefficient (MCC). This is a binary classification problem (with class imbalanced)to estimate booking occurrence/ no booking. There machine learning algorithm were evaluated for this task. Random Forest, regularised Logistic regression and regularised SVM were used. The best performance was achieved by Random Forest.

## Methodology

A new dataset,called train.csv, was create merging case_study_actions_train.csv and case_study_bookings_train.csv to apply the machine learning algoritms to this dataset.

See below the step followed in this challenge



```python
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import os
```

Reading the provided datasets for training  


```python
train_actions = pd.read_csv("trivago_dataset_actions_bookings/case_study_actions_train.csv",sep="\t")
train_bookings = pd.read_csv("trivago_dataset_actions_bookings/case_study_bookings_train.csv",sep="\t")

```

Merging the train_actions train_bookings datasets. Saving the resulted dataset into csv file for reusing the data later


```python
train = pd.merge(train_actions,train_bookings, 
                 how='left', 
                 on=['user_id', 'session_id', 'ymd'], 
                 suffixes=('_action', '_booking'))

train.to_csv("train.csv")
```

Open the training set from train.csv. I will be working with the train.csv from now


```python
data = pd.read_csv("trivago_dataset_actions_bookings/train.csv",dtype = str)
#Convert ymd to datatime
data['ymd'] = pd.to_datetime(data['ymd'], format='%Y-%m-%d' )

#data
X = data[['ymd', 'user_id', 'session_id', 'action_id', 'reference',
          'step', 'referer_code', 'is_app', 'agent_id', 'traffic_type']].copy()

#target
y = data['has_booking'].copy()

X.describe()

```

Is there imbalances between outcome classes?


```python
#number of observations (rows)
print ('# rows = {}'.format(len(y)) )
#5862863

target_count = y.value_counts()

print ('# booking = {}'.format(target_count[1])  )
# booking = 762309

print ('# no booking = {}'.format(target_count[0]))
# no booking = 5100554

print ('# % no booking = {}%'.format(round(float(target_count[0]) / len(y) * 100), 3) ) 
# % no booking = 87%

print ('# % booking = {}%'.format(round(float(target_count[1]) / len(y) * 100), 3) ) 
# % booking = 13%

print ('# ratio = {}'.format(round(target_count[0] /target_count[1] , 2) ) )
# ratio = 6.69
```

## What makes the classification problem difficult in this task?

The class (target) imbalance.

This is an imbalanced dataset. Given that the number of observations (rows) is 5862863, we get 87% of no booking whereas the booking percentage is just 13. Therefore, the ratio of 'No booking' to 'Booking' is roughly 6:1 (5100554:762309). This is a huge issue for the classification challenge.

## How do you handle that?

Using ML algoritms that deal with class imbalance as 
* Random Forest, 
* regularised logistic regression with balanced class parameter and
* regularised LinearSVC with balanced class parameter

In addition, resampling algorithms and Generate Synthetic Samples would has been applied to handle this issue. As the dataset has large number of rows (observations) (big dataset), the under-sampling would has been used - trying both random and non-random sampling. 

** GridSearchCV** was used to find optimal parametes for the ML techniques used. **Random Forest** gave the best performance for both **MCC coef (0.83)** and **MSE (0.95)** in the validation set. Therefore Random Forest was employed for the precdition using the test set. See result in the table below.

![ML Algoriyhms Performance](images/ML-comparation.png)

## General approach followed


```python
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
#2) Use 3 ML algorithm (SVM, logistic regression and random forest) to create a classifier for each one
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

#Grid search return the optimal classifier based on different parameters
#using 5-fold cross validation (a time consuming task)

grid = GridSearchCV(pipe, cv=5, n_jobs=2, param_grid=param_options)
#create the model using training set
grid.fit(X_train,y_train)

#predict using the validation set
y_pred = grid.predict(X_val)

#Create a performance report
report = classification_report( y_val, y_pred )

#Calculate matthews correlation coefficient
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
```

## The 3 more significant features for prediction

(As the result ofor Logist regression were very low, it will not be analysed for variable importance)

**LinearSVC's coefficents**

[    
  'ymd':6.58401017e-09,   
  'user_id':0.00000000e+00,   
  'session_id':0.00000000e+00,  
  'action_id':1.92699472e-05,   
  'reference':6.67675489e-10,   
  'step':5.89627227e-04,   
  'referer_code': -1.89425583e-04,  
  'is_app': -2.57709608e-01,  
  'agent_id': -3.09317086e-02,    
  'traffic_type': -1.02952371e-01     
  ]


**Random forest variable importance plot**

![Random forest variable importance](images/importance-rf2.png)

**LinearSVC** found *'user_id'* and *'session_id'* irrelevants, gaving them zero value to their coefficients  - due the L1 reguralised penalty used. 

Taking into account only the positive relationship with the outcome *'has_booking'* (positive coefficients), the *'step'* variable is the most important with coefficient value of 5.89627227e-04, followed by *'action_id'* (1.92699472e-05) and *'ymd'* (6.58401017e-09) variables respectively. The fourth most relevent variable is *'reference'* with coeficent value of 6.67675489e-10. 

However, Random forest (RF)'s variable importance said diffrent story (see plot above). RF said that the more important variables are *'user_id'* and *'session_id'*, followed by *'reference'*. Without taking into account the *'user_id'* and *'session_id'* variables, we got that *'reference'* variable is fallowed by *'ymd '* and *'step'*.

In conclusion, I would say that the most relevant features for the prediction are *'reference','ymd and 'step'* variables.

## What might the very significant action type refer to?

After concat the actions datasets (train & target), the 5 more frequent action_ids were (related to the content):

    action_ids | count
    -----------|--------
    2142       | 1264950
    2113       |  753448
    8001       |  375075
    2116       |  308160
    6999       |  278243

And the first 5 action_ids with frequency over 15 were (related to function of the website):

    action_ids | count
    -----------|--------
     2148      |  16
     2191      |  17
     2715      |  22
     2382      |  22
     2882      |  25
     

The action and booking datasets were merge to analyse the **action_id equal 2142**, and the counts for **referer_code** and reference were:

   referer_code | count
   -------------|-----------
           1    | 607654
           0    | 384175
           99   | 79676
           11   | 34106
           15   | 20902
           10   | 14337
           24   | 8471
           21   |  888
           23   |  406
           17   |   27

 reference | count
 ----------|----------
 1321090   |  1773 
 1455251   |  1699 
 342836    |  1201
 1700399   |  1095 
 32940     |  1042 
 2055010   |  1027



```python
data = data.merge(df_actions,how='left', on=['user_id', 'session_id', 'ymd'], suffixes=('_action', '_booking'))

row_data = data.loc[data.action_id == 2142]

grouped = row_data.groupby(by=['referer_code','reference']).size().reset_index()
grouped = grouped.rename(columns={0: 'count'})
grouped.sort_values(by=['count'],inplace=True,ascending=False)
```


```python
references = row_data.groupby(by=['reference']).size().reset_index().rename(columns={0: 'count'})
references.sort_values(by=['count'],inplace=True,ascending=False)
```

The action and booking datasets were merge to analyse the **action_id equal 2142**, and the counts for **referer_code** and reference were:

 reference  |   count
 -----------|----------
 1321090    |    1773  
 1455251    |    1699  
 342836     |    1201  
 1700399    |    1095  
 32940      |    1042  
 2055010    |    1027



```python
referer = row_data.groupby(by=['referer_code']).size().reset_index().rename(columns={0: 'count'})
referer.sort_values(by=['count'],inplace=True,ascending=False)
```

   referer_code  |  count
   --------------|--------
           1     | 607654
           0     | 384175
           99    | 79676
           11    | 34106
           15    | 20902
           10    | 14337
           24    | 8471
           21    | 888
           23    | 406
           17    |  27

Analysing **action_id == 2142 && action_id == 2148** (biggest set of refrence values & smallest reference set of values respectively).


```python
group_bookings = data[(data.action_id.isin([2142,2148] ))].copy()

group_bookings = group_bookings.groupby(by=['action_id','reference','has_booking']).size().reset_index().rename(columns={0: 'count'})
group_bookings.sort_values(by=['count'],inplace=True,ascending=False)
group_bookings.sort_values(by=['count','has_booking'],inplace=True,ascending=False)

group_bookings_booked = group_bookings[(group_bookings.has_booking == 1)]
group_bookings_nobooked = group_bookings[(group_bookings.has_booking == 0)]

```

**action_id equal 2142** that has led to bookings: The refrence with higthest booking is *183681* however the refrence *1321090* has led to only 292. Whereas the refrences *1455251* and *1321090* did not lead to bookings.

![Counts](images/counts.png)
![Action ID](images/2142-2.png)
![Action ID no booking](images/2142-nobooking.png)

Note: Cannot say more as I don't have details of the meaning of the reference codes 


```python
p2148 = p2148.groupby(by=['action_id','reference','has_booking']).size().reset_index().rename(columns={0: 'count'})
p2148.sort_values(by=['count'],inplace=True,ascending=False)
p2148.sort_values(by=['count','has_booking'],inplace=True,ascending=False)

```

However the **action_id equal 2148** that has lead to 3 bookings 
![Action Id](images/2148.png)


## Futher work

The machine learning algorithms used in our approach dealed with the class imbalance problem via the 'class_weight' parameter during training process. However, another approach could be to make a prior balance of the class frequency using resampling algorithms and/or Generate Synthetic Samples to deal with this issue.   

There are two resampling methods susch as down-sampling and up-sampling. In this case, the dataset has a large number of observations therefore the down-sampling - trying both random and non-random sampling - is a good option to get a balance class frequency. 



```python

```
