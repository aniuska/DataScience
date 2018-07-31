"""
Coursework - Large-scale text classification
Task 3 - Classifiers NB, DT and LR
Aniuska Dominguez Ariosa
INM432 Big Data
MSc in Data Science (2014/15)
"""

import sys
import numpy as np
import pickle

from operator import add

from collections import defaultdict
from time import time

from pyspark import SparkContext
from pyspark.conf import SparkConf

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.util import MLUtils
from pyspark.mllib.classification import LogisticRegressionWithSGD

# replace the file paths with binary labels
#Parameters:
#   sl -      tuple (f,[subjects list])
#                  sl[0] = file path,
#                  sl[1] = subjects list
#   topSubj - subject classifier
#Return:
#  replace file path with 1 if topSubject is in [subjects list]
#  replace file path with 0 if topSubject is not in [subjects list]
def subjectLabel(sl,topSubj ):
    #subject in 10 top subjects
    if topSubj in sl[1]:
        return (1,sl[0]) #return tuple (contain subject, words weight list  )
    else:
        return (0,sl[0]) #return tuple (does not contain subject, words weight list  )

#Training classifiers
# print the different performance metrics
def accuracy(rm):
    resultMap = defaultdict(lambda :0,rm) # use of defaultdic saves checking for missing values
    total = sum(resultMap.values())
    truePos = resultMap[(1,1,)]
    trueNeg = resultMap[(0,0,)]
    falsePos = resultMap[(0,1,)]
    falseNeg = resultMap[(1,0,)]

    #Metrics array [accuracy,errRate, recall,precision, specificity]
    accuracy = float(truePos+trueNeg)/total
    errRate = float(falseNeg+falsePos)/total
    recall  = float(truePos)/( 1.0 if (truePos+falseNeg) == 0 else truePos+falseNeg)     #to prevent division by zero
    precision = float(truePos)/( 1.0 if (truePos+falsePos) == 0 else truePos+falsePos)   #to prevent division by zero
    specificity = float(trueNeg)/( 1.0 if (trueNeg+falsePos) == 0 else trueNeg+falsePos) #to prevent division by zero

    metrics = [accuracy,errRate, recall,precision, specificity]

    return ( metrics)

# Training and testing using Naive Bayes
def trainAndTestNB(train_lbl_vec, test_lbl_vec,lastTime):

    # create LabeledPoints for training
    lblPnt = train_lbl_vec.map(lambda (x,l): LabeledPoint(x,l))

    #print lblPnt.collect()

    # train the model
    model = NaiveBayes.train(lblPnt, 1.0)

    # evaluate training
    resultsTrain = train_lbl_vec.map(lambda lp :  (lp.label, model.predict(lp.features)))

    resultMap = resultsTrain.countByValue()

    # print 'TRAIN '
    trainAccuracy = accuracy(resultMap)

    # test the model
    data = test_lbl_vec.map(lambda (x,l): LabeledPoint(x,l))
    resultsTest = data.map(lambda lp :  (lp.label, model.predict(lp.features)))

    resultMapTest = resultsTest.countByValue()


    #print 'TEST '
    testAccuracy = accuracy(resultMapTest)
    thisTime = time()

    elapsedTime = thisTime - lastTime
    return [elapsedTime,trainAccuracy,testAccuracy]

#Train and test using Decision Tree
def trainAndTestDT(train_lbl_vec, test_lbl_vec, maxDepth, lastTime):

    # create LabeledPoints for training
    lblPnt = train_lbl_vec.map(lambda (x,l): LabeledPoint(x,l))

    # train the model
    #categoricalFeaturesInfo={} # no categorical features
    model = DecisionTree.trainClassifier(lblPnt, numClasses=2, categoricalFeaturesInfo={}, impurity="entropy", maxDepth=maxDepth, maxBins=5)

    # evaluate training
    # use the following approach  due to a bug in mllib
    predictedLabels = model.predict(lblPnt.map(lambda lp : lp.features))
    trueLabels = lblPnt.map(lambda lp : lp.label)
    resultsTrain = trueLabels.zip(predictedLabels)
    resultMap = resultsTrain.countByValue()

    # print 'TRAIN '
    trainAccuracy = accuracy(resultMap)

    # test the model
    data = test_lbl_vec.map(lambda (x,l): LabeledPoint(x,l))
    predictions = model.predict(data.map(lambda x: x.features))
    resultsTest = data.map(lambda lp: lp.label).zip(predictions)
    resultMapTest = resultsTest.countByValue()
    testAccuracy = accuracy(resultMapTest)

    thisTime = time()
    elapsedTime = thisTime - lastTime
    #return [elapsedTime,trainAccuracy,valAccuracy,testAccuracy]
    return [elapsedTime,trainAccuracy,testAccuracy]

#Train and test using Logistic Regression
def trainAndTestLG(train_lbl_vec, test_lbl_vec, regParam, lastTime):

    # create LabeledPoints for training
    lblPnt = train_lbl_vec.map(lambda (x,l): LabeledPoint(x,l))

    # train the model
    #categoricalFeaturesInfo={} # no categorical features
    model = LogisticRegressionWithSGD.train(lblPnt,miniBatchFraction=0.1,regType='l1', intercept=True, regParam=regParam)

    # evaluate training
    resultsTrain = lblPnt.map(lambda lp :  (lp.label, model.predict(lp.features)))

    resultMap = resultsTrain.countByValue()

    # print 'TRAIN '
    trainAccuracy = accuracy(resultMap)

    # test the model
    data = test_lbl_vec.map(lambda (x,l): LabeledPoint(x,l))

    resultsTest = data.map(lambda lp :  (lp.label, model.predict(lp.features)))
    resultMapTest = resultsTest.countByValue()
    testAccuracy = accuracy(resultMapTest)

    thisTime = time()
    elapsedTime = thisTime - lastTime
    return [elapsedTime,trainAccuracy,testAccuracy]

"""
def bottom(rdd,num):

    def bottomIterator(iterator):
            q = []
            size = rdd.count()
            i = 0
            for k in iterator:
                if i <= size & i > num:
                    heapq.heappush(q, k)
                i += 1
            yield q

    def merge(a, b):
            return next(bottomIterator(a + b))

    return rdd.mapPartitions(bottomIterator).reduce(merge)
"""

#hashSize = 10000
hashSize = 10000

# if this is the main program
if __name__ == "__main__":
    # Make sure we have all arguments we need.
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: <classifiers script> <tfidf_filename> <subjects_filename>"
        exit(-1)
    
    # Connect to Spark
    sc = SparkContext(appName="Training Classifiers: Coursework - Large-scale text classification") # job name

    print("\n************* Reading subject pkl files ........")
    #read rdd subjects
    sl = sc.pickleFile(sys.argv[2])
    #sl = sc.parallelize(slist)
    """
    f = open(sys.argv[2])
    pickledList = pickle.load(f)
    sl = sc.parallelize(pickledList)
    f.close()
    """

    print("\n************* Calculating ten more frequent subjects ........")
    #Calculate 10 more frequent subjects
    fsl = sl.map(lambda (fsl): fsl).reduceByKey(add)
    freqSubjects = fsl.flatMap(lambda (f,sl): [(s.strip(),1) for s in sl])
    freqSubjects = freqSubjects.reduceByKey(add)
    print(freqSubjects.collect())
    #Order ascending by value and take 10 first subjects
    moreFreqSubj = freqSubjects.takeOrdered(10,key=lambda t: -t[1])

    topSubjectsList = [s for (s,c) in moreFreqSubj] #convert to a list
    print ("\nTen more frequent Subjects: ",topSubjectsList)

    print("\n************* Reading TF-IDF pkl file ........")
    #read rdd hash vector
    hv = sc.pickleFile(sys.argv[1])

    """
    f = open(sys.argv[1])
    pickledList = pickle.load(f)
    hv = sc.parallelize(pickledList)
    f.close()
    """

    #Join tfidf and subjects rdd
    fswl = hv.join(fsl)
    if fswl.count() == 0:
        print >> sys.stderr, "\nThe task has stopped because the result of join between subject and content files is empty."
        exit(-1)

    print("\n************* Processing classifiers - NB, DT and LG ........")
    #print ("All: ", fswl.collect())
    allResults = [] # for aggregrating all results-NB
    allResultsDT = [] # for aggregrating all results-DT
    allResultsLR = [] # for aggregrating all results-LR
    iterations = 5 #iteratios for logistic regression

    #Create classifier for each subject
    for subject in topSubjectsList:

        # Create binary vector of tuples (1/0, sl):
        #    (1,sl) if sl contain subject
        #    (0,sl) if sl does not contain subject
        label_vec = fswl.map( lambda (f, srwl): subjectLabel(srwl,subject) )
        results = [] # results per NB model

        #OJO take percent of size as sample when it greater than 1000

        #Calculate training and test set sizes
        #Fixed size proportion
        #Calculate train set size
        #take first 75% of data as Training set
        #     remaining as Test set
        vecSize = label_vec.count()
        trainSize = int( vecSize * 0.75)
        testSize = vecSize - trainSize

        #get all rdd objects as list to create train and test sets easier
        #inefficient way - very slow
        sets = label_vec.collect()

        #Train Set
        #trainSet_vec = sc.parallelize(label_vec.take(trainSize))
        trainSet_vec = sc.parallelize(sets[:trainSize])
        print ("\ntraining: ",trainSet_vec.collect())

        #Test Set
        #testSet_vec = bottom(label_vec,trainSize)
        testSet_vec = sc.parallelize(sets[trainSize:])
        print ("testing: ",testSet_vec.collect())

        #Naive Bayes model
        results.append( trainAndTestNB(trainSet_vec, testSet_vec,time() ))#classifier using Naives Bayes

        # using numpy for averaging
        #npres = np.array(results)
        #avg = np.mean(results,axis=0)
        #results.append(avg)

        results.append(subject)
        allResults.append(results)

        #Decision tree
        for maxDepth in [2,4,8] :

          results2 = [] # results per maxDepth

          # do the training and testing
          results2.append( trainAndTestDT(trainSet_vec, testSet_vec, maxDepth, time()))

        # using numpy for averaging
        #resul = [t for (t,train,test) in results2]
        #npres = np.array(resul)
        #avg = np.mean(npres,axis=0)

        results2.append(subject)
        allResultsDT.append(results2)

        #Logistic Regression
        for regParam in [.1,.3,1.,3.,10.] :

           results3 = []

           # do the training and testing
           results3.append( trainAndTestLG(trainSet_vec, testSet_vec,regParam, time() ) )

        # using numpy for averaging
        #resul = [t for (t,train,test) in results3]
        #npres = np.array(resul)
        #avg = np.mean(npres,axis=0)

        results3.append(subject)
        allResultsLR.append(results3)

    print ("\nResults for Naive Bayes Model\n------------------------------------------------------------\n")
    print("|Subject  | Time  | Training Performance Metrics                         | Testing Performance Metrics|\n")
    print("|                  | Accuracy | Error Rate | Recall | Precision | Specificity | Accuracy | Error Rate | \
    Recall | Precision | Specificity |\n")
    for (performance,subject) in allResults:
        row = "| " + subject + " | "
        row += str(performance[0]) + " | " + str(performance[1][0]) + " | " + str(performance[1][1]) + " | " + str(performance[1][2]) + " | "
        row += str(performance[1][3]) + " | " + str(performance[2][0]) + " | " + str(performance[2][1]) + " | " + str(performance[2][2]) + " | "
        row += str(performance[2][3]) + " | "
        print (row)
        print ("--------------------------------------------------------------------------------------")

    print ("\nResults for Decison Tree Model\n----------------------------------------------------------\n")
    print("|Subject  | Time  | Training Performance Metrics                         | Testing Performance Metrics|\n")
    print("|                  | Accuracy | Error Rate | Recall | Precision | Specificity | Accuracy | Error Rate | \
    Recall | Precision | Specificity |\n")
    for (performance,subject) in allResultsDT:
        row = "| " + subject + " | "
        row += str(performance[0]) + " | " + str(performance[1][0]) + " | " + str(performance[1][1]) + " | " + str(performance[1][2]) + " | "
        row += str(performance[1][3]) + " | " + str(performance[2][0]) + " | " + str(performance[2][1]) + " | " + str(performance[2][2]) + " | "
        row += str(performance[2][3]) + " | "
        print (row)
        print ("--------------------------------------------------------------------------------------")

    print ("\nResults for Logistic Regression Model\n------------------------------------------------------------\n")
    print("|Subject  | Time  | Training Performance Metrics                         | Testing Performance Metrics|\n")
    print("|                  | Accuracy | Error Rate | Recall | Precision | Specificity | Accuracy | Error Rate | \
    Recall | Precision | Specificity |\n")
    for (performance,subject) in allResultsLR:
        row = "| " + subject + " | "
        row += str(performance[0]) + " | " + str(performance[1][0]) + " | " + str(performance[1][1]) + " | " + str(performance[1][2]) + " | "
        row += str(performance[1][3]) + " | " + str(performance[2][0]) + " | " + str(performance[2][1]) + " | " + str(performance[2][2]) + " | "
        row += str(performance[2][3]) + " | "
        print (row)
        print ("--------------------------------------------------------------------------------------")


    sc.stop()

