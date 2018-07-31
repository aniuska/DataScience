"""
Coursework2 – A tiny Data Science Project
Aniuska Dominguez Ariosa
INM430_PRD1_A_2014-15 Introduction to Data Science
MSc in Data Science (2014/15)

Second Estimation - period 2005 - 2013
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances

from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import cross_validation
from scipy import stats

#reading data from excel file
xls_file = pd.ExcelFile('Domestic Passenger Yield.xls')
#xls_file = pd.ExcelFile('Traffic&Capacity Delta Airlines.xls')
passengerFare = xls_file.parse('Sheet1',skiprows=[0,1,2,4,12,13,19,20],index_col=0,header=0,skip_footer=4)

passengerFare = passengerFare.convert_objects(convert_numeric=True)

# keep place names and store them in a 
airLines = passengerFare.index

#filling NaN values with zeros to standarise data
passengerFare=passengerFare.fillna(0)
#get colunms name
colnames = np.asarray(passengerFare.columns.values)

#convert dataFrame to matrix to calculate PCA
numpyArray2 = passengerFare.as_matrix()
#remove columns not important for evalution
numpyArray2 = numpyArray2[:,11::]
colNamesFiltered = colnames[11::]


#calculate two principal components
pca = PCA(n_components=3)
pca.fit(numpyArray2)
r_Array = pca.transform(numpyArray2)
#print(r_Array)
print ('explained variance (first %d components): %f'%(3, sum(pca.explained_variance_ratio_)))
#print('covariance: ',pca.get_covariance())

# plotting results
plt.figure(1)
plt.title('PCA of Domestic Passenger Yield x PC_1_2')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
#plt.legend()
plt.scatter(r_Array[:,0], r_Array[:,1] , c = "#D06B36", s = 50, alpha = 0.4, linewidth='0')
    
# plotting results
plt.figure(2)    
plt.title('PCA of Domestic Passenger Yield x PC_2_3')
plt.xlabel('PC_2')
plt.ylabel('PC_3')
#plt.legend()
plt.scatter(r_Array[:,1], r_Array[:,2] , c = "#E5CC00", s = 50, alpha = 0.4, linewidth='0')

# plotting results
plt.figure(3)
plt.title('PCA of Domestic Passenger Yield x PC_1_3')
plt.xlabel('PC_1')
plt.ylabel('PC_3')
#plt.legend()
plt.scatter(r_Array[:,0], r_Array[:,2] , c = "#A7CC00", s = 50, alpha = 0.4, linewidth='0')
plt.show()

#print( pca.components_)

print ("--- Firstly, the first component: ")
comp1Loadings = np.asarray(pca.components_[0])[np.argsort( np.abs(pca.components_[0]))[::-1]][0:10]
comp1Names = np.asarray(colNamesFiltered)[np.argsort( np.abs(pca.components_[0]))[::-1]][0:10]

for i in range(0, 9):
    print ( "Column \"" , comp1Names[i] , "\" has a loading of: ", comp1Loadings[i])

print ("\n --- Secondly, the second component: ")
comp2Loadings = np.asarray(pca.components_[1])[np.argsort( np.abs(pca.components_[1]))[::-1]][0:10]
comp2Names = np.asarray(colNamesFiltered)[np.argsort( np.abs(pca.components_[1]))[::-1]][0:10]

for i in range(0, 9):
    print ( "Column \"" , comp2Names[i] , "\" has a loading of: ", comp2Loadings[i])

print ("\n --- Thirdly, the third component: ")
comp2Loadings = np.asarray(pca.components_[2])[np.argsort( np.abs(pca.components_[2]))[::-1]][0:10]
comp2Names = np.asarray(colNamesFiltered)[np.argsort( np.abs(pca.components_[2]))[::-1]][0:10]

for i in range(0, 9):
    print ( "Column \"" , comp2Names[i] , "\" has a loading of: ", comp2Loadings[i])

#box plot
f,(ax1,ax2) = plt.subplots(1,2,sharey=True)

ax1.hist(r_Array[:,0],5)
ax1.set_title("Uniform Histogram - PC1")
ax1.set_xlabel("PC1")
ax1.set_ylabel("Frequency")

ax2.hist(numpyArray2[:,8],5)
ax2.set_title("Uniform Histogram - 2013")
ax2.set_xlabel("2013")
ax2.set_ylabel("Frequency")
#ax1.show() 

f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
ax1.hist(r_Array[:,1],5)
ax1.set_title("Uniform Histogram - PC2")
ax1.set_xlabel("PC2")
ax1.set_ylabel("Frequency")

ax2.hist(r_Array[:,2],5)
ax2.set_title("Uniform Histogram - PC3")
ax2.set_xlabel("PC3")
ax2.set_ylabel("Frequency")

f,(ax1,ax2,ax3) = plt.subplots(1,3,sharex=True,sharey=True)
ax1.boxplot(r_Array[:,0],False,showmeans=True,meanline=True)
ax1.set_title("Box Plot - PC1")

ax2.boxplot(r_Array[:,1],False,showmeans=True,meanline=True)
ax2.set_title("Box Plot - PC2")

ax3.boxplot(r_Array[:,2],False,showmeans=True,meanline=True)
ax3.set_title("Box Plot - PC3")
#ax2.show()  
plt.show()

print("\n************** Clustering ***********************\n")
#Clustering
# get the known cluster labels into a separate array
classLabelsKnown = passengerFare.index

# get all the data
DataToCluster = numpyArray2

#computing K-Means with K = 3 (3 clusters)
kmeansModel = KMeans(init='random', n_clusters=3, n_init=10)
kmeansModel.fit_predict(DataToCluster)
clusterResults = kmeansModel.labels_

## check the results and try to compare with known labels
for i, clustLabel in enumerate(clusterResults):
    print("Cluster result: ", clustLabel, " Known labels: ",classLabelsKnown[i])

    
pca = PCA(n_components=3)
#Fit a PCA model to the data
pca.fit(DataToCluster)

# have a look at the components directly if we can notice any interesting 
#structure
projectedAxes = pca.transform(DataToCluster)

dataColumnsToVisualize = projectedAxes
#dataColumnsToVisualize = data
IDsForvisualization = clusterResults
 # to color the points according to the 
#results of k-means

columnIDToVisX = 0 # some variable to keep coind simple and flexible
columnIDToVisY = 1

plt.figure(1)
plt.suptitle('Results of the algorithm visualised over PCs')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
# some plotting using numpy's logical indexing
plt.scatter(dataColumnsToVisualize[IDsForvisualization==0,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==0,columnIDToVisY], c = "#66c2a5", s = 50, alpha = 0.7, linewidth='0') # greenish
plt.scatter(dataColumnsToVisualize[IDsForvisualization==1,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==1,columnIDToVisY], c = "#fc8d62", s = 50, alpha = 0.7, linewidth='0') # orangish
plt.scatter(dataColumnsToVisualize[IDsForvisualization==2,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==2,columnIDToVisY], c = "#8da0cb", s = 50, alpha = 0.7, linewidth='0') # blueish

IDsForvisualization = classLabelsKnown # to color the points according 
#to the known labels 

plt.figure(2)
plt.suptitle('Known labels visualised over PCs')
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.scatter(dataColumnsToVisualize[IDsForvisualization==0,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==0,columnIDToVisY], c = "#66c2a5", s = 50, alpha = 0.7, linewidth='0')
plt.scatter(dataColumnsToVisualize[IDsForvisualization==1,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==1,columnIDToVisY], c = "#fc8d62", s = 50, alpha = 0.7, linewidth='0')
plt.scatter(dataColumnsToVisualize[IDsForvisualization==2,columnIDToVisX], dataColumnsToVisualize[IDsForvisualization==2,columnIDToVisY], c = "#8da0cb", s = 50, alpha = 0.7, linewidth='0')

plt.show()

#projectedAxes
#colNamesFiltered

#show loadings for clusters
print ("--- Firstly, the first component: ")
comp1Loadings = np.asarray(pca.components_[0])[np.argsort( np.abs(pca.components_[0]))[::-1]][0:2]
comp1Names = np.asarray(colNamesFiltered)[np.argsort( np.abs(pca.components_[0]))[::-1]][0:2]

for i in range(0, 2):
    print ( "Column \"" , comp1Names[i] , "\" has a loading of: ", comp1Loadings[i])

print ("\n --- Secondly, the second component: ")
comp2Loadings = np.asarray(pca.components_[1])[np.argsort( np.abs(pca.components_[1]))[::-1]][0:2]
comp2Names = np.asarray(colNamesFiltered)[np.argsort( np.abs(pca.components_[1]))[::-1]][0:2]

for i in range(0, 2):
    print ( "Column \"" , comp2Names[i] , "\" has a loading of: ", comp2Loadings[i])



#Cross-Validation
print('\n********************** Cross Validation ************************\n')

arrVar1 = numpyArray2[:,8]  #to PC1
arrVar2 = numpyArray2[:,1] #to PC2
arrVar3 = numpyArray2[:,6]  #to PC3

numberOfSamples = len(arrVar1)

# generate sampling indices for n points and set the k to be 5, 
#5-fold cross validation  
kf = cross_validation.KFold(numberOfSamples, n_folds=5)
foldCount = 0
for train_index, test_index in kf:
    print("-------- We are having the run: ", foldCount )
    arraySubset1 = arrVar1[train_index]
    arraySubset2 = arrVar2[train_index]
    arraySubset3 = arrVar3[train_index]
    
    #models var1 vs var2, var1 vs var3, vaer2 vs var3
    slope, intercept, r_value, p_value, std_err = stats.linregress(arraySubset1, arraySubset2)
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(arraySubset1, arraySubset3)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(arraySubset2, arraySubset3)
    
    print ("\nvar1 vs var2-Slope: ", slope, "Intercept: ", intercept, "r_value: ", r_value, "p_value: ", p_value,"std_err: ", std_err)
    print ("\nvar1 vs var3-Slope: ", slope1, "Intercept: ", intercept1, "r_value: ", r_value1, "p_value: ", p_value1,"std_err: ", std_err1)
    print ("\nvar2 vs var3-Slope: ", slope2, "Intercept: ", intercept2, "r_value: ", r_value2, "p_value: ", p_value2,"std_err: ", std_err2)
    
    xp = np.linspace(arrVar1.min(), arrVar1.max(), 100)
    evaluatedLine = np.polyval([slope, intercept], xp)
        
    xp1 = np.linspace(arrVar1.min(), arrVar1.max(), 100)
    evaluatedLine1 = np.polyval([slope1, intercept1], xp1)
    
    xp2 = np.linspace(arrVar2.min(), arrVar2.max(), 100)
    evaluatedLine2 = np.polyval([slope2, intercept2], xp2)
    
    f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)    
    ax1.plot(xp, evaluatedLine, 'k--', linewidth = 1, alpha = 0.3)
    title = str(colNamesFiltered[8]) + " vs " + str(colNamesFiltered[1])
    ax1.set_title(title)    
    ax2.plot(xp1, evaluatedLine1, 'k--', linewidth = 1, alpha = 0.3)
    title = str(colNamesFiltered[1]) + " vs " + str(colNamesFiltered[6])
    ax2.set_title(title) 
    ax3.plot(xp2, evaluatedLine2, 'k--', linewidth = 1, alpha = 0.3)
    title = str(colNamesFiltered[8]) + " vs " + str(colNamesFiltered[6])
    ax3.set_title(title) 
    plt.show()
    foldCount += 1 

print('\n****************************** Model Estimation *******************\n')    
#test whether the regression model can estimate unseen points.
print("***************** PC1 and PC2 *********************")
foldCount = 0
for train_index, test_index in kf:
    print("\n-------- We are having the run: ", foldCount )
    
    arraySubset1 = arrVar1[train_index]
    arraySubset2 = arrVar2[train_index]
    
    unseenSubset1 = arrVar1[test_index]
    unseenSubset2 = arrVar2[test_index]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(arraySubset1, arraySubset2)
    
    # Use the regression models to estimate the unseen values 
    estimatedValues = slope * unseenSubset1 + intercept
    
    # check the differences between the estimates and the real values    
    differences = unseenSubset2 - estimatedValues
    print ("\nEstamated values",unseenSubset2)
    print ("\nDifference",np.average(differences))
    foldCount += 1 

foldCount = 0
print("***************** PC1 and PC3 *********************")
for train_index, test_index in kf:
    print("\n-------- We are having the run: ", foldCount )
    
    arraySubset1 = arrVar1[train_index]
    arraySubset2 = arrVar3[train_index]
    
    unseenSubset1 = arrVar1[test_index]
    unseenSubset2 = arrVar3[test_index]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(arraySubset1, arraySubset2)
    
    # Use the regression models to estimate the unseen values 
    estimatedValues = slope * unseenSubset1 + intercept
    
    # check the differences between the estimates and the real values    
    differences = unseenSubset2 - estimatedValues
    print ("\nEstamated values",unseenSubset2)
    print ("\nDifference",np.average(differences))
    foldCount += 1 


