"""
Coursework2 – A tiny Data Science Project
Aniuska Dominguez Ariosa
INM430_PRD1_A_2014-15 Introduction to Data Science
MSc in Data Science (2014/15)

First Estimation period 1995 - 2013
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
from scipy.cluster.vq import kmeans,vq
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
#passengerFare=passengerFare.fillna(0)
#get colunms name
colnames = np.asarray(passengerFare.columns.values)

for names in colnames:
    passengerFare[names]=passengerFare[names].fillna(passengerFare[names].mean())
    print ("\n Mean and Std for %s: %f, %f" % (names,passengerFare[names].mean(),passengerFare[names].std()))

#convert dataFrame to matrix to calculate PCA
numpyArray2 = passengerFare.as_matrix()
#remove an empty columna
numpyArray2 = numpyArray2[:,1::]
colNamesFiltered = colnames[1::]

#calculate two principal components
pca = PCA(n_components=3)
pca.fit(numpyArray2)
r_Array = pca.transform(numpyArray2)
#print(r_Array)
print ('\nexplained variance (first %d components): %f'%(3, sum(pca.explained_variance_ratio_)))
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

for i in range(0, 10):
    print ( "Column \"" , comp1Names[i] , "\" has a loading of: ", comp1Loadings[i])

print ("\n --- Secondly, the second component: ")
comp2Loadings = np.asarray(pca.components_[1])[np.argsort( np.abs(pca.components_[1]))[::-1]][0:10]
comp2Names = np.asarray(colNamesFiltered)[np.argsort( np.abs(pca.components_[1]))[::-1]][0:10]

for i in range(0, 10):
    print ( "Column \"" , comp2Names[i] , "\" has a loading of: ", comp2Loadings[i])

print ("\n --- Thirdly, the third component: ")
comp2Loadings = np.asarray(pca.components_[2])[np.argsort( np.abs(pca.components_[2]))[::-1]][0:10]
comp2Names = np.asarray(colNamesFiltered)[np.argsort( np.abs(pca.components_[2]))[::-1]][0:10]

for i in range(0, 10):
    print ( "Column \"" , comp2Names[i] , "\" has a loading of: ", comp2Loadings[i])

#box plot
f,(ax1,ax2) = plt.subplots(1,2,sharey=True)

ax1.hist(r_Array[:,0],5)
ax1.set_title("Uniform Histogram - PC1")
ax1.set_xlabel("PC1")
ax1.set_ylabel("Frequency")

ax2.hist(numpyArray2[:,3],5)
ax2.set_title("Uniform Histogram - 1998")
ax2.set_xlabel("1998")
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

f,(ax1,ax2,ax3) = plt.subplots(1,3,sharex=True,sharey=True)
ax1.boxplot(numpyArray2[:,1],False,showmeans=True,meanline=True)
ax1.set_title("Box Plot " + str(colNamesFiltered[0]))

ax2.boxplot(numpyArray2[:,4],False,showmeans=True,meanline=True)
ax2.set_title("Box Plot - " + str(colNamesFiltered[3]))

ax3.boxplot(numpyArray2[:,18],False,showmeans=True,meanline=True)
ax3.set_title("Box Plot " + str(colNamesFiltered[18]))
#ax2.show()  
plt.show()

