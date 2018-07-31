
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy import stats
import seaborn as sns
sns.set()
 

#shapefiles = ["england_lad.shp","London_Boroughs.shp","London_Ward.shp","London_LSOA.shp"]
#unempfiles = ["Census_Per_LA.csv","Census_Per_London.csv","Census_Per_Ward_London.csv","Census_Per_LSOA_London.csv"]
crimefiles= ["Crime_Rate_LA.csv","Crime_Rate_London.csv","Crime_Rate_wards.csv","Crime_Rate_lsoa.csv"]

crimefiles= ["Crime_Rate_LA.csv"]
unempfiles = crimefiles
#shapefiles = ["england_lad.shp"]
#unempfiles = ["Census_Per_LA.csv"]

# Open a directory
path = "data"
dirs = os.listdir( path )

#For Unemployment
#rng = range(48)
#pop_ix = 3
#den_ix = 9
#y_ix = 47

#For Crime
rng = range(66)
pop_ix = 4
den_ix = 12
y_ix = 5

#output= open("correlation\correlation_output.txt",'w+')
output= open("correlation\Crime_correlation_outputLA.txt",'w+')
print ("--------------------------------------", file=output)
#print ("--------- UNEMPLOYMENT ---------------", file=output)
print ("--------- All Variables ---------------", file=output)
print ("--------------------------------------", file=output)

#file = "Crime_Rate_LA.csv"
# loop all the files
for file in unempfiles:
    census=pd.read_csv(path + "\\"+ file, usecols =  rng)
    filename = os.path.splitext(file)[0]
    
    print ("\n--------------------------------------", file=output)
    print ("============== Correlation coefficients for  ", filename, file=output)
        
    geo_pop = census.iloc[::,:pop_ix]
    density_unit = census.iloc[::,den_ix]
    Y = census.iloc[::,y_ix]
    #X = census.iloc[::,pop_ix:y_ix] #unemployment
    X = census.iloc[::,y_ix:66] #crime
    #X.drop(X.columns[[0, 1, 6,]], axis=1, inplace=True)
    
    #X.drop(X.columns[6], axis=1, inplace=True) #unemployment
    X.drop(X.columns[7], axis=1, inplace=True)
    #var_names = list(X.columns.values)
    var_names = X.columns.values.tolist()
         
    #calculating correlations between pairs.
    corr_df = X.corr(method='pearson')
         
    print("--------------- CREATE A HEATMAP ---------------")
    # Create a mask to display only the lower triangle of the matrix
    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask)] = True
    
    # Create the heatmap using seaborn library. 
    # List if colormaps (parameter 'cmap') is available here: http://matplotlib.org/examples/color/colormaps_reference.html
    fig, ax = plt.subplots(figsize=(29, 29))
    sns.heatmap(corr_df, ax=ax, cmap='RdYlGn_r', 
                annot=True,  
                mask = mask, linewidths=3.5) #vmax=1.0, vmin=-1.0 ,
    sns.heatmap(corr_df, mask=corr_df < 1,cbar=False,
                annot=True, annot_kws={"weight": "bold"})
    figName = "correlation\\" + filename + "_correlation.png" 
    
    # Show the plot we reorient the labels for each column and row to make them 
    #easier to read.
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90)
    plt.savefig(figName, pdi=300)
    plt.show()
    
    ####### Select those variables that Pearson's coefficient >= 0.70
    #Absolute value: magnitude of correlation without taking into account direction
    #Get only lower triangle of the array (np.tril)
    df = pd.DataFrame(np.tril(corr_df, -1), columns=corr_df.columns, index=corr_df.index).abs()
    #Get dataframe with row and col labels
    df = df[df >= 0.70].stack().reset_index()
    print("\nVariables with Pearson coefficient greater (and equal) than 0.70\n", file=output)
    #print(df.sort_values(by=0,ascending=False), file=output)
    print(df.to_string(), file=output)
    
    ############## Statistical Summary ################################
    print("\nStatistical Summary\n", file=output)
    #print(X.describe().transpose().to_string())
    print(X.describe().transpose().to_string(), file=output)
    
    #print(df)
    #l=df.level_0[df.level_1 == df.level_1.value_counts().idxmax()]
    
    #join cols of df, remove duplicated values and convert the result to a list 
    #get a list only distinct values
    #from functools import reduce
    #ind = reduce(np.union1d, (df.level_0,df.level_1)).tolist()
    
    ###### scatter matrix plots and histograms for Pearson's corr coef >= 0.7
    #List unique values in the df['level_0'] column
    level0_unique = df.level_0.unique()
    X_norm = X
    X_norm = (X_norm - X_norm.mean())/X_norm.std(ddof=0)
    
    #level0_unique = level0_unique[level0_unique > 'k059'] 
    for item in level0_unique:
      fig, ax = plt.subplots()   
      ls = df[df['level_0']== item]['level_1'].values.tolist()
      ls.insert(0, item)   
      figName = "correlation\\" + filename + "_scatter_"+item+".png" 
      sns_plot= sns.pairplot(X_norm[ls], size=1.5, aspect=1, kind="reg",diag_kind="kde")
      sns_plot.savefig(figName, pdi=300)
      #plt.cla()
      plt.close(fig) 
      
output.close()
plt.close("all") 