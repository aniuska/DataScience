"""
Individual Project
Effects 

MSc. Data Science
2016

Aniuska I. Dominguez Ariosa

Lasso script
"""
from __future__ import print_function
print(__doc__)

import os
import numpy as np
import pandas as pd
from cycler import cycler
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns

np.seterr(invalid='ignore')

from scipy import stats

from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path

from sklearn.linear_model import LassoCV

from sklearn import linear_model
import statsmodels.formula.api as sm
import pysal

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
import statsmodels.stats.outliers_influence as oi
import statsmodels.tools.tools as tls
from sklearn.metrics import r2_score

from pysal.contrib.viz import mapping as maps

import statsmodels.api as sm2

####OJO
#OJO-another run for alphas = 10**np.linspace(10,-2,100)*0.5"

shapefiles = ["england_lad.shp","London_Boroughs.shp","London_Ward.shp","London_LSOA.shp"]
unempfiles = ["Census_Per_LA.csv","Census_Per_London.csv","Census_Per_Ward_London.csv","Census_Per_LSOA_London.csv"]
#crimefiles= []

#shapefiles = ["england_lad.shp"]
#unempfiles = ["Census_Per_LA.csv"]

# Open a directory
path_shape = "shapefiles"
dirs = os.listdir( path_shape )
path_data = "data"

#Read Census Data
rng = range(48)
pop_ix = 3
den_ix = 9
y_ix = 47

output= open("lasso\Lasso_output.txt",'w+')
print ("--------------------------------------", file=output)
print ("--------- UNEMPLOYMENT ---------------", file=output)
print ("--------------------------------------", file=output)  

df = pd.DataFrame() #empty dataframe
  
# loop all the files
for file in unempfiles:
    census=pd.read_csv(path_data + "\\"+ file, usecols =  rng)
    filename = os.path.splitext(file)[0]
    
    print ("--------------------------------------", file=output)
    print ("============== Lasso for ", filename,file=output)

    geo_pop = census.iloc[::,:pop_ix]
    density_unit = census.iloc[::,den_ix]
    Y = census.iloc[::,y_ix]
    X = census.iloc[::,pop_ix:y_ix]
    area_code = census.iloc[::,0]
    #X = census.iloc[::,:y_ix]
    #X.drop(X.columns[[1,2, 6,]], axis=1, inplace=True)
    X.drop(X.columns[6], axis=1, inplace=True)
   
    var_names = X.columns.values.tolist()
    
    #By convention: 
    ## X is standardised (mean 0, unit variance)
    ## Y is centered
    #Standardising data: mean zero
    X_norm = stats.zscore(X)
    Y_norm = Y - np.mean(Y)
    
    #Dataset partition: training and test sets - 80/20
    X_train, X_test, y_train, y_test = train_test_split(X_norm, Y_norm, test_size=0.2, random_state=42)
    
    #Regularisation: Lasso regression
    
    #1) Finding best penalisation parameter alpha
    #alphas = np.logspace(-4, -.5, 30)
    #alphas = np.logspace(0.8, -4, 60) #1st run
    alphas = np.logspace(1.6, -2, 60) #2nd run
    #alphas = 10**np.linspace(10,-2,100)*0.5 #3rd run
    lasso_cv = linear_model.LassoCV(alphas=alphas,cv=10,fit_intercept=False,normalize=False)
    model=lasso_cv.fit(X_train, y_train)
    alpha = lasso_cv.alpha_
        
    #2) Run Lasso for best alpha - training set
    lassoreg = Lasso(alpha=alpha,normalize=False, max_iter=1e5, 
                     fit_intercept=False)
    model=lassoreg.fit(X_train, y_train)
    #intrc= lassoreg.intercept_
    lasso_predict = lassoreg.predict(X_test)
    lasso_residual = lasso_predict - y_test
    coefs= lassoreg.coef_
    sparse_coefs = lassoreg.sparse_coef_
    n_iter = lassoreg.n_iter_
    
    #Print coefficients different of zero
    #index = [i for i in range(len(var_names)) if (coefs[i] != 0.0) ]
    
    index = np.flatnonzero(coefs)
    
    print (' ',file=output)
    print("======= Lasso regression for training set =========", file=output)
    print("alpha: ",alpha,file=output)
    print(pd.DataFrame(coefs[index], np.array(var_names)[index]), file=output)
    print("Number of iterations: ",n_iter, file=output)
    # Explained variance score: 1 is perfect prediction
    print('Variance score (test set): %.2f' % lassoreg.score(X_test, y_test), file=output)
    # The mean squared error
    residuals_test = lassoreg.predict(X_test) - y_test
    print("Mean squared error (test set): %.2f"
          % np.mean((residuals_test) ** 2), file=output)
    
    r2_score_lasso = r2_score(y_test, lasso_predict)
    print("r^2 on test data : %f" % r2_score_lasso, file=output)
      
    # test that the feature score of the best features
    #f_test, _ = f_regression(X, y)
    f_values, p_values = f_regression(X_train, y_train)
    #f_test /= np.max(f_test)
    print("F-values & p-values (training set):",file=output)
    print(pd.concat( [pd.DataFrame(np.around(f_values, decimals=3),var_names,columns=['Ftest']), 
               pd.DataFrame(np.around(p_values, decimals=3),var_names,columns=['pValue'])],
               axis=1
               ).sort_values(by='pValue',ascending=False),
          file=output
         )
    print (" ")
    #print("F-values:",f_values, file=output)
    #print("p-values:",p_values, file=output)
    
    #3) Compute Lasso path
    NUM_COLORS = len(var_names)
    cm = plt.get_cmap('gist_rainbow')
    
    eps = 5e-3  # the smaller it is the longer is the path
    
    print("Computing regularization path using the lasso...", file=output)
    alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps, 
                                              alphas=alphas,fit_intercept=False)
    
    #print(coefs_lasso)
    #print(pd.DataFrame(coefs_lasso, var_name_List[5:23]))
    figName = "lasso\\" + filename + "_LassoPathTrainingSet.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = plt.gca()
    
    colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    markers=['.','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_','8',',',1,2,3,4,6,7,0,'.','1','o','v','>','*','x','+','s','^','8',5,',']
    ax.set_prop_cycle(cycler('color', colors) + cycler('marker', markers))
    
    l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)
    
    plt.xlabel(r'-log($\lambda$)')
    plt.ylabel('coefficients')
    plt.title('Lasso Coefficient Paths (train set)')
    plt.legend((l1), (var_names),loc='best',ncol=4)
    #plt.axis('tight')
    plt.vlines(-np.log10(alpha), np.min(coefs_lasso), np.max(coefs_lasso), linestyle='dashdot')

    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.suptitle(filename)
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    
    #plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
    #plt.show()
    
    print (' ',file=output)
    print("======= Lasso regression for all dateset =========", file=output)
    #2.2) Run Lasso for best alpha on all dataset- final model
    
    #alphas = np.logspace(-4, -.5, 30)
    #alphas = np.logspace(0.8, -4, 60)
    lasso_cv = linear_model.LassoCV(alphas=alphas,cv=10,
                                    fit_intercept=False,
                                    normalize=False)
    model=lasso_cv.fit(X_norm, Y_norm)
    alpha_all = lasso_cv.alpha_
    mse_all = lasso_cv.mse_path_
        
    #2) Run Lasso for best alpha - training set
    lassoreg = Lasso(alpha=alpha_all,normalize=False, max_iter=1e5, 
                     fit_intercept=False)
    model=lassoreg.fit(X_norm, Y_norm)
    #intrc= lassoreg.intercept_
    lasso_predict_all = lassoreg.predict(X_norm)
    lasso_residual_all = lasso_predict_all - Y_norm
    coefs_all= lassoreg.coef_
    sparse_coefs_all = lassoreg.sparse_coef_
    n_iter_all = lassoreg.n_iter_
    
    #Print Lasso Results
    print (' ',file=output)
    print("======= Lasso regression for all dataset =========", file=output)
    
    #index_all = [i for i in range(len(var_names)) if (coefs[i] != 0.0) ]
    
    index_all = np.flatnonzero(coefs_all)
    
    #print(pd.DataFrame(coefs[index_all], var_names[index_all]), file=output)
    print("Coefficients (all dataset):\n",pd.DataFrame(np.around(coefs_all[index_all],decimals=3),np.array(var_names)[index_all]).sort_values(by=0,ascending=False), file=output)
    #("Coefficients (all dataset):\n",pd.DataFrame(coefs_all[index_all],np.array(var_names)[index_all]).sort_values(by=0,ascending=False), file=output)
    
    print("Optimal regularisation parameter: %.3f", alpha_all,"log= %.3f", -np.log10(alpha_all), file=output)
    
    print("Number of iterations: ",n_iter_all, file=output)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % lassoreg.score(X_norm, Y_norm), file=output)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((lassoreg.predict(X_norm) - Y_norm) ** 2), file=output)
    
    residuals_all = lassoreg.predict(X_norm) - Y_norm
    #print("Lasso Residual -all ", residuals_all,file=output)
    
    r2_score_all = r2_score(Y_norm, lasso_predict_all)
    print("r^2 on all data : %f" % r2_score_all, file=output)
          
    # test that the feature score of the best features
    #f_test, _ = f_regression(X, y)
    f_values, p_values = f_regression(X_norm, Y_norm)
    #f_test /= np.max(f_test)
    print("F-values & p-values (training set):",file=output)
    print(pd.concat( [pd.DataFrame(np.around(f_values, decimals=3),var_names,columns=['Ftest']), 
               pd.DataFrame(np.around(p_values, decimals=3),var_names,columns=['pValue'])],
               axis=1
               ).sort_values(by=['pValue','Ftest'],ascending=[False,True]),
          file=output
         )
    print (" ")
    
    #print("F-values:",f_values, file=output)
    #print("p-values:",p_values, file=output)
        
    #Save var index
    d = pd.DataFrame()
    d['index'] = index_all
    d['varName'] = [var_names[i] for i in index_all ]
    d['Filename'] = filename
    df = df.append(d)
    
    #3) Compute Lasso path    
    eps = 5e-3  # the smaller it is the longer is the path
    
    print("Computing regularization path using the lasso...", file=output)
    alphas_lasso, coefs_lasso, _ = lasso_path(X_norm, Y_norm, eps, 
                                              alphas=alphas,
                                              fit_intercept=False)
    
    figName = "lasso\\" + filename + "_LassoPathAllDataset.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = plt.gca()
    
    ax.set_prop_cycle(cycler('color', colors) + cycler('marker', markers))
    
    l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)
    
    plt.xlabel(r'-log($\lambda$)')
    plt.ylabel('coefficients')
    plt.title('Lasso Coefficients Path (all dataset)')
    plt.legend((l1), (var_names),loc='best',ncol=4)
    #plt.axis('tight')
    plt.vlines(-np.log10(alpha_all), np.min(coefs_lasso), np.max(coefs_lasso), linestyle='dashdot')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.suptitle(filename)
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    
    #Plot avaerage MSE across folders
    figName = "lasso\\" + filename + "_LassoMSEAllDataset.png"
    #plt.figure(2)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = plt.gca()
    
    plt.plot(-np.log10(alphas_lasso), mse_all,linewidth=0.5)
    l1, = plt.plot(-np.log10(alphas_lasso), np.sqrt(mse_all).mean(axis=1), 'k',
         label='Avg. across folds', linewidth=3,linestyle='-')
    l2=plt.axvline(-np.log10(alpha_all), linestyle='--', color='k',label='alpha CV')
                      
    plt.xlabel(r'-log($\lambda$)')
    plt.ylabel('Mean squared error (MSE)')
    plt.title('Lasso Regression $\lambda$ vs. MSE on each fold')
    #plt.legend((l1),(var_names),loc='best',ncol=4)
    #plt.axis('tight')
    #ax.set_xscale('log')
    plt.legend(handles=[l1], loc=2,fontsize=16)
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis 
    plt.suptitle(filename)
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    
    #=====OLS models==============================
    # Fit OSL regression model for training set (no intercept)
    X_lasso_test = X_test[::,index]
    #mod = sm.OLS( pd.DataFrame(y_train), pd.DataFrame(X_train, columns=var_names),hasconst=False)
    mod = sm.OLS( y_train, X_train,hasconst=False)
    results = mod.fit_regularized(alpha=alpha, L1_wt=1.0,maxiter=100000)     
    
    residuals = results.resid
    #Residual Plots  
    # predicte value vs. residual plot
    plt.figure(3,figsize=(8,10))
    ax1 = plt.gca()
    ax1 = plt.subplot(211)
    
    pred_y = mod.predict(results.params,X_test)
    ax1.plot(pred_y,stats.zscore(y_test - pred_y),'o')
    xmin = min(pred_y) - 1.0
    xmax=max(pred_y) + 1.0
    plt.hlines(y=0,xmin=xmin,xmax=xmax)
    plt.xlabel('Predicted value')
    plt.ylabel('Standardised Residuals')    
    plt.title('Residual plot (ols test)')
    plt.xlim(xmin,xmax)
    plt.suptitle(filename)
    
    
    #Plot Residuals for test set
    ax2 = plt.subplot(212)
    
    ax2.plot(lasso_predict,stats.zscore(lasso_residual),'o')
    xmin = min(lasso_predict) - 1.0
    xmax=max(lasso_predict) + 1.0
    plt.hlines(y=0,xmin=xmin,xmax=xmax)
    plt.xlabel('Predicted value')
    plt.ylabel('Standardised Residuals')
    plt.title('Residual plot (lasso test)')
    plt.xlim(xmin,xmax)
    plt.suptitle(filename)
    figName = "lasso\\" + filename + "_LassoResidualsTest.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    
    
    # Fit OSL regression model for entire dataset
    mod2 = sm.OLS(Y_norm, X_norm,hasconst=False)
    results2 = mod2.fit_regularized(alpha=alpha_all, L1_wt=0.0)
    
    residuals2 = results2.resid
    
    #Residuals Plot  
    plt.figure(5)
    ax = plt.gca()
    
    plt.plot(results2.fittedvalues,results2.resid_pearson,'o')
    xmin = min(results2.fittedvalues) - 1.0
    xmax=max(results2.fittedvalues) + 1.0
    plt.hlines(y=0,xmin=xmin,xmax=xmax)
    plt.xlabel('Predicted value')
    plt.ylabel('Standardised Residuals')
    plt.title('Residual plot (all dataset)')
    plt.suptitle(filename)
    plt.xlim(xmin,xmax)
    figName = "lasso\\" + filename + "_LassoResidualsAll-ols.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
         
    #checking normallity of residuals
    #hist of residuals.
    plt.figure(6)
    plt.hist(results2.resid_pearson,histtype='stepfilled')#histtype='stepfilled'
    plt.title("Standardised Residuals Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.suptitle(filename)
    figName = "lasso\\" + filename + "_LassoResidualsHistAll-ols.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    
    #Quantile-Quantile plot: 
    fig2 = plt.figure(7)
    fig2 = sm2.qqplot(residuals2, dist=stats.norm, line='45', fit=True)
    plt.title("Normal Q-Q plot")
    plt.suptitle(filename)
    figName = "lasso\\" + filename + "_LassoResiduals_QQplotAll.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    
    """
    #hist of residual to check normality.
    plt.figure(7)
    plt.hist(stats.zscore(lasso_residual_all),histtype='stepfilled')#histtype='stepfilled'
    plt.title("Standardised Residuals Histogram ")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.suptitle(filename)
    figName = "lasso\\" + filename + "_LassoResidualsHistAll-lasso.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    
    #Plot Residuals 
    plt.figure(8)
    ax = plt.gca()
    
    plt.plot(Y,stats.zscore(lasso_residual_all),'o')
    plt.hlines(y=0,xmin=-2,xmax=6)
    plt.xlabel('Response')
    plt.ylabel('Residuals')
    plt.title('Residual plot (all dataset)')
    plt.suptitle(filename)
    figName = "lasso\\" + filename + "_LassoResidualsAll-lasso.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    """
    
    # Inspect the results               
    print (' ',file=output)
    print ("1)OSL result training set (fit_reglarised) for ", filename,file=output)
    print( results.summary(xname=var_names),file=output)
    
    print (' ',file=output)
    print ("1.1)OSL result training set (fit) for ", filename,file=output)
    X_lasso=X_train[::,index]
    print( sm.OLS(y_train, X_lasso,hasconst=False).fit().summary(xname=[var_names[i] for i in index]),file=output)
    
    print (' ',file=output)
    print ("2)OSL result all dataset (fit_reglarised) for ", filename,file=output)
    print( results2.summary(xname=var_names),file=output)
    
    print (' ',file=output)
    print ("2)OSL result all dataset (fit) for ", filename,file=output)
    X_lasso=X_norm[::,index_all]
    print( sm.OLS(Y, X_lasso,hasconst=False).fit().summary(xname=[var_names[i] for i in index_all]),file=output)
    print (' ',file=output)
    
    #5) Multicollinearity Diagnostic (VIF) = condition number
    #vifResults = oi.variance_inflation_factor(vifVar.as_matrix(), i)
    # For each Xi, calculate VIF
    if (X_lasso.shape[1] > 1):
        vif = [oi.variance_inflation_factor(X_lasso, i) for i in range(X_lasso.shape[1])]
        print('',file=output)
        print('============== VIF (', filename,') ==========',file=output)
        lasso_names = [var_names[i] for i in index_all ]
        print(pd.DataFrame(np.around(vif), lasso_names).sort_values(by=0),file=output)
        print (' ',file=output)
                 
    #6) Spatial autocorrelation Diagnostic    
    #Read shape file
    shp_link = path_shape + "\\"+ shapefiles[unempfiles.index(file)]    
    map_shape = pysal.open(path_shape + "\\"+ shapefiles[unempfiles.index(file)])
                
    #calculate weigths from a shapefile
    # A rook weights matrix defines a location's neighbors as those areas with 
    #shared borders (in contrast to a queen weights matrix, which only need to 
    #share a single vertex).
    w = pysal.rook_from_shapefile(path_shape + "\\"+ shapefiles[unempfiles.index(file)] )
    w.transform = 'r'
    
    # column contain the desired data: area code
    code = np.array(area_code)
    
    response = np.array(Y_norm) #Unemplyment/crime
    
    #calculate moran's I
    mi = pysal.Moran(response, w)
    print (' ',file=output)
    print ('======= Global Morans I Analysis (Single Run) ========',file=output)
    # actual observed value of spatial autocorrelation in the dataset    
    print ('Observed value for I: ', mi.I,file=output) 
    # expected value of spatial autocorrelation based on a random distribution of the dataset (will always be close to 0.0)
    print ('Expected value for I: ', mi.EI,file=output) 
    # statistical significance of difference between I and EI (goal = p < 0.05)
    # based an assumption that the response variable follow a normal distribution curve
    print ('Calculated p value: ', mi.p_norm,file=output)
    #.VI_norm: variance of I under normality assumption
    if mi.p_norm < 0.05:
        print ('Based on one run, response do not appear to be randomly distributed (p value<0.05).',file=output)
    else:
        print ('Based on one run, response appear to be randomly distributed (p value>0.05).',file=output)
    
    print (' ',file=output)
    
    #calculate randomized Moran's I
    np.random.seed(10)
    #add permutations (multiple runs) of the randomized Moran's I
    mir = pysal.Moran(response, w, permutations = 9999)
    print (' ',file=output)
    print ('======= Global Morans I Analysis (Permutations) ========',file=output)
    print ('Moran - Multiple Permutations',file=output)
    print ('Observed value for I: ', mir.I,file=output)
    print ('Expected value for I: ', mir.EI_sim,file=output)
    print ('Calculated pseudo p value based on these permutations: ', mir.p_sim,file=output)  
    print (' ',file=output)
    if mir.p_sim < 0.05:
        print ('Based on 10,000 runs, responses do not appear to be randomly distributed (p value<0.05).',file=output)
    else:
        print ('Based on 10,000 runs, responses appear to be randomly distributed (p value>0.05).',file=output)
    
    print (' ',file=output)
    
    print ('Adding z transformation to identify whether null hypothesis can be rejected',file=output)
    
    print ('Calculated z value based on these permutations: ', mir.z_sim,file=output)
    
    print ('Calculated p value based on these permutations, using a z transformation: ', mir.p_z_sim,file=output)
    
    if mir.p_z_sim < 0.05:
        print ('Based on the z transformation of p value, the null hypothesis that responses are randomly distributed is rejected (p value<0.05).',file=output)
    else:
        print ('Based on the z transformation of p value, the null hypothesis that responses are randomly distributed is not rejected (p value>0.05).',file=output)
    
    print (' ' ,file=output)
        
    # Local Indicators of Spatial Association (LISAs) for Moran’s I 
    print ('======= Local Morans I (LISA) - (Single Run) ========',file=output)
    lm = pysal.Moran_Local(response, w)
    print ('Observed value for I: ', lm.Is,file=output) 
    print( "Estimated LISA values: ", lm.EI_sim,file=output)
    print ('Calculated p value: ', lm.p_sim,file=output)
    
    lm = pysal.Moran_Local(response, w, permutations = 9999) 
    print (' ',file=output)
    print ('======= Local Morans I (LISA) - (Permutations) ========',file=output)
    print ("Number of LISA values (same as total number counties): ", len(lm.Is),file=output)   
    print( "Observed LISA values: ", lm.Is,file=output)
    print( "Estimated LISA values: ", lm.EI_sim,file=output)
    print( 'Pseudo p-values for LISAs: ', lm.p_sim,file=output)
    print (' ',file=output)
    
    #Moran Plots
    plt.figure(9)
    #Plotting the distribution of simulated I statistics, showing all 
    #of the simulated points, and a vertical line denoting the 
    #observed value of the statistic:
    #ax7 = plt.subplot(211)
    sns.kdeplot(mi.I, shade=True)
    plt.vlines(mi.sim, 0, 1)
    plt.vlines(mi.I, 0, 40, 'r')
    plt.suptitle(filename)
    figName = "lasso\\" + filename + "_LassoMoranStatistics1.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()    
    
    plt.figure(10)
    #ax8 = plt.subplot(212)
    sns.kdeplot(mi.sim, shade=True)
    plt.vlines(mi.sim, 0, 1)
    plt.vlines(mi.EI+.01, 0, 40, 'r')
    plt.suptitle(filename)
    figName = "lasso\\" + filename + "_LassoMoranStatistics2.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    
    
    #Moran scatterplot with statistically significant LISA values highlighted.
    #spatial lags
    Lag_response = pysal.lag_spatial(w, response)
    
    #plot the statistically-significant LISA values in a different color than the others
    #find all of the statistically significant LISAs. Since the p-values are in the same 
    #order as the I_i statistics, we can do this in the following way
    plt.figure(11)
    sigs = response[lm.p_sim <= .001]
    W_sigs = Lag_response[lm.p_sim <= .001]
    insigs = response[lm.p_sim > .001]
    W_insigs = Lag_response[lm.p_sim > .001]
    
    b,a = np.polyfit(response, Lag_response, 1)
    
    #plot the statistically significant points in a dark red color.
    plt.plot(sigs, W_sigs, '.', color='firebrick')
    plt.plot(insigs, W_insigs, '.k', alpha=.2)
    
    # dashed vert at mean of response
    plt.vlines(response.mean(), Lag_response.min(), Lag_response.max(), linestyle='--')
    
    # dashed horizontal at mean of lagged response
    plt.hlines(Lag_response.mean(), response.min(), response.max(), linestyle='--')
    
    # red line of best fit using global I as slope
    plt.plot(response, a + b*response, 'r')
    plt.text(s='$I = %.3f$' % mi.I, x=0.7, y=0.3, fontsize=18)
    plt.title('Moran Scatterplot')
    plt.ylabel('Spatial Lag of response')
    plt.xlabel('response')
    plt.suptitle(filename)
    figName = "lasso\\" + filename + "_LassoMoranScatterplot.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()

    ###################################################################
    ### Moran's I plot for residuals
    mi = pysal.Moran(np.array(residuals2), w)
    print (' ',file=output)
    print ('======= Global Morans I Analysis (Single Run) of Residuals ========',file=output)    
    print ('Observed value for I: ', mi.I,file=output) 
    print ('Expected value for I: ', mi.EI,file=output) 
    print ('Calculated p value: ', mi.p_norm,file=output)
    if mi.p_norm < 0.05:
        print ('Based on one run, residuals do not appear to be randomly distributed.',file=output)
    else:
        print ('Based on one run, residuals appear to be randomly distributed.',file=output)
    
    print (' ')
    
    #calculate randomized Moran's I
    np.random.seed(10)
    #add permutations (multiple runs) of the randomized Moran's I
    mir = pysal.Moran(np.array(residuals2), w, permutations = 9999)
    print (' ',file=output)
    print ('Moran - Multiple Permutations',file=output)
    print ('Observed value for I: ', mir.I,file=output)
    print ('Expected value for I: ', mir.EI_sim,file=output)
    print ('Calculated pseudo p value based on these permutations: ', mir.p_sim,file=output)  
    print (' ',file=output)
    
    if mir.p_sim < 0.05:
        print ('Based on 10,000 runs, residuals do not appear to be randomly distributed.',file=output)
    else:
        print ('Based on 10,000 runs, residuals appear to be randomly distributed.',file=output)
    
    print (' ',file=output)
    
    print ('Adding z transformation to identify whether null hypothesis can be rejected',file=output)
    
    print ('Calculated z value based on these permutations: ', mir.z_sim,file=output)
    
    print ('Calculated p value based on these permutations, using a z transformation: ', mir.p_z_sim,file=output)
    
    if mir.p_z_sim < 0.05:
        print ('Based on the z transformation of p value, the null hypothesis that residuals are randomly distributed is rejected.',file=output)
    else:
        print ('Based on the z transformation of p value, the null hypothesis that residuals are randomly distributed is not rejected.',file=output)
    
    print (' ' ,file=output)
        
    # Local Indicators of Spatial Association (LISAs) for Moran’s I 
    print ('======= Local Morans I (LISA) - (Single Run) of Residuals ========',file=output)
    lm = pysal.Moran_Local(np.array(residuals2), w)
    print ('Observed value for I: ', lm.Is,file=output) 
    print( "Estimated LISA values: ", lm.EI_sim,file=output)
    print ('Calculated p value: ', lm.p_sim,file=output)
    
    lm = pysal.Moran_Local(np.array(residuals2), w, permutations = 9999)
    print (' ',file=output)
    print ("Number of LISA values (same as total number areas): ", len(lm.Is),file=output)   
    print( "Observed LISA values: ", lm.Is,file=output)
    print( "Estimated LISA values: ", lm.EI_sim,file=output)
    print( 'Pseudo p-values for LISAs: ', lm.p_sim,file=output)
    print (' ',file=output)
    
    plt.figure(10,figsize=(8,10))
    #Plotting the distribution of simulated I statistics, showing all 
    #of the simulated points, and a vertical line denoting the 
    #observed value of the statistic:
    ax7 = plt.subplot(211)
    sns.kdeplot(mi.I, shade=True)
    plt.vlines(mi.sim, 0, 1)
    plt.vlines(mi.I, 0, 40, 'r')
    plt.title("Moran - Residuals")
    #figName = filename + "_RidgeResiduals.png"
    #plt.savefig(figName)
    #plt.show()    
    
    #plt.figure(11)
    ax8 = plt.subplot(212)
    sns.kdeplot(mi.sim, shade=True)
    plt.vlines(mi.sim, 0, 1)
    plt.vlines(mi.EI+.01, 0, 40, 'r')
    plt.suptitle(filename)
    plt.title("Moran - Residuals")
    figName = "lasso\\" + filename + "_Lasso_Residual_MoranStatistics.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    
    #Moran scatterplot with statistically significant LISA values highlighted.
    #spatial lags
    arr_residuals = np.array(residuals2)
    Lag_residual = pysal.lag_spatial(w, arr_residuals)
    
    #plot the statistically-significant LISA values in a different color than the others
    #find all of the statistically significant LISAs. Since the p-values are in the same 
    #order as the I_i statistics, we can do this in the following way
    plt.figure(12)
    sigs = arr_residuals[lm.p_sim <= .001]
    W_sigs = Lag_residual[lm.p_sim <= .001]
    insigs = arr_residuals[lm.p_sim > .001]
    W_insigs = Lag_residual[lm.p_sim > .001]
    
    b,a = np.polyfit(arr_residuals, Lag_residual, 1)
    
    #plot the statistically significant points in a dark red color.
    plt.plot(sigs, W_sigs, '.', color='firebrick')
    plt.plot(insigs, W_insigs, '.k', alpha=.2)
    
    # dashed vert at mean of the last year's PCI
    plt.vlines(arr_residuals.mean(), Lag_residual.min(), Lag_residual.max(), linestyle='--')
    
    # dashed horizontal at mean of lagged PCI
    plt.hlines(Lag_residual.mean(), arr_residuals.min(), arr_residuals.max(), linestyle='--')
    
    # red line of best fit using global I as slope
    plt.plot(arr_residuals, a + b*arr_residuals, 'r')
    s='I = %.3f' % mi.I
    plt.text(s=s, x=0.7, y=0.3, fontsize=18)
    plt.title('Moran Scatterplot')
    plt.ylabel('Spatial Lag of residuals')
    plt.xlabel('Residuals')
    plt.suptitle(filename)
    figName = "lasso\\" + filename + "_Lasso_Residual_MoranScatterplot.png"
    plt.savefig(figName,bbox_inches='tight')
    plt.show()
    
    #==================================================================    
    #=================== Moran's I and LISA Maps =============     
        
    #Before any further analysis, it is always good practice to visualize the 
    #distribution of values on a map.
    figName = "lasso\\" + filename + "_LassoResponseMap.png"
    maps.plot_choropleth(shp_link, response, 'quantiles', cmap='Greens', figsize=(9, 6),title="Percent unemployment Map",savein =figName)
    #maps.plot_choropleth(shp_link, response, 'fisher_jenks', cmap='Greens', figsize=(9, 6))
    
    lm2 = pysal.Moran_Local(np.array(residuals2), w, permutations = 9999)
    lm3 = pysal.Moran(np.array(residuals2), w)
    
    figName = "lasso\\" + filename + "_LassoResidualsMap-ols.png"
    maps.plot_choropleth(shp_link, np.array(residuals2), 'quantiles', cmap='Reds', figsize=(9, 6),title="Residual Map",savein =figName)
    #maps.plot_choropleth(shp_link, np.array(residuals2), 'fisher_jenks', cmap='Reds', figsize=(9, 6))
    
    figName = "lasso\\" + filename + "_LassoResidualLISAClusterMap-ols.png"
    #maps.plot_lisa_cluster(shp_link, lm, figsize=(9, 6), title="Lisa Cluster of response")
    maps.plot_lisa_cluster(shp_link, lm2, figsize=(9, 6), title="Lisa Cluster of residulas",savein =figName)
    
    figName = "lasso\\" + filename + "_LassoResidualsMap-lasso.png"
    maps.plot_choropleth(shp_link, np.array(lasso_residual_all), 'quantiles', cmap='Reds', figsize=(9, 6),title="Residual Map",savein =figName)
    
    figName = "lasso\\" + filename + "_LassoResidualLISAClusterMap-lasso.png"
    lm4 = pysal.Moran_Local(np.array(lasso_residual_all), w, permutations = 9999)
    maps.plot_lisa_cluster(shp_link, lm4, figsize=(9, 6), title="Lisa Cluster of residulas",savein =figName)
    
    
    #Saving map using attr savein = Path to png file where to dump the plot
    #Graphdimension using figsize = Figure dimensions as tupla
    #Explain: Moran's I shows negative spatial autocorrelation and is significant (p≤.05)

    
    #Moran of OSL residuals
    #x =np.array(X.iloc[::,index])
    x =np.array(X_lasso)
    dim=x.shape
    if dim[0] < dim[1]:
        continue
    n = len(response)
    y= np.reshape(response, (n,1))
    ols =pysal.spreg.ols.OLS(y, x)
    m_r = pysal.spreg.diagnostics_sp.MoranRes(ols, w, z=True)
    print ('======= Spatial correlation of the OSL residuals - Morans I ========',file=output)
    print( "Value of the Morans I statistic: ",round(m_r.I,4),file=output )
    print("Value of the Morans I expectation: ", round(m_r.eI,4),file=output)
    print("Value of the Morans I variance: ", round(m_r.vI,4),file=output)
    print("Value of the Morans I standardized value. This is distributed as a standard Normal(0, 1): ", round(m_r.zI,4),file=output)
    print("P-value of the standardized Moran’s I value (z): ", round(m_r.p_norm,4),file=output)

output.close()

#Save var index into cvs file
df.to_csv('lasso_vars.csv')