
# Guide Regularised Random Forest
# Evaluating performance using CV


library(RRF)

set.seed(1)

nRep <- 100
nTree <- 1000

#Setting working directory
#setwd("C:\\Users\\Ana\\Dropbox\\__City University\\__Disertation\\code-scripts\\")

setwd("C:\\Users\\IT\\Dropbox\\__City University\\__Disertation\\code-scripts\\")

#Datasets
unempfiles = c("Crime_Rate_LA.csv","Crime_Rate_London.csv","Crime_Rate_London_MP.csv","Crime_Rate_wards.csv","Crime_Rate_lsoa.csv")
#unempfiles = c("Census_Per_LA.csv","Census_Per_London.csv","Census_Per_Ward_London.csv","Census_Per_LSOA_London.csv")
plotTitle = c("% Unemployment - LA","% Unemployment - Boroughs","% Unemployment - Wards","% Unemployment - LSOA")

#unempfiles =c("Crime_Rate_LA.csv")

data_path= 'data'
shapefiles_path = 'shapefiles'

#For unemployment
#pop_ix = 3
#den_ix = 9
#y_ix = 48

#For Crime
rng = range(66)
pop_ix = 2
den_ix = 12
y_ix = 5

seeds<-c(10,159,538)

#name="Census_Per_Ward_London.csv"
#output = paste("forest","output_normalised_5CV_GRRF.txt",sep="/")
output = paste("forest","output_normalised_crime_5CV_GRRF.txt",sep="/")
sink(output)

#Unemployment
#loop
for ( name in unempfiles) {
  #Read non-spatial data from .csv
  filename  = paste(data_path,name,sep="/")
  unemp_data <- read.csv(filename,
                         stringsAsFactors = FALSE,
                         header = TRUE,
                         check.names = FALSE
  )
  #Get variable names to be used (header of columns)
  census = strsplit(name, "[.]")[[1]]
  
  #var_names$varName[var_names$Filename==census[1]]
  #For unemployment
  #s<-c(colnames(unemp_data)[0:pop_ix],
       #colnames(unemp_data)[(pop_ix+1):den_ix],
       #colnames(unemp_data)[(den_ix+2):y_ix]
  #)
  
  #For crime
  s<-c(colnames(unemp_data)[0:pop_ix],
  colnames(unemp_data)[(pop_ix+3):(pop_ix+10)],
  colnames(unemp_data)[(pop_ix+12):66]
  )
  cols = unlist(strsplit(paste(s,collapse=','),",")[[1]])
  
  #remove unnecessary columns
  data = subset(unemp_data, select = cols)
  
  #For unemployment
  #X1 = data[(pop_ix+1):(y_ix - 2)]
  #Y1 = data[(y_ix - 1)]
  
  #For crime
  X1 = data[y_ix:63]
  Y1 = data[(y_ix - 1)]
  
  #normalisind data
  X = scale(X1)
  Y = scale(Y1,scale=FALSE)
  
  cat("=====================================================================\n")
  #cat("============== Random Forest for Unemployment :",filename)
  cat("============== Random Forest for Crime :",filename)
  cat("\n=====================================================================\n")
  cat("\nNormalised\n")
  
  #for each seed
  set.seed(seeds[1])
  cat("\nSeed used: ",seeds[1],filename)
  
  sFactor = 1.5 
  #sFactor = 2.5
  #tuning RRF for the optimal parameter mtry
  rf.tuning <- tuneRRF(X, Y[,1], ntreeTry = nTree, mtryStart = 5, stepFactor = sFactor,doBest=TRUE)
  mtry = rf.tuning$mtry
  
  #if (mtry < 2) {mtry = sqrt(length(X))}
  
  cat("\n\nOptimal mtry for RRF: ",mtry)
  cat("\n\nStepFacto for RRF: ",sFactor)
  
  bestpar = c(0.0001)
  minMSE = 100
  #lambda in c(0.01, 0.1, 0.25, 0.40, 0.65,0.8, 0.9)
  for ( lambda in c(0.0001, 0.001,  0.01, 0.1)) {
    
    result <- RRF(X,Y[,1],
                  flagReg=1,
                  coefReg=lambda,
                  ntree=nTree,
                  importance=FALSE,
                  
                  mtry=mtry 
                  
    )
    if (mean(result$mse) < minMSE) {
      #if (mean(result$rsq) > maxR2) {
      bestpar = lambda
      minMSE = mean(result$mse) #Avg. in ntree
      #maxR2 = mean(result$rsq) #Avg. in ntree - pseudo R-squared
    }
    
  }
  
  cat("\n\nOptimal penalty parameter:",bestpar)
  cat("\n\nMin. mean(MSE):",minMSE)
  #cat("\n\nMax. R_squared:",maxR2)
  
  #minsubset = min()
  lambda = bestpar 
  mtry=mtry
  
  census.rf <- RRF(X,Y[,1], flagReg = 0,ntree=nTree,importance=TRUE,mtry=mtry) # build an ordinary RF 
  impRF <- census.rf$importance #get variable importance
  impRF <- impRF[,"%IncMSE"] # get the importance score 
  imp <- impRF/(max(impRF)) #normalize the importance scores into [0,1]
  
  #Guided Regularised Random Forest (GRRF)
  cat('\n================ Guided Regularised Random Forest (GRRF) ===================\n')
  cat('\n================  5-folds CV for findbest gamma in GRRF) ===================\n')
  
  for ( gamma in c(0.1,0.5,0.9,0.99)) {
    set.seed(seeds[1]+gamma)
    
    #Randomly shuffle the data
    census_shuffle<-data[sample(nrow(data)),]
    
    #Create 5 equally size folds
    folds <- cut(seq(1,nrow(census_shuffle)),breaks=5,labels=FALSE)
    
    MSE <- vector(mode="numeric", length=10)
    MSE_0.01 <- vector(mode="numeric", length=10)
    
    for(i in 1:5){
      #normalisind data
      #For unemployment
      #X = scale(census_shuffle[(pop_ix+1):(y_ix - 2)])
      #Y = scale(census_shuffle[(y_ix - 1)],scale=FALSE)
      
      #For crime
      X = scale(census_shuffle[y_ix:63])
      Y = scale(census_shuffle[(y_ix - 1)],scale=FALSE)
      
      testInd <- which(folds ==i,arr.ind=TRUE)
      testX <- X[testInd, ]
      trainX <- X[-testInd, ]
      
      testY <- Y[testInd, ]
      trainY <- Y[-testInd, ]
      
      cat("\n======================= Fold ",i)
      
      cat("\n=============== Lambda = 1")
      #gamma <- 0.5   #A larger gamma often leads to fewer features. But, the quality of the features selected is quite stable for GRRF, i.e., different gammas can have similar accuracy performance (the accuracy of an ordinary RF using the feature subsets). See the paper for details. 
      coefReg <- (1-gamma) + gamma*imp   # each variable has a coefficient, which depends on the importance score from the ordinary RF and the parameter: gamma
      census.grrf <- RRF(trainX,trainY, flagReg=1, 
                         coefReg=coefReg,
                         ntree=nTree,
                         importance=TRUE,
                         localImp=TRUE,
                         mtry=mtry,
                         xtest = testX,
                         ytest=testY)
      subsetGRRF <- census.grrf$feaSet # produce the indices of the features selected by GRRF
      cat("\n\n The gamma value used by GRRF:",gamma)
      #cat("\n\n The regularised coeffcient value used by GRRF:",coefReg)
      
      MSE[i]<- mean(census.grrf$test$mse)
      
      cat("\n\nNumber of variable subset generated by GRRF:",length(subsetGRRF)) #the subset includes many more noisy variables than GRRF
      cat("\n\nGRRF variable subset:\n",subsetGRRF)
      cat("\n\nGRRF Mean(MSE):",mean(census.grrf$mse))
      cat("\n\nGRRF R-squeared-Avg.:",mean(census.grrf$rsq))
      
      
      cat("\n======================== Test set ================\n")
      cat("\n\nGRRF Mean(MSE):",mean(census.grrf$test$mse))
      cat("\n\nGRRF R-squeared-Avg.:",mean(census.grrf$test$rsq))
      
      cat('\n====================================================================================\n')
      
      cat("\n================================== lambda = ", lambda)
      coefReg <- (1-gamma) * lambda + gamma*imp   # each variable has a coefficient, which depends on the importance score from the ordinary RF and the parameter: gamma
      census.grrf <- RRF(trainX,trainY, flagReg=1, 
                         coefReg=coefReg,
                         ntree=nTree,
                         importance=TRUE,
                         localImp=TRUE,
                         mtry=mtry,
                         xtest = testX,
                         ytest=testY)
      subsetGRRF <- census.grrf$feaSet # produce the indices of the features selected by GRRF
      cat("\n\n The gamma value used by GRRF:",gamma)
      #cat("\n\n The regularised coeffcient value used by GRRF:",coefReg)
      
      MSE_0.01[i]<- mean(census.grrf$test$mse)
      
      cat("\n\nNumber of variable subset generated by GRRF:",length(subsetGRRF)) #the subset includes many more noisy variables than GRRF
      cat("\n\nGRRF variable subset:\n",subsetGRRF)
      #cat("\n\nRRF MSE:",census.grrf$mse)
      cat("\n\nGRRF Mean(MSE):",mean(census.grrf$mse))
      #cat("\n\nRRF R-squeared:",census.grrf$rsq)
      cat("\n\nGRRF R-squeared-Avg.:",mean(census.grrf$rsq))
      #cat("\n\nGRRF R-squeared-Min:",min(census.grrf$rsq))
      #cat("\n\nGRRF R-squeared:-Max",max(census.grrf$rsq))
      
      cat("\n======================== Test set ================\n")
      cat("\n\nGRRF Mean(MSE) - :",mean(census.grrf$test$mse))
      cat("\n\nGRRF R-squeared-Avg. - :",mean(census.grrf$test$rsq))
      
      cat('\n====================================================================================\n')
    }
    
    cat("\nGamma value:", gamma)
    cat("\nGRRF Avg. MSE in 5-folds with la,bda = 1:",mean(MSE))
    cat("\nGRRF Avg. MSE in 5-folds with best lambda:",mean(MSE_0.01))
  
  }
  
  
}

closeAllConnections()

