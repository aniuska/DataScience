# Regularised Random Forest


library(RRF)
library(randomForest)
set.seed(1)

nRep <- 100
nTree <- 1000

#Setting working directory
#setwd("C:\\Users\\Aniuska\\Dropbox\\__City University\\__Disertation\\code-scripts\\")
#setwd("C:\\Users\\Ana\\Dropbox\\__City University\\__Disertation\\code-scripts\\")

setwd("C:\\Users\\IT\\Dropbox\\__City University\\__Disertation\\code-scripts\\")

#Datasets
unempfiles = c("Crime_Rate_LA.csv","Crime_Rate_London.csv","Crime_Rate_London_MP.csv","Crime_Rate_wards.csv","Crime_Rate_lsoa.csv")
#unempfiles = c("Census_Per_LA.csv","Census_Per_London.csv","Census_Per_Ward_London.csv","Census_Per_LSOA_London.csv")
plotTitle = c("% Unemployment - LA","% Unemployment - Boroughs","% Unemployment - Wards","% Unemployment - LSOA")

#unempfiles =c("Census_Per_LSOA_London.csv")

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
#output = paste("forest","output_normalised.txt",sep="/")
output = paste("forest","output_normalised_crime.txt",sep="/")
sink(output)

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
  
  #Original Random Forest
  census.rf1 = randomForest(x=X, y=Y[,1],ntree=nTree,importance=TRUE, mtry=mtry) 
  imp = census.rf1$importance
  imp = imp[,"%IncMSE"]
  impNORM = imp/(max(imp)) #normalize the importance scores into [0,1]
  #fsRF = census.rf1$feaSet
  
  file_name = paste("Forest",census[1],"MSE",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("forest",file_name,sep="/")
  png(filename=file_name,width = 900, height = 600, units = 'px')
  plot(census.rf1,main=paste("Original RF - MSE vs. Number of trees\n",census[1]),log="y")
  dev.off()
  
  file_name = paste("Forest",census[1],"VarsImp",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("forest",file_name,sep="/")
  png(filename=file_name,width = 900, height = 900, units = 'px')
  par(las=2) # make label text perpendicular to axis
  par(mar=c(5,6,2,2)) # increase y-axis margin
  barplot(t(impNORM), 
          main=paste("Original Random Forest Variable Importance\n",census[1]),
          horiz=TRUE,
          xlab="Importance",
          col="darkgreen")
  dev.off()

  #Original Random Forest using RRF
  census.rf = RRF(x=X, y=Y[,1],flagReg=0,ntree=nTree,importance=TRUE, mtry=mtry) 
  imp = census.rf$importance
  imp = imp[,"%IncMSE"]
  impNORM = imp/(max(imp)) #normalize the importance scores into [0,1]
  cat('\n====== Original Random Forest using RRF ======\n')
  cat("\n\nOriginal RF number of variables using RRF package\n ",length(census.rf$feaSet))
  cat("\n\nVariables chosen by roandon foerest- RRF package\n ",census.rf$feaSet)
  #cat("\n\nRRF MSE:",census.rf$mse)
  cat("\n\nRF by RRF Mean(MSE):",mean(census.rf$mse))
  cat("\n\nRF by RRF R-squeared-Avg.:",mean(census.rf$rsq))
  cat("\n\nRF by RRF R-squeared-Min:",min(census.rf$rsq))
  cat("\n\nRF by RRF R-squeared:-Max",max(census.rf$rsq))
  
  #plot(census.rf) #plot error vs. ntrees

  file_name = paste("Forest",census[1],"VarsImpRRF-package",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("forest",file_name,sep="/")
  png(filename=file_name,width = 900, height = 900, units = 'px')
  par(las=2) # make label text perpendicular to axis
  par(mar=c(5,6,2,2)) # increase y-axis margin
  barplot(t(impNORM), 
          main=paste("Original Random Forest Variable Importance (RRF package)\n",census[1]),
          horiz=TRUE,
          xlab="Importance",
          col="orange")
  dev.off()

  #Regularised Random Forest (RRF)
   
  cat('\n====== Regularised Random Forest using RRF ======\n')
  bestpar = c(0.0001)
  minMSE = 100
  #maxR2 = -100
  
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
     
     cat("\n\npenalty parameter:",lambda)
     cat("\n\\nmean(MSE):",mean(result$mse))
     cat("\n\\nnVars:",length(result$feaSet))
  }

  cat("\n\nOptimal penalty parameter:",bestpar)
  cat("\n\nBest-Min. mean(MSE):",minMSE)
  #cat("\n\nMax. R_squared:",maxR2)
  
  #minsubset = min()
  lambda = bestpar 
  mtry=mtry

  ## coefReg is a constant for all variables.   
  #either "X,as.factor(class)" or data frame like "Y~., data=data" is fine, 
  #but the later one is significantly slower. 
  census.rrf = RRF(X,Y[,1],flagReg=1,coefReg=lambda,
                   ntree=nTree,importance=TRUE, mtry=mtry) 

  # produce the indices of the features selected by RRF
  subsetRRF = census.rrf$feaSet 
  cat("\n\nRRF number of variables: ",length(subsetRRF))
  cat("\n\nRRF variable subset:\n",subsetRRF)
  #cat("\n\nRRF MSE:",census.rrf$mse)
  cat("\n\nRRF Mean(MSE):",mean(census.rrf$mse))
  #cat("\n\nRRF R-squeared:",census.rrf$rsq)
  cat("\n\nRRF R-squeared-Avg.:",mean(census.rrf$rsq))
  cat("\n\nRRF R-squeared-Min:",min(census.rrf$rsq))
  cat("\n\nRRF R-squeared:-Max",max(census.rrf$rsq))
  
  vars = colnames(X)[subsetRRF]
  #for (v in vars) {
    #RRF::partialPlot(census.rrf, data, x.var=v, "versicolor")
    
  #}

  file_name = paste("RRF",census[1],"MSE",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("forest",file_name,sep="/")
  png(filename=file_name,width = 900, height = 600, units = 'px') 
  plot(census.rrf, main=paste("RRF MSE vs. Number of trees\n",census[1]),log="y")
  dev.off()

  impRRF <- census.rrf$importance 
  impRRF<- impRRF[,"%IncMSE"] # get the importance score 
  imp <- impRRF/(max(impRRF)) #normalize the importance scores into [0,1]

  file_name = paste("RRF",census[1],"VarsImp",sep="_")
  file_name = paste(file_name,"png",sep=".")
  file_name = paste("forest",file_name,sep="/")
  png(filename=file_name,width = 900, height = 900, units = 'px')
  par(las=2) # make label text perpendicular to axis
  par(mar=c(5,6,2,2)) # increase y-axis margin
  barplot(t(imp), 
          main=paste("Regularised Random Forest Variable Importance\n",census[1]),
          horiz=TRUE,
          xlab="Importance",
          col="orange")
  dev.off()

  #Guided Regularised Random Forest (GRRF)
  cat('\n================ Guided Regularised Random Forest (GRRF) with lambda = 1===================\n')
  cat("\nUsing lambda =1\n")
  
  for ( gamma in c(0.1,0.5,0.9,0.99)) {
    census.rf <- RRF(X,Y[,1], flagReg = 0,ntree=nTree,importance=TRUE) # build an ordinary RF 
    impRF <- census.rf$importance #get variable importance
    impRF <- impRF[,"%IncMSE"] # get the importance score 
    imp <- impRF/(max(impRF)) #normalize the importance scores into [0,1]
    
    #gamma <- 0.5   #A larger gamma often leads to fewer features. But, the quality of the features selected is quite stable for GRRF, i.e., different gammas can have similar accuracy performance (the accuracy of an ordinary RF using the feature subsets). See the paper for details. 
    coefReg <- (1-gamma) + gamma*imp   # each variable has a coefficient, which depends on the importance score from the ordinary RF and the parameter: gamma
    census.grrf <- RRF(X,Y[,1], flagReg=1, coefReg=coefReg,ntree=nTree,importance=TRUE,localImp=TRUE)
    subsetGRRF <- census.grrf$feaSet # produce the indices of the features selected by GRRF
    cat("\n\n The gamma value used by GRRF:",gamma)
    cat("\n\n The regularised coeffcient value used by GRRF:",coefReg)
    
    str_gamma <- toString(gamma)
    file_name = paste("VarImpPlot",census[1],"GRRF_1",str_gamma,sep="_")
    file_name = paste(file_name,"png",sep=".")
    file_name = paste("forest",file_name,sep="/")
    png(filename=file_name,width = 900, height = 600, units = 'px')
    
    RRF::varImpPlot(census.grrf)
    dev.off()
    
    file_name = paste("GRRF",census[1],"VarsImp_1",str_gamma,sep="_")
    file_name = paste(file_name,"png",sep=".")
    file_name = paste("forest",file_name,sep="/")
    png(filename=file_name,width = 900, height = 900, units = 'px')
    par(las=2) # make label text perpendicular to axis
    par(mar=c(5,6,2,2)) # increase y-axis margin
    barplot(t(imp), 
            main=paste("Guided Regularised Random Forest Variable Importance\n",census[1]),
            horiz=TRUE,
            xlab="Importance",
            col="orange")
    dev.off()
    
    file_name = paste("GRRF",census[1],"MSE_1",str_gamma,sep="_")
    file_name = paste(file_name,"png",sep=".")
    file_name = paste("forest",file_name,sep="/")
    png(filename=file_name,width = 900, height = 900, units = 'px')
    
    plot(census.grrf, log="y",main=paste("GRRF MSE vs. Number of trees\n",census[1]))
    
    dev.off()
    
    cat("\n\nNumber of variable subset generated by GRRF:",length(subsetGRRF)) #the subset includes many more noisy variables than GRRF
    cat("\n\nGRRF variable subset:\n",subsetGRRF)
    #cat("\n\nRRF MSE:",census.grrf$mse)
    cat("\n\nGRRF Mean(MSE):",mean(census.grrf$mse))
    #cat("\n\nRRF R-squeared:",census.grrf$rsq)
    cat("\n\nGRRF R-squeared-Avg.:",mean(census.grrf$rsq))
    cat("\n\nGRRF R-squeared-Min:",min(census.grrf$rsq))
    cat("\n\nGRRF R-squeared:-Max",max(census.grrf$rsq))
    cat('\n====================================================================================\n')
    
  }
  cat('\n====================================================================================\n')
  cat('\n====================================================================================\n')
  
  #Guided Regularised Random Forest (GRRF)
  cat('\n================ Guided Regularised Random Forest (GRRF) with best Lambda===================\n')
  cat("\nUsing optimal lambda  = ",lambda) 
  
  for ( gamma in c(0.1,0.5,0.9,0.99)) {
    census.rf <- RRF(X,Y[,1], flagReg = 0,ntree=nTree,importance=TRUE) # build an ordinary RF 
    impRF <- census.rf$importance #get variable importance
    impRF <- impRF[,"%IncMSE"] # get the importance score 
    imp <- impRF/(max(impRF)) #normalize the importance scores into [0,1]
    
    #gamma <- 0.5   #A larger gamma often leads to fewer features. But, the quality of the features selected is quite stable for GRRF, i.e., different gammas can have similar accuracy performance (the accuracy of an ordinary RF using the feature subsets). See the paper for details. 
    coefReg <- (1-gamma) * lambda + gamma*imp   # each variable has a coefficient, which depends on the importance score from the ordinary RF and the parameter: gamma
    census.grrf <- RRF(X,Y[,1], flagReg=1, coefReg=coefReg,ntree=nTree,importance=TRUE,localImp=TRUE)
    subsetGRRF <- census.grrf$feaSet # produce the indices of the features selected by GRRF
    cat("\n\n The gamma value used by GRRF:",gamma)
    cat("\n\n The regularised coeffcient value used by GRRF:",coefReg)
    
    #str_gamma <- toString(gamma)
    str_gamma <- paste(gamma,lambda,sep="_")
    file_name = paste("VarImpPlot",census[1],"GRRF",str_gamma,sep="_")
    file_name = paste(file_name,"png",sep=".")
    file_name = paste("forest",file_name,sep="/")
    png(filename=file_name,width = 900, height = 600, units = 'px')
    
    RRF::varImpPlot(census.grrf)
    dev.off()
    
    file_name = paste("GRRF",census[1],"VarsImp",str_gamma,sep="_")
    file_name = paste(file_name,"png",sep=".")
    file_name = paste("forest",file_name,sep="/")
    png(filename=file_name,width = 900, height = 900, units = 'px')
    par(las=2) # make label text perpendicular to axis
    par(mar=c(5,6,2,2)) # increase y-axis margin
    barplot(t(imp), 
            main=paste("Guided Regularised Random Forest Variable Importance\n",census[1]),
            horiz=TRUE,
            xlab="Importance",
            col="orange")
    dev.off()
    
    file_name = paste("GRRF",census[1],"MSE",str_gamma,sep="_")
    file_name = paste(file_name,"png",sep=".")
    file_name = paste("forest",file_name,sep="/")
    png(filename=file_name,width = 900, height = 900, units = 'px')
    
    plot(census.grrf, log="y",main=paste("GRRF MSE vs. Number of trees\n",census[1]))
    
    dev.off()
    
    cat("\n\nNumber of variable subset generated by GRRF:",length(subsetGRRF)) #the subset includes many more noisy variables than GRRF
    cat("\n\nGRRF variable subset:\n",subsetGRRF)
    #cat("\n\nRRF MSE:",census.grrf$mse)
    cat("\n\nGRRF Mean(MSE):",mean(census.grrf$mse))
    #cat("\n\nRRF R-squeared:",census.grrf$rsq)
    cat("\n\nGRRF R-squeared-Avg.:",mean(census.grrf$rsq))
    cat("\n\nGRRF R-squeared-Min:",min(census.grrf$rsq))
    cat("\n\nGRRF R-squeared:-Max",max(census.grrf$rsq))
    cat('\n====================================================================================\n')
  }
}

closeAllConnections()
