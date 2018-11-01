function [ classifier,T] = trainDTClassifier( X,Y,cName )
%trainDTClassifier use Decistion Tree algorithm as classifier
%K-fold Cross validation is used to training different models. The model with the
%best accuracy is return as output variable (classifier).
%   Parameters
%   X:      dataset of examples
%   labels: label for each example
%   cName:  algorithm to train the classifier
%   acc:    validation set accuracy
%
%   classifier: the trained classifier. The best classifier

%Initialisation
avgAcc = 0; %Average accuracy
Acc = zeros(10,1); %accuracy vector
Acc2 = zeros(10,1); %accuracy vector
%bestModel = 1; %best model index
maxAcc = 0; %for determine best accuracy on the 10 models
best=1;

%Partion of dataset into training and validation sets
% Set the random number seed to make the results repeatable in this script
rng(1); % For reproducibility
cv = cvpartition(size(X,1),'kfold',10); %getting number of rows

%DT parameters for different models
nsplits = size(X(cv.training(1),:),1) -1;
criteria = {'gdi','deviance','twoing', 'gdi','deviance','twoing','gdi','deviance','twoing','twoing'};
for k=1:10
    % training/validation indices for the k fold
    trainIdx = cv.training(k);
    valIdx = cv.test(k);
    
    %Training Set
    xTrain = X(trainIdx,:); 
    lTrain = Y(trainIdx,:); 
        
    %Validation Set
    xVal = X(valIdx,:); 
    lVal = Y(valIdx,:); 
    noImages = size(xVal,1);
    
    for i=1:size(xTrain,1)
        trainLabels(i)=single(find(lTrain(i,:)));
    end
    for i=1:size(xVal,1)
        valLabels(i)=single(find(lVal(i,:)));
    end
    fprintf('\n \n ---- Cross-Validation fold %d -----\n',k);
    % Train the decision tree
    if mod(k,2) == 0
        splits = size(xTrain,1) - 1;
    else
        splits = round(size(xTrain,1)/2);
    end
    
    model = fitctree(xTrain,trainLabels','MaxNumSplits',splits,'SplitCriterion',criteria{k});
   
    % Make a prediction for the test set
    plVal = predict(model,xVal);
    plVal = round(plVal);
     
    [c,cm] = confusion(valLabels', plVal);
    ConfusionMat1 = confusionmat(valLabels',plVal);
    Acc(k) = trace(cm)/sum(cm(:));
    Acc2(k) = 1 - ( noImages - sum(diag(ConfusionMat1)) )/noImages;
    avgAcc = avgAcc + (Acc2(k)/10);
    
    fprintf('Method2-Accuracy(%d)= %f, fraction of samples misclassified: %f, 1-c= %f\n',k,c,1-c);
    fprintf('Accuracy(%d)= %f, Acc2 = %f\n',k,Acc(k),Acc2(k));
    %Best model - hightest accuracy
    if maxAcc < Acc2(k)
        bestModel = model;
        nsplits = splits;
        maxAcc = Acc2(k);
        best = k;
        T=table(valLabels',plVal,'VariableNames',{'TrueLabel','PredictedLabel'});
    end
    clear trainLabels;
    clear valLabels;
end
fprintf('Average Accuracy %f \n%f the best accuracy in folder %d\n',avgAcc,maxAcc,best);
fprintf('Number of splits %d \n sliptcriteria: %s\n',nsplits,criteria{best});
%Choose the classifier with best performance(hieghtest accuracy)

if 1==2
    %%%%
    %Training model with all images using best model
    noImages = size(X,1);
    
    for i=1:size(X,1)
        trainLabels(i)=single(find(Y(i,:)));
    end
    
    %Train the classifier
    model = fitctree(X,trainLabels','MaxNumSplits',825,'SplitCriterion','deviance');
    
    %predict labels
    plabels = predict(model,X);
    plabels = round(plabels);
    
    %Compute the confusion matrix
    [c,cm] = confusion(trainLabels', plabels);
    Acc2(1) = 1 - ( noImages - sum(diag(cm)) )/noImages;
    ConfusionMat1 = confusionmat(trainLabels',plabels);
    
    Acc(1) = 1 - ( noImages - sum(diag(ConfusionMat1)) )/noImages;
    T=table(trainLabels,plabels','VariableNames',{'TrueLabel','PredictedLabel'});
    fprintf('Accuracy(%d)= %f, acc2=%f\n',1,Acc(1),Acc2(1));
    %%%%
    bestModel = model;
end
classifier = bestModel;
end

