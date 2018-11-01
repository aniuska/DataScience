function [classifier,T] = trainNNClassifier( X,Y,cName )
%Train a NN classifier using 10-fold cross validation
%   X: example dataset
%   labels: Label for each example
%   cName: algorithm to train the classifier
%   acc:   validation set accuracy
%   classifier: the trained classifier. The best classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialisation
avgAcc = 0; %Average accuracy
Acc = zeros(10,1); %accuracy vector
%bestModel = 1; %best model index
maxAcc = 0; %for determine best accuracy on the 10 models
best=1;

%Partion of dataset into training and validation sets
% Set the random number seed to make the results repeatable in this script
rng(1); % For reproducibility
cv = cvpartition(size(X,1),'kfold',10); %getting number of rows

%number of hidden neurons
nn= 20; %start number of hidden neurons. It is increased by in each fordel iteration
for k=1:10
    % training/validation indices for the k fold
    trainIdx = cv.training(k);
    valIdx = cv.test(k);
    
    %Training Set
    xTrain = X(trainIdx,:); xTrain = xTrain';
    lTrain = Y(trainIdx,:); lTrain = lTrain';
        
    %Validation Set
    xVal = X(valIdx,:); xVal = xVal';
    lVal = Y(valIdx,:); lVal = lVal';
    noImages = size(xVal,2);
    
    for i=1:noImages
        trainLabels(i)=uint8(find(lTrain(:,i)));
        valLabels(i)=uint8(find(lVal(:,i)));
    end
    
    fprintf('\n \n ---- Cross-Validation fold %d -----\n',k);
     
    %Train the classifier
    model = feedforwardnet(nn*k, 'trainscg');
    model = configure(model,xTrain,lTrain);
    model = train(model,xTrain, lTrain);    

    %labels prediction
    plTrain = model(xTrain);
    plVal = model(xVal);
    for i = 1 : noImages
        [value, predLabels(1,i)] = max(plTrain(:,i));
        [value, predLabelsVal(1,i)] = max(plVal(:,i));
    end
    
    %Network performance and accuracy for evaluation set
    %network performance
    perf = perform(model,plVal,lVal);
    
    %Accuracy is the proximity of measurement results to the true value;
    %Accuracy refers to the closeness of a measured value to a standard or known value.
    [c,cm] = confusion(lVal, plVal);
    %c: fraction of samples misclassified
    %cm: confusion matrix
    Acc(k) = trace(cm)/sum(cm(:)); %sum(predLabelsVal == valLabels)/noImages;
    avgAcc = avgAcc + (Acc(k)/10);
    
    fprintf('%d, fraction of samples misclassified: %f, 1-c= %f\n',k,c,1-c);
    fprintf('Accuracy(%d)= %f\n',k,Acc(k));
    %Best model - hightest accuracy
    if maxAcc < Acc(k)
        bestModel = model;
        maxAcc = Acc(k);
        best = k;
        %T=table(valLabels,predLabelsVal','VariableNames',{'TrueLabel','PredictedLabel'});
        T=table(valLabels,predLabelsVal,'VariableNames',{'TrueLabel','PredictedLabel'});
        bestperf = perf;
    end
    
end

fprintf('Average Accuracy %f\nThe best accuracy %f in folder %d, net performance=%f\n',avgAcc,maxAcc,best,bestperf);

if 1==2
    %Training model with all images using best model
    X1 = X';
    Y1 = Y';
    clear  trainLabels;
    noImages = size(X1,2);
    for i=1:noImages
        trueLabels(i)=uint8(find(Y1(:,i)));
    end
    
    %Train the classifier
    model = feedforwardnet(nn*7, 'trainscg');
    model = configure(model,X1,Y1);
    model = train(model,X1, Y1);
    
    %predict labels
    plabels = model(X1);
    
    %for i=1:noImages
    %trainedLabels(i)=uint8(find(plabels(:,i)));
    %end
    
    %Compute the confusion matrix
    [c,cm] = confusion(Y1,plabels);
    T=[];
    Acc(1) = 1 - ( noImages - sum(diag(cm)) )/noImages;
    %T=table(trueLabels,trainedLabels,'VariableNames',{'TrueLabel','PredictedLabel'});
    fprintf('Accuracy(%d)= %f\n',1,Acc(1));
    fprintf('%d, fraction of samples misclassified: %f, 1-c= %f\n',1,c,1-c);
    %%%%
    bestModel = model;
end
%Choose the classifier with best performance(hieghtest accuracy)
classifier = bestModel;
end

