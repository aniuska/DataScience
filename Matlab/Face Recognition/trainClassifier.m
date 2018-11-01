function [ classifier,T] = trainClassifier( X,Y )
%Train a SVM classifier using 10-fold cross validation
%   X: Dataset of features of faces
%   Y: Label for each face
%Output
%   classifier: a SVM trained classifier which give the best accuracy
%   T : table with true and predicted labes for the best model

%%Initialisation
avgAcc = 0; %Average accuracy
Acc = zeros(10,1); %accuracy vector
%bestModel = 1; %best model index
maxAcc = 0; %for determine best accuracy on the 10 models
best=1;
best_cm =[];

%%Partion of dataset into training and validation sets
rng(1); % Set the random number seed for reproducibility

%Croos validation patition. The observations are dividedinto 10 disjoint subsamples, 
%chosen randomly but with roughly equal size.
cv = cvpartition(size(X,1),'kfold',10); %

%%Training the classifier
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
     
    %Train the classifier
    model = fitcecoc(xTrain, trainLabels');
            
    %predict labels 
    plabels = predict(model,xVal);
         
    %Compute the confusion matrix
    ConfusionMat1 = confusionmat(valLabels',plabels);
    %figure;plotconfusion(valLabels',plabels);
    %figure;plotroc(valLabels',plabels);
                       
    Acc(k) = 1 - ( noImages - sum(diag(ConfusionMat1)) )/noImages;
    avgAcc = avgAcc + (Acc(k)/10);
    
    fprintf('Accuracy(%d)= %f\n',k,Acc(k));
    %Best model - hightest accuracy
    if maxAcc < Acc(k)
        bestModel = model;
        maxAcc = Acc(k);
        T=table(valLabels',plabels,'VariableNames',{'TrueLabel','PredictedLabel'});
        best = k;
        best_cm = ConfusionMat1;
    end
    clear trainLabels;
    clear valLabels;
end

fprintf('Average Accuracy %f \n%f the best accuracy in folder %d\n',avgAcc,maxAcc,best);

T1 = array2table(best_cm);
writetable(T1,'ConfusionMat_SVM','FileType','spreadsheet');

if 1==2
%Training model with all images using best model
noImages = size(X,1);
    
for i=1:size(X,1)
    trainLabels(i)=single(find(Y(i,:)));
end

%Train the classifier
model = fitcecoc(X, trainLabels');

%predict labels
plabels = predict(model,X);

%Compute the confusion matrix
ConfusionMat1 = confusionmat(trainLabels',plabels);

Acc(1) = 1 - ( noImages - sum(diag(ConfusionMat1)) )/noImages;
T=table(trainLabels',plabels,'VariableNames',{'TrueLabel','PredictedLabel'});
fprintf('Accuracy(%d)= %f\n',1,Acc(1));
%%%%
bestModel = model;
end

%Choose the classifier with best performance(hieghtest accuracy)
classifier = bestModel;
end

