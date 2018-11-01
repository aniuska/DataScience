function [ classifier,T] = trainBoWClassifier( X )
%Train a classifier for especified algorithm using 10-fold cross validation
%   X: images dataset
%   classifier: the trained classifier. The best classifier

%Initialisation
avgAcc = 0; %Average accuracy
Acc = zeros(10,1); %accuracy vector
%bestModel = 1; %best model index
maxAcc = 0; %for determine best accuracy on the 10 models
best=1;

%Partion of dataset into training and validation sets
% Set the random number seed to make the results repeatable in this script
rng(1); % For reproducibility

for k=1:10
    % training/validation indices for the k fold
    [xTrain, xVal]=partition(X,0.15 ,'randomize');
                
    fprintf('\n \n ---- Cross-Validation fold %d -----\n',k);
     
    %Train the classifier
    if mod(k,2) == 0 %using grid
        bag = bagOfFeatures(xTrain,'Verbose',false);
    else %using MSER with SURF feature descriptor
        extractorFcn = @faceBagOfFeaturesExtractor;
        bag = bagOfFeatures(xTrain,'Verbose',false,'CustomExtractor',extractorFcn);
    end
    
    classifier = trainImageCategoryClassifier(xTrain,bag);
            
    %Evaluating classifier on Validation Set
    [comfMatrix,knownLabelIdx,predictedLabelIdx] = evaluate(classifier,xVal,'Verbose',false);
                         
    Acc(k) = mean(diag(comfMatrix));
    avgAcc = avgAcc + (Acc(k)/10);
    
    fprintf('Accuracy(%d)= %f\n',k,Acc(k));
    %Best model - hightest accuracy
    if maxAcc < Acc(k)
        bestModel = classifier;
        maxAcc = Acc(k);
        best = k;
        T=table(classifier.Labels(knownLabelIdx)',classifier.Labels(predictedLabelIdx)','VariableNames',{'TrueLabel','PredictedLabel'});
    end
    
    clear 'classifier';
    clear 'comfMatrix';
end

fprintf('Average Accuracy %f \n%f the best accuracy in folder %d\n',avgAcc,maxAcc,best);

%Choose the classifier with best performance(hieghtest accuracy)
classifier = bestModel;
end

