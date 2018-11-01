%Training a classifier to detect FP (non-faces) using shape properties
%Features: shape properties after skin segmentation
%Classifier: SVM
%Binary classification: face(1)/non-face(0)
%Region properties:
%Area,Centroid_1,Centroid_2,BoundingBox_1,BoundingBox_2,BoundingBox_3,BoundingBox_4,
%MajorAxisLength,MinorAxisLength,Eccentricity,Orientation,ConvexArea,FilledArea,
%EulerNumber,EquivDiameter,Solidity,Extent,Perimeter,PerimeterOld
%dataset = xlsread('myDataset1-done.xls','B1:U844');

%read feature file
%dataset = xlsread('myDataset1-done.xls');
dataset = xlsread('dataset-properties.xls');
Y = dataset(:,1);
%X = dataset(:,2:20);
X = dataset(:,2:12);

%%Initialisation
avgAcc = 0; %Average accuracy
Acc = zeros(10,1); %accuracy vector
%bestModel = 1; %best model index
maxAcc = 0; %for determine best accuracy on the 10 models
best=1;

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
        
    fprintf('\n \n ---- Cross-Validation fold %d -----\n',k);
     
    %Train the classifier
    model = fitcecoc(xTrain, lTrain);
            
    %predict labels 
    plabels = predict(model,xVal);
         
    %Compute the confusion matrix
    ConfusionMat1 = confusionmat(lVal',plabels);
    %figure;plotconfusion(valLabels',plabels);
              
            
    Acc(k) = 1 - ( size(lVal,1) - sum(diag(ConfusionMat1)) )/size(lVal,1);
    avgAcc = avgAcc + (Acc(k)/10);
    
    fprintf('Accuracy(%d)= %f\n',k,Acc(k));
    %Best model - hightest accuracy
    if maxAcc < Acc(k)
        bestModel = model;
        maxAcc = Acc(k);
        
        best = k;
        
    end
    clear trainLabels;
    clear valLabels;
end

fprintf('Average Accuracy %f \n%f the best accuracy in folder %d\n',avgAcc,maxAcc,best);

%Choose the classifier with best performance(hieghtest accuracy)
classifier = bestModel;
filename = strcat('binaryClassifierSVM_2.mat');
save(filename,'classifier');

%figure;plotconfusion(lVal',plabels);



