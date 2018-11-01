function [ P ] = RecogniseFace( I, featureType, classifierName )
%Recognise faces in image I and 
% classify the detected faces using a ML algorithm
%   
% I - RGB image to recognise faces and classify them
% featureType - Feature extraction supported HOG, LBP
% classifierName - Classifier (ML) algorithms supported: NN,DT,SVM and BoF
%
%%Output
% P - returned Matrix representing people in an RGB image
%
%Usage: 
% RecogniseFace( I, 'HOG', 'SVM' )
% RecogniseFace( I, 'HOG', 'DT' )
% RecogniseFace( I, 'HOG', 'NN' )
% RecogniseFace( I, 'LBP', 'SVM' )
% RecogniseFace( I, 'LBP', 'DT' )
% RecogniseFace( I, 'LBP', 'NN' )
% RecogniseFace( I, 'SURF', 'BoF' )
% RecogniseFace( I, '', 'BoF' )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P=[];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Load Trained classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
%Verify if file exits
filename = '';
switch featureType
        case 'HOG'
            filename = strcat('classifier',featureType, '-',classifierName,'.mat');
        case 'LBP'
            filename = strcat('classifier',featureType, '-',classifierName,'.mat');
        otherwise
            if strcmp(classifierName,'BoF');
                filename = strcat('classifier','SURF', '-',classifierName,'.mat');
            else
                errorMessage = sprintf ('The classifier %s and feature type %s are not supported in the system.\n',classifierName,featureType);
                errorMessage = sprintf('%s It supports feature extraction HOG/LBP combine with SVM, NN, DT.\n',errorMessage);
                errorMessage = sprintf('%s Also BoF with SURF is supported.\n',errorMessage);
                uiwait(warndlg(errorMessage));
                return;
            end
end

if ~exist(filename,'file')
    errorMessage = sprintf ('The classifier file (%s) does not exist in the current folder. Please check it and try again.',filename);
    uiwait(warndlg(errorMessage));
    return;
else
    classifier = load(filename);
end

%BinaryClassifier for filter out non-faces
if 1==2
nonFace_filename = 'binaryClassifierSVM_2.mat';
if ~exist(nonFace_filename,'file')
    errorMessage = sprintf ('The binary classifier file (%s),needed for filter out non faces, does not exist in the current folder. Please check it and try again.',nonFace_filename);
    uiwait(warndlg(errorMessage));
    return;
else
    nonFace_classifier = load(nonFace_filename);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Face recognition and extraction from image I
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP','MergeThreshold',8);
FaceDetector = vision.CascadeObjectDetector('MergeThreshold',8);
%FaceDetector = vision.CascadeObjectDetector('ClassificationModel','ProfileFace','MergeThreshold',8);
%ProfileFace
%'FrontalFaceLBP'
BBOX = step(FaceDetector,I);
if isempty(BBOX)
    FaceDetector2 = vision.CascadeObjectDetector();
    BBOX= step(FaceDetector2,I);
end
people = [];
faceBBOX = [];
label_str = [];

%post-procesing face verification rules
for face=1:size(BBOX,1)

    img = imcrop(I,BBOX(face,:));
    
    %1)Post-procesing for filter out non-face using skin segementation and
    %region area and Euler number
    
    %Al-Tairi's skin segmentation
    YUV = rgb2ycbcr(img); %YCbCr luminance (Y) and chrominance (Cb and Cr) color values
    U = YUV(:, :, 2); % Extract chrominance Cb -Blue  %explain meaning
    V = YUV(:, :, 3); % Extract chrominance Cr -Red
    R = img(:, :, 1);
    G = img(:, :, 2);
    B = img(:, :, 3);
    [rows, cols, planes] = size(img);
    skin = zeros(rows, cols);
    ind = find(80 < U & U < 130 & 136 < V & ...
        V <= 200 & V > U & R > 80 & G > 30 & ...
        B > 15 & abs(R-G) > 15);
    skin(ind) = 1;
    area = sum(skin(:));
    %stats = regionprops('table',skin,'all');
    stats = regionprops('table',skin,'Area','Centroid','MajorAxisLength','MinorAxisLength',...
             'Eccentricity', 'ConvexArea','FilledArea','EulerNumber','Solidity','Extent');
    %aspect ratio
    as = stats.MinorAxisLength/stats.MajorAxisLength; %width/height
    % compute the roundness metric
    %stats2 = regionprops('table',skin,'Perimeter');
    circularity = 4*pi*stats.Area/stats2.Perimeter^2;
    %Euler number: checking for holes in skin region
    %E = C - H = # connected regions - # holes in the region
    E = bweuler(skin);
    %Check area and Euler numer to filet out small region without holes (eyes)
    if area > 2200  || E < 0  
        
        %faceBW = imresize(rgb2gray(img), [200 NaN]);
        %str_title = sprintf('E: %d, A:%d, Ratio:%.2f, Eccentricity:%.2f, Extent:%.2f, C:%.2f',E,area,as,stats.Eccentricity,stats.Extent,circularity);
        %figure;
        %s1=subplot(141);imshow(skin);
        %subplot(142);imshow(faceBW);
        %title(s1,str_title);
        
        %if  circularity < 0.25
            %continue;
        %end
        
        %2) Post-processing filter out non-face using classifier
        %X = table2array(stats);
        %nonFace = predict(nonFace_classifier.classifier,X);
        %if nonFace == 0
            %continue;
        %end
        
        faceBBOX = cat(1,faceBBOX,BBOX(face,:));
        faceBW = imresize(rgb2gray(img), [200 NaN]);
        
        if ~isempty(faceBW)
            %Features extraction
            switch featureType
                case 'HOG' %extract info about structure of image- direction of
                    faceFeatures = extractHOGFeatures(faceBW);
                case 'LBP'
                    %Call bag of feature
                    faceFeatures = extractLBPFeatures(faceBW);
                otherwise
                    %Call bag of feature
                    faceFeatures = faceBW;
            end
            %Predict labels for extracted faces
            labelInd = predict(classifier.classifier,faceFeatures);
            
            %Create people vector
            x1 = BBOX(face,1);
            y1 = BBOX(face,2);
            x2 = x1 + BBOX(face,3);
            y2 = y1 + BBOX(face,4);
            cx = BBOX(face,1) + stats.Centroid(1);
            cy = BBOX(face,2) + stats.Centroid(2);
            
            
            if isempty(labelInd)
                %people= cat(1,people,[0 BBOX(face,1) BBOX(face,2)]);
                people= cat(1,people,[0 cx cy]);
                label_str = cat(2,label_str,'0');
            else
                if strcmp(featureType,'BoW')
                    label = str2num(classifier.classifier.Labels{labelInd});
                else
                    label = labelInd;
                end
                %people= cat(1,people,[label BBOX(face,1) BBOX(face,2)]);
                people= cat(1,people,[label cx cy]);
                label_str = cat(2,label_str,label);
            end
        end
    end
    
end

%for testing function
if 1==2
    %=size(faceBBOX,1);
    K = fspecial('disk', 40);
    J = imfilter(I,K,'replicate');
    
    for face=1:size(faceBBOX,1)
        x1 = faceBBOX(face, 1);
        y1 = faceBBOX(face, 2);
        x2 = x1+faceBBOX(face, 3);
        y2 = y1+faceBBOX(face, 4);
        J(y1:y2, x1:x2, :) = I(y1:y2, x1:x2, :);
    end
    labeledImg = insertObjectAnnotation(J,'rectangle',faceBBOX,label_str,'TextBoxOpacity',0.7,'FontSize',18);
    figure;imshow(imresize(labeledImg,0.5));
end


P = people;
end

