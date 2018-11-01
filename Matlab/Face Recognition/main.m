%Computer Vision (INM460) Coursework 
%Aniuska Dominguez
%Task 1 - Face recognition

clear all; close all;

fprintf('------------------------------------------------------------\n');
fprintf('- Coursework Computing Vision                              -\n');
fprintf('- Face Recognition using Machine Learning                  -\n');
fprintf('----- Features extraction algorithms                       -\n');
fprintf('      * HOG                                                -\n');
fprintf('      * LBP                                                -\n');
fprintf('                                                           -\n');
fprintf('----- Machine Learning algorithms                          -\n');
fprintf('      * Suppot Vector Machine (SVM)                        -\n');
fprintf('      * Neural Network (NN)                                -\n');
fprintf('      * Desicion Tree(DT)                                  -\n');
fprintf('                                                           -\n');
fprintf('----- Bag of Features (BoF)                                -\n');
fprintf('----------------------------------------------------------\n\n');
fprintf('-----------Usage of RecogniseFace function-----------------\n\n');
fprintf('RecogniseFace( I, ''HOG'', ''SVM'' )\n\n');
fprintf('RecogniseFace( I, ''HOG'', ''DT'' )\n\n');
fprintf('RecogniseFace( I, ''HOG'', ''NN'' )\n\n');
fprintf('RecogniseFace( I, ''LBP'', ''SVM'' )\n\n');
fprintf('RecogniseFace( I, ''LBP'', ''DT'' )\n\n');
fprintf('RecogniseFace( I, ''LBP'', ''NN'' )\n\n');
fprintf('RecogniseFace( I, ''SURF'', ''BoF'' )\n\n');
fprintf('RecogniseFace( I, '''', ''BoF'' )\n\n');

%%%Pick an image File
[filename, pathname] = uigetfile({'*.jpg';'*.gif';'*.bmp';'*.*'}, 'Pick an Image File');
I = imread([pathname,filename]);


%%Face Recognition: detect and label recognised faces on an image 
featureType = 'HOG';
classifierName= 'SVM';

P  = RecogniseFace( I, featureType, classifierName );

%Show group image with labeled faces
pos=zeros(size(P,1),2);
text=zeros(size(P,1),1);
for j=1:size(P,1)
    pos(j,1) = P(j,2);
    pos(j,2) = P(j,3);
    text(j) = P(j,1);
end

if isempty(P)
    pos = [10 10];
    text = ':( No face detected';
end

RGB = insertText(I,pos,text,'FontSize',80,'BoxOpacity',0,'TextColor','green');
figure;
imshow(RGB);