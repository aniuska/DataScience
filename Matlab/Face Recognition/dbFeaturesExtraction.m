function [X, Y] = dbFeaturesExtraction( imgDB,type )
%Extract features of face database using specified algorithm
%   Detailed explanation goes here
%   imgDB: set of face images
%   type:  feature type to extact from images
%   X:     feature matrixNxM. N number of face images and M features length
%   Y:     label vectorNx1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 1;
n = numel(imgDB); %number of people on image database
m= imgDB(1).Count; %number of face images per people

oneOfK = repmat(eye(n),1); %1-of-K representation for labels - K=# of people in database 
X = [];
Y = zeros(n * m,n);

for p=1:n
    label = imgDB(p).Description;
    for i=1:imgDB(p).Count %number of images of person p
        I = read(imgDB(p),i);
        
        switch type
            case 'LBP' 
                %
                features = extractLBPFeatures(I);
            otherwise
                %HOG-extract info about structure of image- direction of
                features = extractHOGFeatures(I);
        end
        
        X =cat(1,X,features);
        Y(N,:) = oneOfK(:,str2num(label));
        N = N + 1;
        
    end
end

end

