function [features, featureMetrics, varargout] = faceBagOfFeaturesExtractor(I)
% This function implements the default SURF feature extraction used in
% bagOfFeatures. I must be a grayscale image.
%
% [features, featureMetrics] = exampleBagOfFeaturesExtractor(I) returns
% SURF features extracted MSER regions
%
% [..., featureLocations] = exampleBagOfFeaturesExtractor(I) optionally
% return the feature locations. This is used by the indexImages function
% for creating a searchable image index.
%

%% Step 1: Select Point Locations for Feature Extraction
% Here, use a feature detector such as detectSURFFeatures
% or detectMSERFeatures to select point locations.
%Find features using MSER with SURF feature descriptor.
%Regions of uniform intensity

multiscaleSURFPoints = detectMSERFeatures(I);
                    
%% Step 2: Extract features
% Extract upright SURF features from the selected point locations. 
%Upright: when set this property to true, the orientation of the feature vectors are not estimated 
% and the feature orientation is set to pi/2. 
%Set this to true when you do not need the image descriptors to capture rotation information. 
%When you set this property to false, the orientation of the features is estimated and 
%the features are then invariant to rotation.
%SURFSize: ength of the SURF feature vector (descriptor). Largest size of 128 provides greater accuracy
%but decreases the feature matching speed.
[features, validPts] = extractFeatures(I, multiscaleSURFPoints,'Upright',true,'SURFSize',128);

%% Step 3: Compute the Feature Metric
% The feature metrics indicate the strength of each feature, where larger
% metric values are given to stronger features. The feature metrics are
% used to remove weak features before bagOfFeatures learns a visual
% vocabulary. You may use any metric that is suitable for your feature
% vectors.
%
% Use the variance of the SURF features as the feature metric.
featureMetrics = var(features,[],2);

% Alternatively, if a feature detector was used for point selection,
% the detection metric can be used. For example:
%
% featureMetrics = multiscaleSURFPoints.Metric;

% Optionally return the feature location information. The feature location
% information is used for image search applications. See the retrieveImages
% and indexImages functions.
if nargout > 2
    % Return feature location information
    varargout{1} = validPts;
end


