# A Face Recongnition Application

## Introduction

Face detection and recognition are initial steps for a wide number of applications in computer vision such as personal identity and video surveillance. Our application used the state-of-art methods for detecting faces from a RGB image and recognising the detected faces by using a trained multiclass classifier. The main aim is to compare the performance of three machine learning tecniques for face classification using diffrent feature extraction methods.

The face database was created from labelled RGB images and videos' frames of 27 known individuals - students and lectures in this module. For our experiments, we used 46 different face images per each of the 27 people in a total of 1242 face images. Before creating a classifier for the recognition, it is needed to extract features for the images. There are several tecniques to accomplish it. In our application, the  Histograms of Oriented Gradients (HOG), Local Binary Patterns (LBP) and bag of features were the features extraction methods used to encode relevant facial information. And fed these features into a supervised machine learning algorithm. 

Several supervised classification models were trained using a cross validation approach to choose the best model for the classification task. Supported Vector Machine (SVM), Neural Network (NN), and Decision Tree (DT) were the chosen methods to create the classifiers for our application. The best accuracy result was obtained by SVM, obtaining a 96% accuracy for HOG. A number of experiments were run to recognise unseen face images to compare performance results.

The diagram below shows the steps followed by a face recognition application. A detailed description of our approach for each step is given in separate subsections.

[Steps followed by  a face recognition application](faceRecognition-Steps.png)

## Faces database creation

A database of facial images was created from RGB labelled images and videos' frames of 27 individuals - students and lectures in this module - that were taken on different days, background and illumination. The images and video shows full body person in different angles with plain and clear background. The gathered images have a variety face representation for each individual – different facial expressions, skin colours, age and sex. Some people are wearing glasses and others have beard. The faces on the group images were also added to the database to provide more variety on scale, background, and lighting and face positions (looking up, left, right or downs).

To produce a consistent results, the faces were cropped to 200 x200 pixels (keeping the aspect ratio), and converted to grayscale colour space. The length of resulted database was 1242 faces with 46 different face images per person.

The faces, for the database, were detected using the cascade object detector provided by Matlab, which uses the Viola-Jones algorithm. The Viola-Jones method finds the position of the face using Haar Cascade Classifier. The object detector provided three trained cascade classification models for face detection-Frontal Face CART, Frontal Face LBP and Profile Face. All three were used to produce small changes in scale and postures.  

The Frontal Face CART and Frontal Face LBP classification models detect faces that are upright and forward but use different feature extractors and weak classifiers. Frontal Face CART uses Haar features and classification and regression tree analysis whereas   Frontal Face LBP utilises local binary patterns to encode facial features.  However the Profile Face also uses Haar features to detect upright face profiles.

We also employed the object detector to detect eyes, nose and mouth for false negative cases. The image below shows a case where the facial points detected, marked with asterisk, made hard to crop correctly to 200x200 pixels, by an automated script,  keeping its aspect ratio. Therefore this image was removed from the database. It was not relevant keeping it. 

## Feature extraction on faces

Features are patterns found in an image that distinguish images from each other. The features (or attributes) can be colour, texture, shape and length to mention few. We needed to extract facial features before to pass this facial information to the machine learning algorithms used in our experiments. Histograms of Oriented Gradients (HOG), Local Binary Patterns (LBP) and bag of features were utilised to extract local face features for comparing results. These methods are good for classification tasks.

HOG is a feature descriptor that encodes shape and spatial information, counting occurrences of gradient orientation in localized portions of an image. This method is good at detecting people. However it is not invariant to scale.  The length of our HOG feature vector was 20736 features per face. The image below displays HOG features for one of my face image in the database.

[HOG and LBP features representation]()



