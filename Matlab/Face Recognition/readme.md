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


LBP features can provide robustness against variation in illumination and rotation but not invariant to scale. LBP features are often used to detect faces because they work well for representing fine-scale textures. Our LBP feature vector’s length was 57 features per face. Image above displays a LBP features, as a block, for one of my face image in the database.

Bag of features approach uses a sparse set of visual word histograms (dictionary) to represent a collection of images in compacted manner. The histogram bins represent the count of occurrence of the visual word in the image that it represents.  It removes spatial layout but increases the invariance to scale and deformation. To extract the features, a combination of Maximally Stable Extremal Regions (MSER) and SURF (Speeded-Up Robust Features) was used which ensure good feature localization. The MSER detects regions of uniform intensity and handles changes to scale and rotation. SURF provides invariance to small positional shifts of the regions when extracting features on images with a difference in scale and camera position.

## Supervised machine learning algorithms

Face recognition is a multiclass classification problem. In our case a face image is classified into one of 27 possible individual classes. Experiments were run using three supervised machine learning algorithms - SVM, NN and DT - to create a multiclass classifier to recognise unseen faces. A 10-fold cross validation approach was used to choose the best model (classifier), for the recognition step, based on accuracy. The dataset was split, randomly, into training and evaluation sets, based on the default Matlab ratio portion.

SVM is a linear binary classifier which can be used for multiple class classification, creating one SVM per class. It guarantees good predictive performance. We training a SVM classifier using the fitcecoc function provided by Matlab. 

A feed forward neural network is an artificial neural network where the information moves in only one direction, forward, from the input nodes, through the hidden nodes and to the output nodes. We trained several multilayer feed forward NN using the feedforwardnet function. The number of input nodes varied according to the features extractor used (20736 for HOG and 57 for LBP). The number of output was 27 (number of classes). Several numbers of hidden neurons were used to train our NN classifiers. 

DT is a non-probabilistic technique whereby the input space is recursively partitioned. This offers a graphical representation of a Boolean or discrete function, represented by a tree. 

In our experiments, we trained six classifiers - feeding each machine learning method with HOG and LBP feature vectors. The table below shows the accuracy results on the evaluation dataset for the six combinations. We also run an experiment using SVM with bag of features to see the performance of SVM using a more robust feature extraction (MSER and SURF).

The average accuracy was the performance metric employed. The best accuracy was for SVM combined with HOG features, which was 96%. However it got the worst performance when used with LBP features. SVM combined with bag of features shown good performance as well. See Table 1. 

Similar performances were obtained for NN and DT. But NN shown improvements when the number of hidden neurons increase (best accuracy was for 100 hidden neurons). Although DT used models based on different combinations of number of splits and splitting criterion, it did not improve the performance over cross iterations.

## Recognition results

Face recognition is a classification task where we evaluated how well our classifier performs on unseen image.  We created a function called FaceRecognition to detect and recognise faces in an image.  The function returns a matrix with identification and position of detected faces (if any). The function receives an image, the type of features to extract and the classification method to use. This function followed the steps shown on Fig 1 for faces recognition (face detection, image registration, features extraction and recognition). We evaluated the performance of recognition (determining the class membership of each detected face in the image) based on the best classifier model which was SVM with HOG features). 

The faces on the passed image were detected using the cascade object detector provided by Matlab. We tried both the Frontal Face CART and Frontal Face LBP classification models, to provide a discussion and analysis of our classifiers accuracy for this report.  The threshold parameter was increased to 8 to help removing false detections on areas where there was multiples detection around an object. The best method was CART.

A post-processing verification was done after the faces were detected to reduce non-face detection (false positive). Two approaches were tried, without success, for the post-processing step. Our first approach was skin segmentation, using the method proposed by Al-Tairi et al. which performs simple skin detectors using colour thresholding. This method converts RBG images to YUV colour space to extract chrominance components and eliminate the effects of luminance. The colour channels are threshold as followed. 
Skin = (80 < U < 130) and 
           (136 < V < 200) and 
           (V > U) & (R > 80) and 
           (G > 30) & (B > 15) and 
           |R-G| > 15

A binary image is produced, assigning a value of 1 to any pixel that satisfies the threshold above and 0 otherwise. The image below shows both the face detected and the skin segmentation method applied on the face image. 

[Al-Tairi et al.'s skin segmentation]()

We used region shape properties of binary image, using the Matlab regionporps function, to help filter out non-face detection (false positive). We utilised an area (A) with threshold at 2200 to discriminate the detected objects with smaller area. The Euler number (E) was also used to reject regions which have no holes under the assumption that a face region will contain at least one hole. Circularity metric (C) was employed as well – calculated as 4*pi*Area/Perimeter^2. These properties were not helpful removing all false positives due that some false detection has bigger area than the detected faces (true positive) or even similar values on other properties compared with a face. For example, the ratio aspect on the image above is similar to the image below (false positive) whereas the area is much bigger (A: 46752) on the false positive image below. It was difficult to choose the right threshold based on area length or other shape properties. 

Another approach was to train a SVM classifier that filter out non-faces based on region shape properties (as features). A database of shape properties was created. This approach did not resolve the problem of false negative either. 

Several images were passed to our classifier to evaluate the recognition accuracy. Image of group of students and lectures in this module were used. Those images were taken on lecture and lab rooms. In addition, we also made use of our own faces and non-faces images. 


The image above displays the best result from our function using Frontal Face CART method. All 23 faces on the image were detected however 2 were misclassified and the remained 21 faces were classified correctly. The misclassified faces are marked with a red and blue rounded marker; the blue one represents a face that was not included in our database. The recognition accuracy was 91%.  However the face detection, using Frontal Face LBP, detected 22 faces (false negative detection marked with yellow square in image below). The recognition accuracy was 91% as well but the misclassification was on different face, highlighting how the selection of detection methods can impact on recognition accuracy.

In the image below, the face recognition accuracy was 50%. Notice that the face misclassified (as 17) in the image below was correctly classified on the image above (as 13). It is due that the scale of both faces are different and HOG descriptor is not good at variation in scale ( in relation with camera position).

The image below, one of my own images, was classified incorrectly. This image had similar background to images on the database but illumination was different. HOG descriptor cannot capture changes in illumination.

## Conclusion

Face recognition is not an easy task. It requires the application of many techniques for detecting and classifying faces. We have done a number of experiments combining supervised machine learning algorithms and features extraction methods. A Cross validation technique was employed to choose the best classifier. SVM with HOG showed the best performance and was used for the recognition task.  In general SVM classifier proved to work well as a face classifier.  

The major drawback of our approach was the methods used to filter out non-faces (false positive). These approaches were not effective for our database sample. To improve accuracy, we could train our own face detector for faces in our class using cascade classification model. Another better approach is the use of deep learning techniques, such as Deep NN or Convolutional NN, to add multilayer levels of abstraction for our face images database. It will capture the most significant features and improve performance.


