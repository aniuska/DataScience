# Binary Classification

## The problem

* Predicting student performance (i.e. passing/failing) based on information about students.
* Previous studies (Kotsiantis et al. 2004) show that student achievement is highly affected by previous performances.
* Although such scores were available we decided to exclude them to try and investigate other predictors (demographic, 
social/emotional and school related).

## Analysis of Dataset
* Student data from 2005-2008 for 2 public secondary schools from Alentejo region of Portugal derived from schools reports and 
questionnaires.
* Two datasets were used: Portuguese language (549 observations) and Maths (339 observations). They were analysed independently.
* 29 out of 36 discrete features used (16 categorical and 13 numeric). The labels were converted from discrete scale 0-20 to a 
binary Pass/Fail scale.
* A basic statistical summary and histogram plots created to determine probabilistic distribution of data.
* Zero p value indicates dependence amongst features.

## Hypothesis
* How well can student performance be predicted using demographic, social/emotional and school related features?
* What are the factors that affect student performance?

## Summary of ML models
Naïve Bayes (NB) was used to create the classifier. NB is a supervised and probabilistic model based on Bayes rule that
assumes, naively, that the features are conditional independent of one another given the class. NB classifier assigns observations to the most probable value of the class (maximum posterior decision rule) .

* **NB Pros**: Simple to train; efficient when conditional independence assumption is correct; learn well with few training examples and effective for datasets containing many features. Perform very well even if NB assumption (not always true) does not hold.
* **NB Cons**: High bias because of strong assump

## Training and evaluation methodology
A 10-fold cross validation (90% training/10% test) was utilised. Several rounds were run for each model. Accuracy and RMSE were used for comparing models' performance and generalisation. ROC curves used to inspect classifier performance more closely (graph below shows the ROC curve of the highest AUC on each model).

## Choice of Parameters and Experimental Results

Naïve Bayes was produced using fitcnb Matlab function. Priors were calculated as relative frequency distribution of the classes in the data set. Logarithmic transformations to estimate posteriors were attempted but didn't improve results. Multivariate multinominal distribution used for categorical and normal/density smoothers for numeric. Smoother were used as histograms show
skewed curves. 
* Portuguese: same accuracy result obtained for normal distribution and normal smoother. RMSE was slightly lower
when using the normal smoother.
* Maths: the best result was obtained for the normal assumption.

## Analysis and Critical Evaluation

Best accuracy was achieved by DT on Portuguese data set but NB
outperformed on Maths. Both DT and NB produce accuracy scores
comparable to original paper by Cortez and Silva. However, such is the
class imbalance between pass and failure, F1 score is perhaps a better
indicator of the robustness of DT model. These scores are much lower
indicating the difficulty in predicting failure.

The result of NB classifiers shows that the learning process for these  data sets is better when the prediction (posterior) is determined by a mixture of the prior knowledge and the evidence provided by the data. NB accuracy was very good despite that features were not independent among them given the classes.

ROC curve show the class prediction for each observation on NB. In general all models perform consistently and the trade-offs between TP and FP (costs) is good. However the TPRs for some observations were low on two models for Math data set indicating some randomly guessing.

## Conclusions and Future Work

Two different cross-validation techniques were used. Accuracy metric, F1 score and ROC curve were used for
getting better insight from algorithm performance. NB demonstrates to be good-enough for this application
despite features dependence and gives the best result for the smaller data set (Math). DT shown to be good at
knowledge representation. NB learns incrementally and DT gives better insight about underlying data.

Extension of models beyond a simple binary pass/fail classifier to incorporate multiple classes (e.g. grades A to F)
and regression analysis. Investigation of other ML algorithms such as Random Forests and ensemble methods.

## References

1 Paulo Cortez and Alice Silva, Using Data Mining to Predict Secondary School Student Performance.
2 Kotsiantis S.; Pierrakeas C.; and Pintelas P., 2004. Pre-dicting Students’ Performance in Distance Learning Using Machine Learning Techniques. Applied Artificial Intelligence (AAI), 18, no. 5, 411–426.
3 Kevin Murphy, Machine Learning A Probablistic Model P546-552, MIT 2012.
