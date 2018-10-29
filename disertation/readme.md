# Abstract 
This work investigates the effects of varying geographic scale in the accuracy of variable selection using regularised regression 
methods. This work is motivated by the gaps in the research about the impact of applying regularisation on data that are dependent 
of geographic scale. The goal of this research is to explore, compare and inform pattern in the regularized methods (Lasso, Ridge, 
Lasso-Lar and Elastic Net) by adding the geographical scale as a variable. GWR is applied to assess non-stationary relationships. 
Regularised Random Forest is also explored. The main contribution of this work is a detailed analysis of effect in the variable 
selection process as spatial scale change using two case studies. The findings suggest that Lasso-Lar should be preferred in 
presence high multicollinearity and higher resolution less collinearity but more complex model.

# Introduction
Regression analysis, particularly linear regression, is a very popular machine learning technique which permits to investigate the 
relationship between predictor variables and outcome. The selection of the predictor variables that best describes a phenomenon is 
an essential step in the model building process [9]. Linear regression is widely applied to a variety of fields such as social, 
health and scientific [6, 7 and 9] due its simplicity and easy interpretability of the model. However, linear models are sensitive 
to multicollinearity (or collinearity). Collinearity, multiple variables giving the same information to explain the outcome, 
increases the variance of regression coefficients making the model unstable. Multiollinearity causes the coefficients to have a 
large size, consequently small changes in predictors cause large changes in the response. That is, an inadequate estimate of the 
impact of changes on predictors. In addition, the model interpretation became difficult in presence of large number of variables, 
thus many parameters to estimate. However, deciding properly which variables should be in the model is a complex task, even more in
the presence of highly correlated variables.

Regularised regression techniques were introduced to overcome multicolinearity and reduce model complexity with relation to the 
number of predictors in the model. The regularised methods imposed a penalty on - regularise - the size of the estimated regression
coefficients of the model which reduces the variance and provides better explanation for the model [10, 13, 14]. Ridge regression
[12], Lasso (Least Absolute Shrinkage and Selection Operator) [13], Elastic Net [67] and LAR (Least Angle Regression) [14] are 
the most popular regularised techniques. Although these techniques have been widely applied and researched, there are very few 
studies about the effect of the regularisation on data that are dependent of geographic scale which has inspired this research.

It is known that patterns as well as the correlation between factors vary with spatial scale when modelling geographical 
phenomena. However it is not always taken into consideration [1]. The study of spatial dependence of the model's variables is 
essential to determine how the relationship among those variables varies across space. This helps uncovering distinctive local 
variation of the variables at different geographic scale as well as estimating the model's parameters locally. The linear 
regularised techniques built global models, that is, the regression coefficients are estimated for the whole dataset and assumed 
to be constant across of space (i.e. stationary); being unable to detect more precise local spatial variations in the data. There
are several approaches to model local spatial variation, being the Geographically Weighted Regression (GWR) [2] very popular. GWR
is a visual analytic technique [2] that is able to describe the spatial dependence of explanatory variables by exploring the 
variations of each explanatory variable at each location. GWR uses spatial weights and assign higher weight values nearest points
, marking them as more relevant than the distant points. GWR is an efficient method for detecting local variations; however it is
unable to detect collinearity between predictors that are highly inter-correlated [3 and 4]. It also lacks of diagnostic tools to
evaluate the goodness of fit of the model [4]. GWR ridge and GWR lasso methods have been proposed to solve collinearity problems
in GWR [4, and 32] but these methods are computational expensive for models with large number of variables due that GWR builds 
several models for each variable at different locations.

The relationship between response and predictors is sometimes very complex to be modelled linearly. In this case, an alternative
analytic technique (such as Random Forest) should be explored. In addition, linear regression is a parametric method that 
estimates a number of unknown parameters from the data which may be computational expensive for models with large number of 
variables. Random Forest (RF), introduced by Breiman [23], is very popular non-parametric technique which imposes few 
presumptions on the regression function. Beside, RF provides metrics for variable importance that can be used for variable 
selection. Nevertheless, RF does not have automatic way to choose a subset of variables. It can be problematic for a large number
of variables. Several approaches have been developed to help the variable selection in RF [28, 31, 58]. However, studies have 
observed poor results in presence of variables highly correlated [58]. Recently, Regularised Random Forest (RRF) was propounded 
by [68] to aid the variable selection process in RF. RRF applies a penalty to the variable importance metric of RF to choose 
small number of variables. The Guided RRF (GRRF) [69] is an improvement of RRF which adds a new penalty coefficient and, unlike 
RRF, applies individual penalty value to variable importance score of each variable. There are very few researches about RRF and
GRRF. Therefore, this gap motivates this investigation about the behaviour of variable selection by RRF in regression problems as
spatial scale varies.

The purpose of this research is a detailed study of socio-demographic patterns at different geographical scales as well as the 
evaluation of different regularised regression techniques for variable selection in a geographic context. This analysis will be 
useful to find potential explanations of the factors associated with geographic variations in socio-demographic variables and the
influence of varying spatial scale in the variable selection process. This will be accomplished by exploring how the selection of
the penalty parameter together with the geographical scale impact on the result of the regularised regression techniques and 
their ability to describe the association between predictors and response. The comparison and evaluation will be focused on the 
number of variables and regression coefficient estimates in the final model using two case studies - percent of unemployment and 
crime rate - to find out which census variables are the best to explain the percent of unemployment and crime rate in each region
of study. GWR is applied to the best global obtained from regularized techniques to investigate the local variability on the 
global model at each geographical level. In addition, RRF and GRRF are utilised to explore the potentially existence of more 
complex variable associations at each region of study, focusing on the setting of the regularisation coefficients.

The research question that this study intents to answer is: ***What are the effects of varying geographic scale in the accuracy of 
variable selection using regularised regression methods?*** 
This research has the followed objectives:
1. To explore and compare linear regularised regression methods in geographical context at different spatial scales.
2. To evaluate the significance of the resulted models using appropriate statistical methods.
3. To explore the behaviour of variable selection in Regularised Random Forests as geographical scale varies.

The beneficiaries of this work are researchers (e.g. sociologist) who want getting a better understanding of relationships among
socio-demographic indicators at different geographic levels, and broaden the knowledge of the effects on performance of 
regularised regression methods as spatial scale varies. The results obtained from our experiments will be analysed in concordance
with the approaches assumptions and parameter characteristics. The analysis will be well documented through graphics, maps and 
reports describing the geographical patterns of coefficients estimates and tuning parameters as space vary.

This study is organized as follows: chapter 2 presents the literature review on the research of relevant publications about 
variable selection techniques and methods for modelling spatial variation. The research gap detected in this section gives the 
motivation and framework of this work. Chapter 3 describes in details the methods used in this study: the chosen linear 
regularised methods for variables selection, GWR and Regularised Random Forest thecniques. The procedure of the analysis is also
described. The description is focused on the computation of estimated regression coefficients as well as variable importance 
score and the penalty parameters. The chapter 4 is dedicated to present the result of the analysis and investigate the 
performance of the regularized regression methods used in this research based on two case studies. In addition, the GWR is 
applied to the best linear regression model and the results are mapped. RRF is also analysed to assess their capacity and compare
with linear regularized techniques. Chapter 5 & 6, are dedicated to compare results and summaries findings in the analysis in 
pervious sections and evaluation and conclusion of this work.

# Literature Review
Social researchers have been concerned with the decrease in survey response over years [5, 6, 7and 8]. Different approaches, 
such as auxiliary data, topic interest and paradata, have been used to investigate this phenomena but the finding had no 
significant relevance to explain the causes [7, 8]. Exploring the use of statistics derivate from others data sources could 
bring benefits [7]. Governments and administrations have made extensive datasets publically available which might offer a 
potential value for social researchers in using a new range of variables to study the factors of the disposition to participate
in surveys.

However these datasets have different resolutions and variability, and involve a large number of variables. Looking for patterns
and suitable auxiliary variables in these data sets would be challenging and uncertain. Having a comprehension, in advance, of 
the structure, relationships and distribution of these data would facilitate the work in selecting quality auxiliary variables. 
One of the focuses of this proposal is to investigate the geographical variation in patterns of socio-demographic factors which 
would bring insight of potential auxiliary variables for social researches. More researches are needed in this matter applying 
modern techniques (e.g. machine learning methods). To the best of my knowledge, there are not similar works in the literature.
Linear regression model is extensively used by researchers to understand how the changes on one or more predictors (independent 
variables) affect the outcome variable (dependent variable) – i.e. to describe a linear dependence between predictors and 
response variables. An important step in model building is the selection of the model's variables, mainly when a large number of
variables are available [9] to make model more stable and interpretable. There are several techniques that can be used for 
variable subset such as leaps and bounds algorithm for best subset and Forward-and-Backward-Stepwise [10]. However choosing a 
subset could introduce high variability in the regression errors of the model; other alternatives can be used [10, 11] such as 
regularised regression techniques e.g. ridge regression [12], Least absolute shrinkage and selection operator (Lasso) [13], 
Elastic Net [67] and Least angle regression (LAR) [14]. The regularised methods imposed a penalty on (regularise) the size of 
the estimated regression coefficients of the model which allows reducing the variance and providing a better model explanation 
[10, 13 and14]. However, bias is introduced in the model by the variance reduction procedure [10].

Ridge regression performs L2 penalty to shrink the coefficients toward zero, for large penalty value. All predictors are kept 
in the model – i.e. irrelevant predictors are not removed from the final model but their estimated coefficients are reduced. 
Whereas, Lasso uses L1 penalty reducing the value of coefficients of insignificant predictors exactly to zero for large penalty 
value as a result few nonzero coefficients stay in the final model, that is, Lasso produces a sparse models. Consequently, Lasso
performs coefficient penalisation as well as variable selection. However, Lasso has not a unique solution because of it is 
quadratic problem. In addition to this, Lasso, unlikely to Ridge, performances poorly in presence of highly correlated variables.
Lasso chooses randomly only one predictor among the correlated variables. Therefore, several alternatives, such as Elastic Net 
and LAR, have been developed to overcome Lasso issues. Elastic Net can be seen as generalization of Lasso which combines both L1
and L2 regularisation producing sparse models and small (in magnitude) coefficients. In presence of several highly correlated 
variables, Elastic Net manages better the correlated variables than Lasso. Elastic Net keeps or removes correlated predictors 
together from the model. LAR is a computational efficient alternative of Lasso which can compute the full path of lasso for 
variable selection. LAR adds iteratively variables to the model, moving the estimated coefficient towards its least-square 
coefficient until another predictor more correlated to the current residual is found [10, 14]. The number of steps of LAR 
algorithm for the lasso solution is seen as the tuning parameter. A key and hard task is to find the optimal value of the penalty
parameter. The Akaike Information Criterion (AIC) and Cross-Validation (CV) are two popular and well-established tools for 
tuning the penalty parameter [10]. AIC measures the balance between model fit and model complexity [35]. Whereas, CV chooses the
model with the best prediction performance i.e. the model that minimises of the Mean Squared Error (MSE) [33].

These regularised regression techniques have been extensively researched and applied to a variety of problems. Nevertheless, only
few studies are concentrated on geographic context at different spatial scale [15, 16 and 17]. Therefore, there is a need for 
more research on regularised regression methods focused on varying the geographical scale. This lack of studies has motivated 
this investigation about the effects on performance of regularised regression methods as spatial scale varies.

The regularised regression methods mentioned above built global models i.e. the estimated coefficients are assumed to be 
unchanged (stationary) across the space, that is, they are independent of the locations. A model is fitted from the data and 
the relationships between response and predictors are fixed on whole area of study. Global summary statistics have been commonly
employed for investigating human behaviours in social science [18]. However, a global approach is useful to show patterns in the
entire dataset being unable to show the variability in different parts of the dataset [19]. The study of human behaviour in 
geographic context at different scales is very important for social researchers [20]. Analysing data with geographic location at
local level can show similarity and/or differences in patterns across the space which allows discovering spatial dynamics at 
different scales [21 and 22].

Local weighted regression techniques were originally proposed by Cleveland and Devlin [21] which provides the technical origin 
of the Geographically Weighted Regression (GWR) [2]. GWR is a popular exploratory method for modelling the variation of 
relationships (nonstationary association) over space [1, 2, 37and 39].

The extensively use of GWR is due its ability to discover complex pattern across space [1, 2] which estimates the parameters 
locally throughout the data surface (i.e. the regression model changes over space) capturing non-stationary patterns. GWR fit a 
model (producing a set of regression parameter estimates) at each point in the space, creating neighbourhoods around the 
points in which the point nearest to the centre of the neighbourhood receive larger weight than those further away. GWR allows 
the relationship between variables to vary across the region of study, i.e. the coefficient estimates vary spatially 
(over space).

As a regression modelling technique, GWR is affected by multicollinearity, having a greater effect on the coefficient estimates.
This produce unstable models and dependence in the local regression coefficient. Wheeler and Tiefelsdorf [3] were the first that
demonstrated the local collinearity affecting GWR. It was asserted later in more in-depth studies [3, 4, 39, 40, 41, 42, 43, 44 
and 46]. The local collinearity in GWR can be even more severe for sample of small size which is utilized to estimate the local 
parameters [39]. In the presence of collinearity, the GWR may also find untrue patterns in the coefficients [3 and 39]. The lack
of diagnostic tools to evaluate the goodness of fit of the model is another drawback of GWR [3 and 4]. Two penalised versions of
GWR, GWR ridge [42 and 44] and GWR lasso [43], has been proposed to solve collinearity problems in GWR models [42 and 43]. These
penalised methods added a constraint on the magnitude of the estimated regression coefficients providing more accurate local 
coefficient estimates. The GWR lasso also provides sparse models at some locations of the study area. Locally-compensed ridge 
GWR was proposed as alternative of GWR ridge [51]. Locally-compensed ridge GWR, unlikely GWR ridge, allows the ridge parameter 
to change over the study area and the penalty is only applied at points with ill-collinearity. Consequently, the local 
estimation is based on a given local condition number threshold. However, these regularised GWR are computationally expensive 
being impracticable for dataset with large number of variables.

Regularised linear regression and GWR are unable to capture complex (e.g. non-linear) relationships between variables. Therefore
alternative techniques, such as Random Forest (RF), should be employed to model more complex relationship. RF, introduced by 
Breiman [23], is an ensemble of multiple decision trees which are trained on different parts of the dataset and subset of 
variables. The algorithm uses bootstrap sampling to grow the decision trees and randomly choose a set of variables for the 
partitions in a tree [53]. The trees in the forest are unpruned to decrease bias. The split at each node of the tree is based on
randomly selected a subspace from entire variables space. RF can be used for classification and regression. In regression (our 
interest), the average of the multiple decision trees outcome is the output of RF; the tree split is based on the prediction 
squared error [10 and 23]. The aggregation of individual trees allows decreasing the correlation between the trees which reduce 
variance. RF technique has become very popular due its ability to control over-fitting, predictive accuracy and variable 
importance metrics [10, 24, 25, 26, 27 and 54]. RF also handles well problems with small sample size and large number of 
variable [54].

Random Forest provides a variable importance metric to assess the contribution of the variables in the model as well as the 
interaction between more than one variable together [52]. The most popular variable importance metrics are permutation importance
and the Gini importance. The permutation importance, the focus of this study, is based on the permutation of out-of-bag (OOB) 
sample. In the permutation importance metric, a variable is permuted randomly on each tree and the OOB error (the OOB, an 
important characteristic of RF, estimates the accuracy error by using an internal test set) is computed for the variable before 
and after the permutation. Then the difference between both OOB errors is averaged over all trees. The OOB error is also used to
estimate error rate and control correlation.

The variable importance score in RF is frequently utilised as tool for variable selection however, it does not have an automatic
mechanism to choose the best subset of variables that describe effectively the model being analysed [32]. The permutation 
importance metric, in particular, has shown good performance for guiding the variable selection algorithms nevertheless, several
studies have demonstrated its sensitive to variable highly correlated [55, 56, 57, 58 and 59]. Several algorithms have been 
proposed for variable selection based on importance metric in RF. A naïve significant test of normality of standard score was 
employed on [60] however the study in [52] demonstrated the lack of statistical rigour of this approach. A backward elimination 
algorithm was also introduced for variable selection using the variable importance metric provided by RF [28]. This algorithm 
removes the least important variables in every iteration until OOB error drops [62]. Nevertheless, this approach is sensitive to
multicollinearity as well [55]. Conditional permutation importance was propounded to regulate correlation among variables [58] 
which create a conditional permutation grid based on the partition of the variables space made by RF.

The Regularized Random Forest (RRF) has been introduced recently [66]. This approach, inspired by Lasso, penalises the predictors
that have similar importance score to those variables previously selected in all trees. RRF apply a penalty parameter to the 
ordinary importance score of RF only if the variable is already in the subset of variables created in previous trees. RRF built 
only one ensemble wherewith the variables are evaluated on a portion of the data on each node of the tree. In nodes with small 
number of instances may lead RRF to choose irrelevant variables. The Guided Regularized Random Forest (GRRF) [69] is an 
improvement of RRF. GRRF computes a guided coefficient from the importance score of an original Random Forest. This guided 
coefficient is used to supervise the variable selection process in RRF which allows removing the irrelevant variables. GRRF 
assesses variables locally at each node and the variables selection in the current tree take in consideration the subset of 
variables chosen in previous trees. The penalty coefficient and guided coefficient are the regularised parameters in those 
algorithms. A deeper understanding about variable importance in RF is still need [24, 25, 26, 27 and 28]. The literature 
available is mainly focused on discussion of variable importance for classification problems [24, 28 and 29]. Very few 
researches are focused on the use of the newly regularised approach for Random Forest. This void has motivated our exploration 
of variable importance metric by RF in the regularization context as varying spatial scale.
