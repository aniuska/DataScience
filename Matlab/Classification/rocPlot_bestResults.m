clear;=
%% Plot ROC curves for best test reslts on cross validation for 5 Naive Bayes experimental models
% Five Roc curve
%1- default by system - distributions: 'mvmn' and 'normal'
%2- distributions: 'mvmn' and kernel smoother = 'normal', prior='empirical', ScoreTransform ='logit' 
%3- distributions: 'mvmn' and kernel smoother = 'normal' and prior='empirical'
%4- distributions: 'mvmn' and kernel smoother = 'epanechnikov' and prior='empirical'
%5- distributions: 'mvmn' and kernel smoother = 'normal' and prior='uniform'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
roc = load('rocPor_test.mat');
%roc = load('rocLowerMath_test.mat');
K=10;
%Plot the ROC curves on the same graph.
figure;
pmarker = {'o','*','x','s','+','d','v','p','.','h'};
%pmarker = {'o','*','x','s','+'};
modelsVectorName={'model1_X','model1_Y','model2_X','model2_Y','model3_X','model3_Y','model4_X','model4_Y',...
                  'model5_X','model5_Y','mmodel1_X','mmodel1_Y','mmodel2_X','mmodel2_Y','mmodel3_X','mmodel3_Y',...
                  'mmodel4_X','mmodel4_Y','mmodel5_X','mmodel5_Y'};
c=1;
figure.PaperSize=[10 10];
figure.PaperUnits ='centimeters';
subplot(2,2,1);
for k=1:2:10 
  pl = plot(roc.(modelsVectorName{k}),roc.(modelsVectorName{k+1}));
  pl.Marker=pmarker{c};
  lv{c}=strcat('model ',int2str(c));
  c=c+1;
  hold on;
end
legend(lv);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC curves-Naive Bayes Classification on Test set (Portuguese)');
%print('roc_Por','-dpng','-r72');
%hold off
%ROC curve for Maths
a=1;
figure;
subplot(2,2,3);
for k=11:2:20 
  pl = plot(roc.(modelsVectorName{k}),roc.(modelsVectorName{k+1}));
  pl.Marker=pmarker{c};
  lv{a}=strcat('model ',int2str(a));
  c=c+1;
  a=a+1;
  hold on;
end
legend(lv);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC curves-Naive Bayes Classification on Test set (Maths)');
print('roc_all','-dpng','-r72');
%print('roc_lowMat','-dpng','-r72');
hold off
