% RANDOM FOREST
function Mdl = test_rf(X_train, y_train, X_test, y_test)

rf_bag = fitrensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 100');
rf = fitrensemble(X_train, y_train, 'Method', 'LSBoost', 'NumLearningCycles', 100','LearnRate', 0.1);

% predict on the test set
y_predict_train = predict( rf, X_train);
y_predict_train = round(y_predict_train);
% running classier performance
cp_train = classperf(y_train{:,'survived'}');
classperf(cp_train, y_predict_train);
% this is the accurace of our model
fprintf ("RF BOOST Train accuracy: %0.3f \n", round(cp_train.CorrectRate,2))


rf_cv = fitrensemble(X_train, y_train, 'Method', 'LSBoost', 'NumLearningCycles', 100,'LearnRate', 0.1, ...
    'CrossVal', 'on', 'KFold', 10);

cv_loss = kfoldLoss(rf_cv,'Mode', 'cumulative');
figure;
plot(cv_loss);
ylabel("Cross validation loss")
xlabel("Learning cycle")
title("Cross validation loss depends on number of cycles")

fprintf("max cross validation accuracy %0.3f \n", 1 - min(cv_loss));
%================================================================================

% predict on the test set
y_predict_train = predict( rf, X_train);
y_predict_train = round(y_predict_train);
% running classier performance
cp_train = classperf(y_train{:,'survived'}');
classperf(cp_train, y_predict_train);
% this is the accurace of our model
fprintf ("RF BOOST  Train accuracy: %0.2f \n", round(cp_train.CorrectRate,2))
%-------------------------------------------------------
% predict on the test set
y_predict = predict( rf, X_test);
y_predict = round(y_predict);
% running classier performance
cp = classperf(y_test{:,'survived'}');
classperf(cp, y_predict);
% this is the accurace of our model
fprintf ("RF BOOST Test accuracy: %0.3f \n", round(cp.CorrectRate,2))
%======================================================================

Mdl = rf;
rf = rf_bag;  

impOob = oobPermutedPredictorImportance(rf)
% get the feature importance
[impGain, pred] = predictorImportance(rf)
% start new figure
figure
% plot importance gains for each feature
plot(1:numel(rf.PredictorNames),[impOob' impGain'],['b']);
% beautify the chart
title('Predictor importance');
xlabel('Predictor name');
ylabel('importance');
legend('MSE improvement');
% the x axis labels and rotation
h = gca;
h.XTick = 1:numel(rf.PredictorNames);
h.XTickLabel = rf.PredictorNames;
h.TickLabelInterpreter = 'none';
h.XTickLabelRotation =45;
grid on


%OPTIMIUSATION OF HYPER PARAMETERS
optimise = 1;
if optimise == 1
 
 tree = templateTree('Reproducible', true');
 rf_opt = fitrensemble(X_train, y_train, 'OptimizeHyperparameters','auto','Learners', tree ,...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus'));

 % predict on the test set
 y_predict_opt = predict( rf, X_test);
 y_predict_opt = round(y_predict_opt);
 % running classier performance
 cp_opt = classperf(y_test{:,'survived'}');
 classperf(cp_opt, y_predict_opt);
 % this is the accurace of our model
 fprintf ("RF Optimised accuracy: %0.3f \n", round(cp_opt.CorrectRate,2))
end



end