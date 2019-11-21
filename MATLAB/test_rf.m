% RANDOM FOREST
function test_rf(X_train, y_train, X_test, y_test)


rf = fitrensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 200);

% predict on the test set
y_predict_train = predict( rf, X_train);
y_predict_train = round(y_predict_train);
% running classier performance
cp_train = classperf(y_train{:,'survived'}');
classperf(cp_train, y_predict_train);
% this is the accurace of our model
fprintf ("RF Train accuracy: %d \n", round(cp_train.CorrectRate,2))
%-------------------------------------------------------
% predict on the test set
y_predict = predict( rf, X_test);
y_predict = round(y_predict);
% running classier performance
cp = classperf(y_test{:,'survived'}');
classperf(cp, y_predict);
% this is the accurace of our model
fprintf ("RF Test accuracy: %d \n", round(cp.CorrectRate,2))
%======================================================================


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


end