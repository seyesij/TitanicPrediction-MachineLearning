
function test_lg(X_train, y_train, X_test, y_test)

% Logistic Regression

% Logistic Regression Cross Validation
lg_cv = fitclinear(X_train{:,:}, y_train{:, 'survived'},'Learner','logistic','Regularization','ridge', 'KFold', 5);
% predict on the test set
class_error = kfoldLoss(lg_cv)
accuracy = 1 - class_error

% this is the accurace of our model
fprintf ("Log Reg cross Validation accuracy: %0.2f\n", round(accuracy,2))

%
lg = fitclinear(X_train{:,:}, y_train{:, 'survived'},...
    'Learner','logistic','Regularization','ridge','Solver','bfgs');
% predict on the train set
y_predict_train = predict( lg, X_train{:,:});
% running classier performance
cp_train = classperf(y_train{:,'survived'}');
classperf(cp_train, y_predict_train);
% this is the accurace of our model
fprintf ("Log Reg Train accuracy: %0.2f \n", cp_train.CorrectRate)
%----------------------------------------------------------------
% predict on the test set
y_predict = predict( lg, X_test{:,:});
% running classier performance
cp = classperf(y_test{:,'survived'}');
classperf(cp, y_predict);
% this is the accurace of our model
fprintf ("Log Reg Test accuracy: %0.2f \n", cp.CorrectRate)
%===================================================================


%Logistic Regression optimization
rng default
lg_opt = fitclinear(X_train{:,:}, y_train{:, 'survived'},'ObservationsIn', 'rows', ...
        'OptimizeHyperparameters',{'Lambda','Regularization'},...
        'HyperparameterOptimizationOptions', ...
        struct('AcquisitionFunctionName','expected-improvement-plus'))

    
cp = classperf(y_test{:,'survived'}');
classperf(cp, y_predict);
% this is the accurace of our model
fprintf ("Log Reg Hyper Params Optimized  accuracy:%0.2f \n",round(cp.CorrectRate,2))

end
