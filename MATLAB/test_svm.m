
function test_svm(X_train, y_train, X_test, y_test)

% SVM
% KernelFunction : rbf, linear
% fitting SVM
svm = fitcsvm(X_train, y_train, 'KernelFunction', 'linear');
%,'OptimizeHyperparameters','auto')

% predict on the test set
y_predict = predict( svm, X_test);

% running classier performance
cp = classperf(y_test{:,'survived'}');
classperf(cp, y_predict);
% this is the accurace of our model
disp ("SVM Test accuracy:")
cp.CorrectRate

end 

