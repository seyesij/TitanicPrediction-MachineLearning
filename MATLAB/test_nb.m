function test_nb(X_train, y_train, X_test, y_test)

% NB
nb = fitcnb(X_train, y_train)
% predict on the test set
y_predict = predict( nb, X_test);

% running classier performance
cp = classperf(y_test{:,'survived'}');
classperf(cp, y_predict);
% this is the accurace of our model
disp ("NB Test accuracy:")
cp.CorrectRate


end