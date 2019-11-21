function test_knn()

% KNN
knn = fitcknn(X_train, y_train,'NumNeighbors', 2);
% predict on the test set
y_predict = predict( knn, X_test{:,:});

% running classier performance
cp = classperf(y_test{:,'survived'}');
classperf(cp, y_predict);
% this is the accurace of our model
disp ("KNN Test accuracy:")
cp.CorrectRate


end