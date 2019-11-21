function test_tree(X_train, y_train, X_test, y_test)
% TREES
tree = fitctree(X_train, y_train, 'MaxNumSplits', 15);
% predict on the test set
y_predict = predict( tree, X_test);

% running classier performance
cp = classperf(y_test{:,'survived'}');
classperf(cp, y_predict);
% this is the accurace of our model
disp ("Tree Test accuracy:")
cp.CorrectRate

%view(tree, 'Mode', 'graph')

end