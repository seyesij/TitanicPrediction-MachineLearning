indices = crossvalind('kfold', y{:,'survived'}', 5) ;

%cp = classperf(y{:, 'supervised'}')
AvgAccuracy = 0;
for i=1:5
    test = (indices == i);
    train = ~test;
    
    sum(test);
    sum(train);
    
    svm = fitcsvm(X(train,:), y(train,:));
    y_predict = predict(svm, X(test, :));
    y_test = y(test, :);
    cp = classperf(y_test{:,'survived'}');
    classperf(cp, y_predict);

    AvgAccuracy = AvgAccuracy+ cp.CorrectRate/5;
    
end
fprintf("cross valid accuracy %f", AvgAccuracy)