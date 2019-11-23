format shortg

% read entire data
X = readtable("X.csv");
y = readtable("y.csv");

% read train data
X_train = readtable("X_train.csv");
y_train = readtable("y_train.csv");
%X_train.size()
%y_train.size()

% read test data
X_test = readtable("X_test.csv");
y_test = readtable("y_test.csv");


%train and test Logistic Regression model
%test_lg(X_train, y_train,X_test,y_test)

%train and test Random Forest model
rf = test_rf(X_train, y_train,X_test,y_test)


