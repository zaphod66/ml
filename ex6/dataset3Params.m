function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%minerr = 100;
%minC   = 1;
%minSig = 1;
%for Ci=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
%    for Si=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
%        fprintf('Train with C=%f and sigma=%f', Ci, Si);
%        model = svmTrain(X, y, Ci, @(x1, x2) gaussianKernel(x1, x2, Si));
%        preds = svmPredict(model, Xval);
%        error = mean(double(preds ~= yval))
%        if (error < minerr)
%            minerr = error;
%            minC   = Ci;
%            minSig = Si;
%        endif
%    endfor
%endfor
%
%fprintf('minerr = %f, minC = %f, minSig = %f', minerr, minC, minSig);
%
%C     = minC;
%sigma = minSig;

% minerr = 0.030000, minC = 1.000000, minSig = 0.100000

C     = 1.0;
sigma = 0.1;

% =========================================================================

end
