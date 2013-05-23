function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


sum1 = 0;
for i = 1:m
	hypo = sigmoid(X(i,:) * theta);
    t1 = y(i)     * log(    hypo);
    t2 = (1-y(i)) * log(1 - hypo);
    sum1 = sum1 - t1 - t2;

    for j = 1:size(theta)
        grad(j) = grad(j) + ((hypo - y(i))*X(i,j));
    endfor
endfor
sum1 = sum1 / m;
sum2 = lambda * (sumsq(theta) - theta(1)^2) / (2 * m);

J    = sum1 + sum2;
grad = grad / m;

for j = 2:size(theta)
	grad(j) = grad(j) + lambda * theta(j) / m;
endfor

% =============================================================

end
