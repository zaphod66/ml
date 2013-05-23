clear ; close all

% load examples
load('ex3data1.mat');
m = size(X, 1);

% load nn-weights
load('ex3weights.mat');

Xn = [ones(m,1), X];
A1 = sigmoid(Xn * Theta1');
An = [ones(m,1), A1];
A2 = sigmoid(An * Theta2');

[mx, p] = max(A2, [], 2);

acc = sum(p == y) * 100 / m;
fprintf('\nTraining Set Accuracy: %f\n', acc);