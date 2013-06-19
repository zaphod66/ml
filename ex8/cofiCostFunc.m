function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% fprintf('==========================\n');

% fprintf('size(X)     = (%d,%d)\n', size(X,1), size(X,2));
% fprintf('size(Theta) = (%d,%d)\n', size(Theta,1), size(Theta,2));
% fprintf('size(Y)     = (%d,%d)\n', size(Y,1), size(Y,2));
% fprintf('size(R)     = (%d,%d)\n', size(R,1), size(R,2));
% fprintf('num_movies  = %d\n', num_movies);
% fprintf('num_users   = %d\n', num_users);

mY = ( Theta * X' )';
mJ = ( (mY - Y) .* R ) .^ 2;

J = 0.5 * sum( sum( mJ ) ) + (lambda / 2) * ( sum(sumsq(Theta)) + sum(sumsq(X)) );

for i=1:num_movies
    idx = find( R(i, :) == 1 );
    Theta_temp = Theta(idx, :);
    Y_temp     = Y(i, idx);

    X_grad(i,:) = ( X(i,:) * Theta_temp' - Y_temp ) * Theta_temp;
endfor

for j=1:num_users
	theta_sum = 0;
	t_j = Theta(j,:);
    for i=1:num_movies
    %	printf('(i,j) = (%d,%d)\n', i, j);
        x_i = X(i,:);
        tmp = ( ( ( t_j * x_i' ) - Y(i,j) ) * R(i,j) ) * x_i;
        theta_sum = theta_sum + tmp;
    endfor
    Theta_grad(j,:) = theta_sum;
endfor

X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda * Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
