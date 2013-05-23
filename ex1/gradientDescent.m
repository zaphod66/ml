function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % t1 = (X*theta-y).*X(:,1)
    % t2 = (X*theta-y).*X(:,2)
    % u1 = alpha * sum(t1) / m
    % u2 = alpha * sum(t2) / m
    % tu = [u1;u2]
    % theta = theta - tu
    
    tl = length(theta);
    ts = zeros(tl,1);
    tu = zeros(tl,1);

    for i=1:tl
        ts(i) = sum((X*theta-y).*X(:,i));
    endfor

    tu = alpha * ts / m;
    
    theta = theta - tu;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
