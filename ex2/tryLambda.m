data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
Xm = mapFeature(X(:,1), X(:,2));
initial_theta = zeros(size(Xm, 2), 1);
lambda = 1;
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, Xm, y, lambda)), initial_theta, options);

plotDecisionBoundary(theta, Xm, y);
hold on;
title(sprintf('lambda = %g', lambda));
hold off;
