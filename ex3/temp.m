load('ex3data1.mat'); % training data stored in arrays X, y

m = size(X, 1);

m = size(X,1)
n = size(X,2)

initTheta = zeros(n + 1, 1);
Xn = [ones(m,1),X];

hypo=sigmoid(Xn * initTheta);

rTheta    = theta;
rTheta(1) = 0;

t1 =    y1    .* log(  hypo  );
t2 = (1 - y1) .* log(1 - hypo);

J    = sum( -t1 - t2 ) / m + lambda * sumsq(rTheta) / ( 2 * m );
grad = Xn' * ( hypo - y1 ) / m + lambda * rTheta / m;

% ===================================================
% oneVsAll Classifier

options = optimset('GradObj', 'on', 'MaxIter', 50);
for i=1:num_labels
    initTheta = zeros(n + 1, 1);
    theta = fmincg(@(t)(lrCostFunction(t, X, (y == i), lambda)), initTheta, options);

    all_theta(i,:) = theta';
endfor

% ===================================================
% oneVsAllPredict

temp = X * all_theta';
[m, p] = max(temp, [], 2);
