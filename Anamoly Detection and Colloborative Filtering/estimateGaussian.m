function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.

  meancalc = mean (X);
  mu = meancalc(:); % converting row matric to column matrix 
  intercal = X .- meancalc ;
  intercal2 = ((1 / m) * (sum(intercal .^ 2))); % here we do not use var because it calculates over 1/(m -1) rather than 1/m
  sigma2 = intercal2(:);
% =============================================================


end
