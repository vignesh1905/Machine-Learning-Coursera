function g = tanhGradient(z)
%TANHGRADIENT returns the gradient of the tanh function
%evaluated at z
%   g = TANHGRADIENT(z) computes the gradient of the tanh function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the tanh function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

  %Calculating tanh
  tan = tanh(z); 
  %Calculation tanh gradient
  g = (1- (tan .^ 2)); 

% =============================================================




end
