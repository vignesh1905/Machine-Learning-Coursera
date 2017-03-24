function g = tanh(z);

% tanh function is used to compute the non-linear function in the hidden layer for linear systems.
%The non-linear function is often the tanh() function - it has an output range from -1 to +1, and its gradient is easily implemented. Let g(z)=tanh(z). tanh is also a type of sigmoid.
%The gradient of tanh is g′(z)=1−g(z).Use this in backpropagation in place of the sigmoid gradient.

 g = ((exp(2 * z)) - 1.0)./ ((exp(2 * z)) + 1.0);
 end