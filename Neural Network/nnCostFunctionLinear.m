 function [J grad]  = nnCostFunctionLinear(nn_params,
                                           input_layer_size,
										   hidden_layer_size,
										   X,y,lambda)

%NNCostFunctionLinear implements the costfunction for a 2 layer neural network
%which peforms linear regression function

%   [J grad] = NNCOSTFUNCTONLINEAR(nn_params, hidden_layer_size, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 1, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1), X]; % adding bias unit to X(input)
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
     
	 % Feed forward network computation
      a1 = X;
      z2 = a1 * Theta1'; 
      a2 = tanh(z2); % using tanh instead of logistic since here we are training for a linear system
      a2 = [ones(m,1), a2]; % adding the bias unit as per 
      a3 = a2 * Theta2';
	  
	  %Unregularized cost computation
	  sqrerror = (a3 - y).^ 2; % Square error computation
      J = 1/(2*m)* sum(sqrerror);  %Cost function calculation
	  
	  % Regularization terms computation
	  j1 = ((lambda/ (2*m)) * sum(sum(Theta1(:, 2:end) .^ 2))); % We use (:, 2:end) to exculde the bias column from the regularization
	  j2 = ((lambda/ (2*m)) * sum(sum(Theta2(:, 2:end) .^ 2)));
	  % Regularized cost function
	  J =  J + j1 + j2 ; 
	  
	  % Back Propogation
	  d3 = a3 - y ; %Computing small delta (errors) 
	  d2 =  (d3 * Theta2(:,2:end) ) .* (tanhGradient(z2)); 
	  Delta1 = d2' * a1 ; %Computing capital delta(gradient)
	  Delta2 =  d3' * a2 ; 
	  
	  %Gradient without regularization
	  Theta1_grad = (1/m) * Delta1;
	  Theta2_grad = (1/m) * Delta2; 
	  
	  % Gradient regularization
	  Theta1(:,1) = 0; % This is done to avoid the bias terms
	  Theta2(:,1) = 0;
      Theta1 = (lambda/m) * (Theta1); % Regularizing	  
	  Theta2 = (lambda/m) * (Theta2);
	  Theta1_grad = Theta1_grad + Theta1;
	  Theta2_grad = Theta2_grad + Theta2;	  	  
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
	  
	  
	  
	  
	  
	  