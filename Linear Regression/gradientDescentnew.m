function [theta, J_history] = gradientDescentnew(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

 for iter = 1:num_iters

    % ====================== Calculating the weights ======================
    %
	prediction = X * theta;
	errorvector = (prediction - y);
	theta_change = (alpha * (X' * errorvector));
	theta = theta - theta_change;
    theta1(iter) = theta(1);
    theta2(iter) = theta(2);
    % ============================================================
    % Save the cost J in every iteration    
	J_history(iter) = computeCost(X, y, theta);
    plot(J_history,'r');
    xlabel('Iteration');
    ylabel('Cost');
    title('Cost Vs Iteration');
 end
    figure
    plot(theta1,'r');
    xlim([0,250]);
    hold on;
    plot(theta2, 'b');
    hold off;
    legend('Bias','W');
 
end
