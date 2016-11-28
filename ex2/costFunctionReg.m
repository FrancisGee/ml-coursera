function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

theta_tmp = theta(2:length(theta),:);
J = 1./m * (-y' * log(sigmoid(X * theta)) - (1-y') * log(1 - sigmoid(X * theta)))+...
    (lambda./(2*m))*(theta_tmp'*theta_tmp);
grad_temp = 1./m * X' * (sigmoid(X * theta) - y);
grad(1) = grad_temp(1);
grad_re = 1./m * X' * (sigmoid(X * theta) - y) + (lambda./m)*theta;
grad(2:length(grad)) = grad_re(2:length(grad_re));


% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
