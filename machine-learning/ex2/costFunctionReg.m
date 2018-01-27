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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
A = h - y;

for i=1:m
        J += -y(i) * log(h(i)) - (1 - y(i)) * log(1 - h(i));
end;

J /= m;

regularizationTerm = (lambda / (2 * m)) * (theta(2:end)' * theta(2:end));

J += regularizationTerm;

gradRegularizationMultiplier = eye(size(theta, 1), size(theta, 1));
gradRegularizationMultiplier(1, 1) = 0;

grad = (X' * A ./ m) + ((lambda / m) * gradRegularizationMultiplier * theta);

% =============================================================

end
