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
h = sigmoid(X*theta);
J = ((-y'*log(h)-(1-y)'*log(1-h))/m) + (lambda*(sum([theta(2:size(theta,1),:).^2]))/(2*m));

for i = 1 : m,
	grad = grad + (h(i) - y(i)) * X(i,:)';
end

ta = [0;theta(2:end)];

grad = 1/m*grad + lambda/m*ta;
grad1 = grad;

% both of these method works

grad = ((X'*(h-y))/m) + ((lambda*[0;theta(2:end)])/m);
if grad1 == grad
  fprintf('\ncoooooooooooooool\n');
end
%grad = (X'*(h-y))/m;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
