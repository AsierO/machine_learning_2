function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J1 = 0;
J2=0;
J=0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% for i = 1:m
%     J1 = J1 -( y(i)*log(ones(size(-theta' *  X(i,:)'))./(1+exp(-theta' *  X(i,:)')))+ (1-y(i))*log(1-(ones(size(theta' *  X(i,:)'))./(1+exp(-theta' *  X(i,:)')))));
%     
% end
% 
% J1 = J1/m;
% 
% J2=(lambda/(2*m))*sum(theta.^2);
% 
% J=J1+J2;


% for k = 1:m
%     
%     if k==1
%     grad = grad +((((ones(size(-theta' *  X(k,:)'))./(1+exp(-theta' *  X(k,:)')))-y(k)).*X(k,:))')/m;
%     
%     else
%         
%      grad = grad +((((ones(size(-theta' *  X(k,:)'))./(1+exp(-theta' *  X(k,:)')))-y(k)).*X(k,:))')/m +(lambda/m)*theta;   
%         
%     end
% 
% end
% 
% for k = 1:m
%     
%     grad = grad +((((ones(size(-theta' *  X(k,:)'))./(1+exp(-theta' *  X(k,:)')))-y(k)).*X(k,:))')/m;
%     
% end
% 
% 
% grad(:,2:length(grad))=grad/m+(lambda/m)*theta(2:length(theta));

J = ( (1 / m) * sum(-y'*log(sigmoid(X*theta)) - (1-y)'*log( 1 - sigmoid(X*theta))) ) + (lambda/(2*m))*sum(theta(2:length(theta)).*theta(2:length(theta))) ;



grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) );

grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';





% =============================================================

end
