function [b,F,valueVec] = UFS2(X,nClass,beta,nSel) 
% main algorithm
% solving:
%   min   ||diag(b)*X-G*F||_F^2 + beta*b'*X*X'*b
%  G,F,b
%   s.t.   b \in {0,1}^d, 1'*b = m,
%          F is a one-hot label matrix.

% Input: 
%    X        -each column is a data point
%    nClass   -total number of classes
%    beta	  -regularization parameter: beta*b'*X*X'*b
%    nSel	  -the dimension of selected features

% Output: 
%    b   	  -feature selection vector
%    F        -each column is a one-hot label
%    valueVec -the objective function value in each iteration


addpath(genpath('.\file'));
%% settings
[Dim, Smp] = size(X);
TEMP = X*X';
max_Iter = 1e2; % maximum number of iterations
tol = 5e-7;     % the other stop criterion for iteration

%% initialization
b0 = ones(Dim,1)*nSel/Dim;
[G0,F0] = solveGF(X,rand(Dim,nClass),rand(nClass,Smp),nClass);
% [G0,F0] = preProcessGF(X,nClass);
valueVec = norm(diag(b0)*X-G0*F0,'fro')^2+beta*(b0'*TEMP*b0);

%% iteration
iter = 1;
while iter <= max_Iter    
    
    % update G and F
    [G,F] = solveGF(diag(b0)*X,G0,F0,nClass,5);
    
    % update b
    A = beta*TEMP;
    b = diag(TEMP-2*X*F'*G');
    [b,~,~] = ForwardGreedy_iter(A, b, nSel);
    
    ObjValue = norm(diag(b)*X-G*F,'fro')^2+beta*(b'*TEMP*b);
    valueVec = [valueVec ObjValue];
    
    % check if stop criterion is satisfied
    leq = b - b0;
    stopC = norm(leq); % F-norm
    if (iter==1 || mod(iter,2)==0 || (stopC<tol) )
        disp(['Iter ' num2str(iter) ...
        ', stopC= ' num2str(stopC,'%2.4e') ...
        ', ObjValue=' num2str(ObjValue,'%0.6f')]);
    end
    if (stopC<tol)
        break;
    end
    
    % update init    
    b0 = b;
    G0 = G;
    F0 = F;
    iter = iter + 1;
    
end
end % whole function