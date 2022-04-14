function [b,F,valueVec] = SimpleUFS_feature(X,nClass,beta,nSel) 
% main algorithm

% Input£º
%    X        -each column is a data point
%    nClass   -total number of classes
%    beta	  -regularization parameter: beta*Tr[diag(b)*X*H*X'*diag(b)]
%    nSel	  -the dimension of selected features

% Output£º
%    b   	  -feature selection vector
%    F        -each column is a one-hot label
%    valueVec -the objective function value in each iteration


%% settings
[Dim, Smp] = size(X);
TEMP = X*X';
rho = 1.1;
max_mu = 1e10;
mu = 1e-6;
max_Iter = 1e2; % maximum number of iterations
tol = 5e-7;     % the other stop criterion for iteration

%% initialization
b0 = ones(Dim,1)*nSel/Dim;
% [G0,F0] = solveGF(X,rand(Dim,nClass),rand(nClass,Smp),nClass);
[G0,F0] = preProcessGF(X,nClass);
y1 = zeros(Dim,1);
y2 = zeros(Dim,1);
b1 = zeros(Dim,1);
b2 = zeros(Dim,1);
valueVec = norm(diag(b0)*X-G0*F0,'fro')^2+beta*(b0'*TEMP*b0);


%% iteration
iter = 1;
while iter <= max_Iter    
    
    % update G and F
    [G,F] = solveGF(diag(b0)*X,G0,F0,nClass,5);
    
    % update b
    A = beta*TEMP + mu*eye(Dim);
    b = diag(TEMP-2*X*F'*G') + y1 + y2 - mu*(b1+b2);
    b = linsolve((A+A'+(b*ones(1,Dim)+ones(Dim,1)*b')/nSel),ones(Dim,1));
    b = nSel*b/sum(b);
    
    % update b1 and b2
    b1 = PSb(b+(y1/mu));
    b2 = PSp(b+(y2/mu));
    
    ObjValue = norm(diag(b)*X-G*F,'fro')^2+beta*(b'*TEMP*b);
    valueVec = [valueVec ObjValue];
    
    % check if stop criterion is satisfied
    leq1 = b - b1;
    leq2 = b - b2;
    stopC1 = max(max(abs(leq1))); % Infinite norm
    stopC2 = max(max(abs(leq2))); % Infinite norm
    if (iter==1 || mod(iter,2)==0 || ((stopC1<tol)&&(stopC2<tol)) )
        disp(['Iter ' num2str(iter) ',mu=' num2str(mu,'%2.4e') ...
        ',stopC= ' num2str(stopC1,'%2.4e') ', '...
        num2str(stopC2,'%2.4e') ', ObjValue=' num2str(ObjValue,'%0.6f')]);
    end
    if (stopC1<tol)&&(stopC2<tol)
        break;
    end
    
    % update Lagrange multipliers    
    y1 = y1 + mu*leq1; %(b-b1);
    y2 = y2 + mu*leq2; %(b-b2);
    mu = min(max_mu, mu*rho);
    
    % update init    
    b0 = b;
    G0 = G;
    F0 = F;
    iter = iter + 1;
    
end
end % whole function

%% PSb function
function X = PSb(X,a,b)
if ~exist('a','var')
    a = 0; b =1;
end
A = X(:); 
ind1 = A<a; ind2 = A>b;
A(ind1)=a; A(ind2) = b; 
X = reshape(A, size(X));
end

%% PSp function
function X = PSp(X)
Dim = length(X); 
ones1 = ones(Dim,1);
t0 = sqrt(Dim)/(2*norm(X-(ones1/2),2)); 
X1 = (ones1/2) + t0*(X-(ones1/2)); 
X2 = (ones1/2) - t0*(X-(ones1/2)); 
d1 = norm(X-X1,2); d2 = norm(X-X2,2);
if d1>d2 
    X = X2;
else
    X = X1;
end
end