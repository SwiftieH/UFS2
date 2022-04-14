function [G,F] = solveGF(X,Ginit,Finit,nClass,nRun)
% Using K-means to update G and F
%  min  ||X-G*F||_F^2
%  G,F

% Input£º
%       X       -each column is a data point
%       Ginit   -an initialization of G
%       Finit   -an initialization of F
%       nClass  -total number of classes
%       nRun	-times of randomly running k-means

% Output£º
%       G       -each column is a non-negative base
%       F       -each column is a non-negative orthogonal coefficient


if ~exist('nRun','var')
    nRun = 1; 
end
[~, nSmp] = size(X);
nPerClass = zeros(1,nClass);
F = zeros(nClass, nSmp);
for i=1:1:nRun
   [label, center] = litekmeans(X',nClass,'Replicates',10);
    for j=1:1:nClass
       nPerClass(j) = sum(label==j); 
       V(j,(label==j)) = 1;
    end
    G = center';
    if ( norm(X-G*F,'fro') < norm(X-Ginit*Finit,'fro') )
        Ginit = G;
        Finit = F;
    end
end
G = Ginit;
F = Finit;