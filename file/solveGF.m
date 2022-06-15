function [G,F] = solveGF(X,Ginit,Finit,nClass,nRun) 
% Using K-means to update G and F
%  min  ||X-G*F||_F^2
%  G,F 

% Input: 
%       X       -each column is a data point
%       Ginit   -an initialization of G
%       Finit   -an initialization of F
%       nClass  -total number of classes
%       nRun	-times of randomly running k-means  

% Output: 
%       G       -each column is a basis
%       F       -each column is a one-hot label


if ~exist('nRun','var')
    nRun = 1; 
end

for i = 1:1:nRun
    [label, center] = litekmeans(X',nClass,'Replicates',10);
    F = labelConvert(label);
    G = center';
    if ( norm(X-G*F,'fro') < norm(X-Ginit*Finit,'fro') )
        Ginit = G;
        Finit = F;
    end
end
G = Ginit;
F = Finit;