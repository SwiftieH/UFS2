function [G,F] = preProcessGF(X,nClass,nRun) 
% Using K-means+PCA to initialize G and F

% Input: 
%       X       -each column is a data point
%       nClass  -total number of classes
%       nRun	-times of randomly running k-means  

% Output: 
%       G       -each column is a non-negative base
%       F       -each column is a non-negative orthogonal coefficient

if ~exist('nRun','var')
    nRun = 1; 
end

options.ReducedDim = nClass;
[W,~] = myPCA(X',options);

[~, nSmp] = size(X);
nPerClass = zeros(1,nClass);
F = zeros(nClass, nSmp);
for i=1:1:nRun
   [label, center] = litekmeans(X'*W,nClass,'Replicates',10);
    for j=1:1:nClass
       nPerClass(j) = sum(label==j); 
       F(j,(label==j)) = 1;
    end
    G = X*F'/(F*F');
end
