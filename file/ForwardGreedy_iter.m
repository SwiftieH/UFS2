function [s,ID,optV] = ForwardGreedy_iter(A, b, m) %iteration
% to solve
%  min_s  s'*A*s + s'*b
%   s.t.  s is binary, ||s||_0 = m
% 
% Input:
%       A       -square matrix
%       b       -column vector, the same length as s
%       m       -||s||_0 = m
% Output:
%       s       -column vector
%       ID      -row vector: a index set for 1 in s
%       optV    -minimal value of s'*A*s+s'*b


%% setting
A = 0.5*(A+A');
m = min(max(1,m),length(b));

%% search
s = zeros(length(b),1);
[optV,ID] = min(diag(A)+b);
s(ID) = 1;
if m==1    
    return;
else
    for jdx = 2:1:m
        Temp = optV + diag(A) + 2*sum(A(:,ID),2) + b;
        Temp(ID) = NaN;
        [optV,id] = min(Temp);
        s(id) = 1;
        ID = [ID id];
    end
    return;
end
end