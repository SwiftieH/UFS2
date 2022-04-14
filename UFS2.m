% solving:
%   min   ||diag(b)*X-G*F||_F^2 + beta*b'*X*X'*b
%  G,F,b
%   s.t.   b \in {0,1}^d, 1'*b = m,
%          F is a one-hot label matrix.



addpath(genpath('.\files'));
DATAset = {...
    'ALLAML';...
    'COIL20';...
    'PIE10P';...
    'TOX_171';...
    'PIX10P';...
    'Prostate_GE';...
    'UMIST';...
    };
DATAsetID = 7;
load(strcat('.\Data\',DATAset{DATAsetID},'.mat'),'fea','gnd');

%% Normalization and Initialization
[nSmp, nFea] = size(fea); % each row is a sample
nClass = length(unique(gnd));
% fea = (fea-min(fea(:)))/(max(fea(:))-min(fea(:)));
% fea = bsxfun(@rdivide, bsxfun(@minus,fea,mean(fea,2)), std(fea,[],2));
fea = normcols(fea);

%% 

lamda1Vec = [1e-6 1e-4 1e-2 1 1e2 1e4 1e6];% beta
lamda2Vec = 1; %[0.1:0.2:0.9];% beta
if nFea < 300 
    lamda3Vec = [50:30:200]; % m: number of selected features
else
	lamda3Vec = [50:50:300]; % m: number of selected features
end

ACC = zeros(length(lamda1Vec),length(lamda2Vec),length(lamda3Vec));
NMI = zeros(length(lamda1Vec),length(lamda2Vec),length(lamda3Vec));
for k=1:length(lamda3Vec)
    lamda3 = lamda3Vec(k);
    for i=1:length(lamda1Vec)
        lamda1 = lamda1Vec(i);
        for j=1:length(lamda2Vec)
            lamda2 = lamda2Vec(j);
            fprintf('\nWorking...\n');
            [b,F,valueVec] = SimpleUFS_feature(fea',nClass,lamda1,lamda3);
            [~,feaIdx] = sort(b,'descend');
            [idxClass,~] = litekmeans(fea(:,feaIdx(1:lamda3)),nClass,'Replicates',10);
            fprintf('Done!\n');
			idxClass = bestMap(gnd,idxClass);
            Acc = sum(idxClass==gnd)/nSmp;
            Nmi = nmi(idxClass,gnd);
            fprintf('beta=%.06f,m=%.06f:\n ACC=%.06f,NMI=%.06f.\n',lamda1,lamda3,Acc,Nmi);
            ACC(i,j,k) = Acc;
            NMI(i,j,k) = Nmi;
        end
    end
end


save(strcat('.\result\',DATAset{DATAsetID},'_result.mat'),...
    'ACC','NMI');