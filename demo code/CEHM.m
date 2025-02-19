function [cl,medoids] = CEHM(clusters,k)
l=size(clusters,2);
clb = BI_clusters(clusters);
[~,medoids]=kmedoids(clb',k,'Distance','jaccard');
medoids=medoids';
ll=sum(medoids,2)>1;
medoids(ll,:)=0;
[cl,medoids] = Step_3_assign2(clb,medoids);
end


function clb = BI_clusters(cl)
[n,m]=size(cl);
[newE, no_allcl] = relabelCl(cl);

if sum(newE)==0
    clb = ones(n,1);
else
    clb=zeros(n,no_allcl);
    for i=1:m
        now_cl=newE(:,i);
        for j=min(now_cl):max(now_cl)
            clb(:,j)=now_cl==j;
        end
    end
end
end


function [cl,medoids] = Step_3_assign2(clb,medoids)
[n,k]=size(medoids);
n_c=size(clb,2);
n_assign=sum(sum(medoids,2)==1);

step=n-n_assign;
step=ceil(step/k);
step=max(floor(sqrt(n)),step); 

while n_assign<n
    obj_sample=zeros(n,k);
    n_clb=sum(clb);
    all_jc_old=1-clb'*medoids./n_clb';
    for i=1:n_c
        now_locat=clb(:,i)==1;
        obj_sample(now_locat,:)=obj_sample(now_locat,:)+all_jc_old(i,:);
    end
    [confidence,lable]=sort(obj_sample,2);
    confidence = confidence(:,1)-confidence(:,2);
    n_assign=n_assign+min(n-n_assign,step);
    [~,locat]=sort(confidence);
    indices = [locat(1:n_assign) lable(locat(1:n_assign),1)];
    medoids=zeros(n,k);
    medoids(sub2ind(size(medoids), indices(:,1), indices(:,2))) = 1;
end

flag=1;
while flag
    obj_sample=zeros(n,k);
    n_clb=sum(clb);
    all_jc_old=1-clb'*medoids./n_clb';
    for i=1:n_c
        now_locat=clb(:,i)==1;
        obj_sample(now_locat,:)=obj_sample(now_locat,:)+all_jc_old(i,:);
    end
    [~,lable]=min(obj_sample,[],2);
    indices = [(1:1:n)' lable];
    new_medoids=zeros(n,k);
    new_medoids(sub2ind([n,k], indices(:,1), indices(:,2))) = 1;

    if isequal(medoids, new_medoids)
        flag=0;
    end  
    medoids=new_medoids;
end

[~,cl]=max(medoids,[],2);
end


function [newE, no_allcl] = relabelCl(E) 
%==========================================================================
% FUNCTION: [newE, no_allcl] = relabelCl(E)
% DESCRIPTION: This function is used for relabelling clusters in the ensemble 'E'
%
% INPUTS:    E = N-by-M matrix of cluster ensemble
%
% OUTPUT: newE = N-by-M matrix of relabeled ensemble
%     no_allcl = total number of clusters in the ensemble
%==========================================================================
% copyright (c) 2010 Iam-on & Garrett
%==========================================================================

[N,M] = size(E); % no. of clustering
newE = zeros(N,M);

%--- first clustering
ucl = unique(E(:,1)); % all clusters in i-th clustering
if (max(E(:,1) ~= length(ucl)))
    for j = 1:length(ucl)
        newE(E(:,1) == ucl(j),1) = j; % re-labelling
    end
end

%--- the rest of the clustering
for i = 2:M
    ucl = unique(E(:,i)); % all clusters in i-th clustering
    prevCl = length(unique(newE(:,[1:i-1])));
    for j = 1:length(ucl)
        newE(E(:,i) == ucl(j),i) = prevCl + j; % re-labelling
    end
end

no_allcl = max(max(newE));
end