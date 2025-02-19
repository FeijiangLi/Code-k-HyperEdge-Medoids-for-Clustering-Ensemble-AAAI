function [clusters] =creat_clusters_randomk_kmeans(data,H,k)
[n,~]=size(data);
clusters=zeros(n,H);
k1=min(floor(sqrt(n)),50);
k2=floor(3*k/2);
maxk=max(k1,k2);
allk=round(rand(1,H)*(maxk-k))+k;
for i=1:H
    k=allk(i);
   clusters(:,i)=kmeans(data,k,'emptyaction','singleton');
end

end