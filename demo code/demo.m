load Iris.mat
% load wine.mat
% load data_USPS.mat
gt=data(:,end);
H=30;
k=length(unique(gt));
data_feature=data(:,1:end-1);
data_feature=predata(data_feature);
[clusterings] =creat_clusters_randomk_kmeans(data_feature,H,k);
[result,medoids] = CEHM(clusterings,k);
[ac,ARI,NMI]=evaluate2(result,gt,k)