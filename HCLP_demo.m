% Reference:
% Bokai Cao, Xiangnan Kong and Philip S. Yu. Collective Prediction of 
% Multiple Types of Links in Heterogeneous Information Networks. 
% In ICDM 2014.
%
% Dependency:
% Chih-Chung Chang and Chih-Jen Lin. 
% LIBSVM: A Library for Support Vector Machines.
% In ACM Transactions on Intelligent Systems and Technology 2011.
% Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

clear
clc

addpath(genpath('./libsvm-3.22/matlab'));

dataset = ExpDatasetSYN();
[train_data, train_label, test_data, test_label, schema] = dataset.load();

classifier = ExpClassifierHCLP();

[outputs, pre_labels, model] = classifier.classify(...
    train_data, train_label, test_data, schema);
