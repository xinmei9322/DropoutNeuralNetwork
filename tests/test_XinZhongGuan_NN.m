%function test_XinZhongGuan_NN
clear all;close all;clc;
addpath(genpath('D:\download\DeepLearnToolbox'));
Tr = load('D:\download\XinZhongGuan\XinZhongGuan.F1.train.C.mat');
Tt = load('D:\download\XinZhongGuan\XinZhongGuan.F1.test.C.mat');
nsamples = size(Tr.input,1);
train_x = double(Tr.input) / max(max(abs(Tr.input)));
test_x  = double(Tt.input)  / max(max(abs(Tt.input)));

train_y2 = toOneOfMany(Tr.classes,Tr.nClasses,0);
test_y2 = toOneOfMany(Tt.classes,Tt.nClasses,0);

train_y = train_y2;
test_y = test_y2;


rand('state',0)
num_in = size(Tr.input,2);
num_out = size(Tr.class_labels,1);
num_hid = floor(0.3*num_in+0.7*num_out);
nn = nnsetup([num_in 190 55 num_out]);
nn.weightPenaltyL2 = 0;  %  L2 weight decay
nn.learningRate = 2;
%nn.scaling_learningRate = 0.998;            %  Scaling factor for the learning rate (each epoch)
nn.momentum = 0.5;          %  Momentum
nn.inputZeroMaskedFraction = 0;            %  Used for Denoising AutoEncoders
nn.dropoutFraction = 0.5;   %  //Dropout fraction，每一次mini-batch样本输入训练时，随机扔掉50%的隐含层节点
nn.output  = 'softmax';                   %  use softmax output
opts.numepochs =70;        %  //Number of full sweeps through data
opts.batchsize = 50;       %  //Take a mean gradient step over this many samples
opts.plot = 1;              %  enable plotting
train_x = train_x(1:nsamples-mod(nsamples,opts.batchsize),:);
train_y = train_y(1:nsamples-mod(nsamples,opts.batchsize),:);
% normalize
[train_x, mu, sigma] = zscore(train_x);
%test_x = normalize(test_x, mu, sigma);
test_x = zscore(test_x);
% split training data into training and validation data
randindex = randperm(nsamples-mod(nsamples,opts.batchsize));
% vx   = train_x(randindex(1:200),:);
% tx = train_x(randindex(201:end),:);
% vy   = train_y(randindex(1:200),:);
% ty = train_y(randindex(201:end),:);
 vx = test_x;
 tx = train_x;
 vy = test_y;
 ty = train_y;
nn = nntrain(nn, tx, ty, opts, vx, vy);
%nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
str = sprintf('testing error rate is: %f',er);
disp(str)

