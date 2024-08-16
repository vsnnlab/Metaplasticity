function [outputArg1,outputArg2] = generate2DMNIST(category)
%GENERATE2DMNIST Summary of this function goes here
%   Detailed explanation goes here

rawdata = {};
for ii = 1:numel(category)
    rawdata{ii} = load(strcat(pwd,'\MNIST2digit\training\img'));
end

ds = {};
for ii = 1:nume(category)
    XTrain = rawdata{ii}.dataOut;

    ds{ii} = augmentedImageDatastore([227 227],XTrain,categorical(YTrain));

