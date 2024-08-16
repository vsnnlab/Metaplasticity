function [YTest, TTest] = calculateChance(net,imdsTest,classes)
numOutputs = 1;

auimds = augmentedImageDatastore([227 227],imdsTest);

miniBatchSize = 500;

mbqTest = minibatchqueue(auimds,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB", ...
    OutputEnvironment="auto");

YTest = modelPredictions(net,mbqTest,classes);

TTest = imdsTest.Labels;

end