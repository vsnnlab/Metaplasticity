function accuracy = testModel(net,imdsTest,classes)
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

accuracy = zeros(1,numel(classes));
for cc = 1:numel(classes)
    ii = find(TTest == classes(cc));
    accuracy(cc) = numel(find(YTest(ii)==classes(cc)))/numel(ii);
end
clear cc ii

% confMat = confusionmat(string(YTest),string(TTest),'Order',classes);
% accuracy = diag(confMat./(numel(YTest)/numel(classes)));
clear confMat

end