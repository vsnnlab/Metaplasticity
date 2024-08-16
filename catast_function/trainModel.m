function [net, epochPerformance] = trainModel(trainOpt,net,initNet,imdsClassTrain,flexibility,s,layersIdx,epochWiseTest,imdsClassTest,targ_category)

if nargin <= 8
    imdsClassTest = []; targ_category = [];
end

numEpochs = trainOpt(1);
miniBatchSize = trainOpt(2);
learnRate = trainOpt(3);
velocity = [];
momentum = 0;
learnableIdx = find(contains(net.Learnables.Layer, "fc"));

auimds = augmentedImageDatastore([227 227],imdsClassTrain);

mbq = minibatchqueue(auimds,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB",""], ...
    OutputEnvironment='auto');
    
epoch = 0;
iteration = 0;
epochPerformance = [];

alpha = cell(3,1);
for ii = 1:length(layersIdx)
    w = net.Layers(layersIdx(ii)).Weights;
    wI = initNet.Layers(layersIdx(ii)).Weights;
    flex = flexibility{ii};
    coeff = (flex-1)./flex;
    alphaTemp = 1-(tanh(coeff.*(w-wI)*s).^2);
    alphaTemp(isinf(coeff) & w-wI == 0) = 0;
    alpha{ii} = alphaTemp;
    clear alphaTemp
end

% Loop over epochs.
while epoch < numEpochs % && ~monitor.Stop
    epoch = epoch + 1;
    batch = 0;
    errorEpoch=0;

    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq) %&& ~monitor.Stop

        iteration = iteration + 1;
        batch = batch + 1;
        
        % Read mini-batch of data.
        [X,T] = next(mbq);
        
        [error,gradients,state] = dlfeval(@lossCalculation,net,X,T);
        if isnan(error); disp("break at iter = "+num2str(iteration)); break; end
        net.State = state;
        errorEpoch = errorEpoch + error;
        gradIdx = find(gradients.Parameter == "Weights");
        for ii = 1:length(layersIdx)
            a = alpha{ii};
            gTemp = gradients{gradIdx(ii),3};
            gTemp = gTemp{1};
            gradients{gradIdx(ii),3} = {gTemp.*a};
            clear gTemp a
        end
        [net.Learnables(learnableIdx,:),~] = sgdmupdate(net.Learnables(learnableIdx,:),gradients,velocity,learnRate,momentum);

    end

    if epochWiseTest; epochPerformance = [epochPerformance; testModel(net,imdsClassTest,string(targ_category))]; end

    disp(['epoch ',num2str(epoch),' done!!'])
    disp(['error: ',num2str(errorEpoch)])
end
end

