%%%%%%%%%%%%%
% Code for generating "biased flexibility distribution" results (Supplementary Fig 2-3)

%% Create folders
if ~exist(loadPath+"results", 'dir')
   mkdir(loadPath+"results")
end
if ~exist(loadPath+"results\info", 'dir')
   mkdir(loadPath+"results\info")
end
if ~exist(loadPath+"results\networks", 'dir')
   mkdir(loadPath+"results\networks")
end

%% Load two-digit MNIST dataset
total_category = []; % possible two-digit catagories
for ii = 0:8
    for jj = ii+1:9
        total_category = [total_category string(strcat(num2str(ii),num2str(jj)))];
    end
end

if ~exist(loadPath+"twoDigitMNIST", 'dir')
   create_twoDigitMNIST;
end

% load create imageDataStore for train and test
imdsTrain = imageDatastore(fullfile(strcat(loadPath,"\twoDigitMNIST\training"),total_category),"LabelSource","foldernames");
imdsTrain.ReadFcn = @imgAugmentation;
imdsTrain = splitEachLabel(imdsTrain,0.4);

imdsValidation = imageDatastore(fullfile(strcat(loadPath,"\twoDigitMNIST\test"),total_category),"LabelSource","foldernames");
imdsValidation.ReadFcn = @imgAugmentation;

%% Create arrays to store performance data and trained category information
performanceCell = zeros(numCategories+1,numCategories,numTrial,nwNum); % memory performance of classes after training each item
epochPerformanceCell = zeros(numEpochs*numCategories+1,numCategories,numTrial,nwNum); % memory performance of classes after training each epoch
categoryCell = strings(numTrial,numCategories); % trained catagories on each trial
if calculateShuffleAccuracy; shuffleAccCell = zeros(numCategories+1,numCategories,shuffleNum,numTrial,nwNum); end

% Load alexnet
net = alexnet;

layers = [
    net.Layers(1:end-9)
    fullyConnectedLayer(4096,"Name","fc6")
    reluLayer("Name","relu6")
    fullyConnectedLayer(4096,"Name","fc7")
    reluLayer("Name","relu7")
    fullyConnectedLayer(numCategories,"Name","fc8")
    softmaxLayer("Name","prob")];

FClayersIdx = [17, 19, 21]; % Indices of fully connected layers
initNet = dlnetwork(layers);

%%
for trial = 1:numTrial
    
    % Select categories to be trained on networks
    targ_category = total_category(randperm(numel(total_category),numCategories));
    categoryCell(trial,:) = targ_category;
        
    % Indices of images composing the training dataset
    trainImdIdx = cell(1,numCategories);
    for ii = 1:numCategories
        trainImdIdx{ii} = find(imdsTrain.Labels == targ_category(ii));
        if numel(trainImdIdx{ii})==0; trainImdIdx{ii} = find(imdsTrain.Labels == "0"+targ_category(ii)); end
    end
    clear ii
    
    % Indices of images composing the test dataset
    clsImdIdx = [];
    tempImdsTest = copy(imdsValidation);
    for ii = 1:numCategories
        clsImdIdx = [clsImdIdx; find(tempImdsTest.Labels == targ_category(ii))];
    end
    imdsClassTest = subset(tempImdsTest,clsImdIdx);
    clear tempImdsTest clsImdIdx ii

    % Initialize networks
    clear netCell

    initNet = initializeWeight(initNet,1,1,FClayersIdx);
    netCell = cell(nwNum,1);
    for nn = 1:nwNum
        netCell{nn} = initNet;
    end

    % Define synaptic flexibility values of three different models
    flexibility = cell(nwNum,numel(FClayersIdx));
    
    a = 0.5;
    x = 0:0.001:1;pd = makedist('PiecewiseLinear','x',x,'Fx',(1-a)*x.^2+a*x);
    for jj = 1:numel(layersIdx)
        for nn = 1:nwNum
            numW=size(initNet.Layers(layersIdx(jj)).Weights);
            if nn==1; flexibility{nn,jj} = random(pd,numW); end % linearly increasing
            if nn==2; flexibility{nn,jj} = 1-random(pd,numW); end % linearly decreasing
        end
        clear numW jj
    end
    clear pd x

    % Before training, measure the initial memory performance of networks
    for nn = 1:nwNum
        testPerformance = testModel(netCell{nn},imdsClassTest,targ_category);
        performanceCell(1,:,trial,nn) = testPerformance;
        epochPerformanceCell(1,:,trial,nn) = testPerformance;
        if verbal
            if nn == 1; disp('Conventional network performance: ');
            else if nn == 2; disp('Hybrid network performance: ');
            else if nn == 3; disp('Stable network performance: ');
            end; end; end
            disp(performanceCell(1,:,trial,nn))
        end
        
        if calculateShuffleAccuracy
            % Calculate shuffled-accuracy
            [YTest, TTest] = calculateChance(netCell{nn},imdsClassTest,targ_category);
            for chanceIdx = 1:shuffleNum
                shuffle_TTest = TTest;
                shuffle_TTest = shuffle_TTest(randperm(numel(shuffle_TTest)));
                for target = 1:size(performanceCell,2)
                    T_index = find(shuffle_TTest == targ_category(target));
                    shuffleAccCell(1,target,chanceIdx,trial,nn) = numel(find(YTest(T_index) == targ_category(target)))/numel(T_index);
                end
                clear target shuffle_TTest T_index
            end
        end
    end
    clear nn

    %% Training loop
    trainOpt = [numEpochs, miniBatchSize, learnRate];

    % Items are trained sequentially
    for trainIdx = 1:numCategories
        
        % Permute the labels of 'non-target' items
        clear tempImdsTrain
        tempImdsTrain = copy(imdsTrain);
        
        % Create a shuffled set of non-target labels
        clear clsImdIdx nclsImdIdx nClsLab noncurrClsLab addIdx
        clsImdIdx = trainImdIdx{trainIdx};
        noncurrClsIdx = setdiff(1:numCategories,trainIdx);
        noncurrClsNum = round(numel(clsImdIdx)/(numCategories-1));
        noncurrClsLab = [];
        for nonIdx = noncurrClsIdx
            repCls = setdiff(noncurrClsIdx,nonIdx);
            noncurrClsLab = [noncurrClsLab,repmat(repCls,1,floor(noncurrClsNum/numel(repCls)))];
            addIdx = randi(numel(repCls),1,rem(noncurrClsNum,numel(repCls)));
            noncurrClsLab = [noncurrClsLab, noncurrClsIdx(addIdx)];
        end
        
        nclsImdIdx = [];
        for jj=noncurrClsIdx
            tempIdx=trainImdIdx{jj};
            tempIdx = tempIdx(randperm(noncurrClsNum));
            nclsImdIdx = [nclsImdIdx; tempIdx];
        end
        clear jj tempIdx
        
        tempImdsTrain = subset(tempImdsTrain,[clsImdIdx; nclsImdIdx]);
        clsImdIdx = find(tempImdsTrain.Labels==targ_category(trainIdx));
        nclsImdIdx = setdiff(1:numel(tempImdsTrain.Labels),clsImdIdx);
        
        % Re-label the non-target items
        tempImdsTrain.Labels(nclsImdIdx) = categorical(targ_category(noncurrClsLab));
        tempImdsTrain.Labels = removecats(tempImdsTrain.Labels,setdiff(total_category,targ_category));
        tempImdsTrain.Labels = reordercats(tempImdsTrain.Labels,targ_category);
        
        % Train different networks parallely
        for nn = 1:nwNum
            [netCell{nn}, epochPerformanceCell((numEpochs*(trainIdx-1)+2):(numEpochs*trainIdx+1),:,trial,nn)] = trainModel(trainOpt,netCell{nn},initNet,tempImdsTrain,flexibility(nn,:),s,FClayersIdx,epochwiseTest,imdsClassTest,targ_category);
        end
        
        % Measure the performance of networks after training an item
        for nn = 1:nwNum
            performanceCell(trainIdx+1,:,trial,nn) = epochPerformanceCell(numEpochs*trainIdx+1,:,trial,nn);
            if verbal
                if nn == 1; disp('Conventional network performance: ');
                else if nn == 2; disp('Hybrid network performance: ');
                else if nn == 3; disp('Stable network performance: ');
                end; end; end
                disp(performanceCell(1:trainIdx+1,:,trial,nn))
            end
            if calculateShuffleAccuracy
                % Calculate shuffled-accuracy
                [YTest, TTest] = calculateChance(netCell{nn},imdsClassTest,string(targ_category));
                for chanceIdx = 1:shuffleNum
                    shuffle_TTest = TTest;
                    shuffle_TTest = shuffle_TTest(randperm(numel(shuffle_TTest)));
                    for target = 1:size(performanceCell,2)
                        T_index = find(shuffle_TTest == targ_category(target));
                        shuffleAccCell(trainIdx+1,target,chanceIdx,trial,nn) = numel(find(YTest(T_index)==targ_category(target)))/numel(T_index);
                    end
                    clear target shuffle_TTest T_index
                end
            end
        end
        if verbal; disp(['Class-' num2str(trainIdx) '-trial-' num2str(trial) '-running']); end
        clear label tempImdsTrain imdIdx tempSeq
    end
    if saveLastNetwork; save(loadPath+"results\networks\biased-flexibility-distribution-networks-trial-"+num2str(trial),"initNet","netCell","flexibility"); end
end

if calculateShuffleAccuracy
    save(strcat(loadPath,"results\biased-flexibility-distribution.mat"),"performanceCell","epochPerformanceCell","shuffleAccCell","categoryCell");
    save(strcat(loadPath,"results\info\info-biased-flexibility-distribution.mat"),"numTrial","numCategories","numEpochs","miniBatchSize","learnRate","s","nwNum","shuffleNum");
else
    save(strcat(loadPath,"results\biased-flexibility-distribution.mat"),"performanceCell","epochPerformanceCell","categoryCell");
    save(strcat(loadPath,"results\info\info-biased-flexibility-distribution.mat"),"numTrial","numCategories","numEpochs","miniBatchSize","learnRate","s","nwNum");
end

