%%%%%%%%%%%%%
% Code for generating "learning frequency-dependent memory consolidation" results (Fig 5)

%% Basic settings, load random seed
if ~exist(loadPath+"results", 'dir')
   mkdir(loadPath+"results")
end
if ~exist(loadPath+"results\info", 'dir')
   mkdir(loadPath+"results\info")
end
if ~exist(loadPath+"results\networks", 'dir')
   mkdir(loadPath+"results\networks")
end

load(strcar(loadPath,"hyperparameters.m"))
if numCategories == 10; s = s_item_10;
else if numCategories == 30; s = s_item_30; end; end

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

% We offer sample train orders where the correlation between item order and
% item's training frequency is weak enough.
trainOrderList = load('sample_frequency_profiles.mat').trainOrderList;
trainOrderList = trainOrderList(1:numFreqProfile,:);

numTrial = numInitNetwork*numFreqProfile;

freqVarPerformanceCell = zeros(numCategories*(numCategories+1)/2+1,numCategories,numTrial,nwNum); % precision of each class after training each item
freqVarCategoryCell = strings(numTrial,numCategories*(numCategories+1)/2); % trained catagories on each trial
if calculateShuffleAccuracy; shuffleAccCell = zeros(numCategories*(numCategories+1)/2+1,numCategories,shuffleNum,numTrial,nwNum); end

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
% Loop for various network initialization status
for nwTrial = 1:numInitNetwork

    % Initialize networks
    clear netCell

    initNet = initializeWeight(initNet,1,1,FClayersIdx);
    netCell = cell(nwNum,1);
    for nn = 1:nwNum
        netCell{nn} = initNet;
    end
    
    % Define synaptic flexibility values of three different models
    flexibility = cell(nwNum,numel(FClayersIdx));
    
    for jj = 1:numel(FClayersIdx)
        for nn = 1:nwNum
            numW=size(initNet.Layers(FClayersIdx(jj)).Weights);
            if nn==1; flexibility{nn,jj} = ones(numW); end % Conventional network
            if nn==2; tempFlexibility = rand(numW); flexibility{nn,jj} = tempFlexibility; end % Hybrid network
            if nn==3; flexibility{nn,jj} = 0.3*ones(numW); end % Stable network
        end
        clear numW tempFlexibility jj nn
    end
    
    % Loop for various item training frequency status
    for freqTrial = 1:numFreqProfile
        trial = numFreqProfile*(nwTrial-1) + freqTrial;

        % Select categories to be trained on networks
        targ_category = total_category(randperm(numel(total_category),numCategories));
        trainOrder = trainOrderList(freqTrial,:);
        freqVarCategoryCell(trial,:) = targ_category(trainOrder);

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

        % Actual training loop
        % Before training, measure the initial precision of networks
        for nn = 1:nwNum
            freqVarPerformanceCell(1,:,trial,nn) = testModel(netCell{nn},imdsClassTest,string(targ_category));
            if verbal
                if nn == 1; disp('Conventional network performance: ');
                else if nn == 2; disp('Hybrid network performance: ');
                else if nn == 3; disp('Stable network performance: ');
                end; end; end
            end
            disp(freqVarPerformanceCell(1,:,trial,nn))
            if calculateShuffleAccuracy
                % Calculate shuffled-accuracy
                [YTest, TTest] = calculateChance(netCell{nn},imdsClassTest,string(targ_category));
                for chanceIdx = 1:shuffleNum
                    shuffle_TTest = TTest;
                    shuffle_TTest = shuffle_TTest(randperm(numel(shuffle_TTest)));
                    for target = 1:size(freqVarPerformanceCell,2)
                        T_index = find(shuffle_TTest == targ_category(target));
                        shuffleAccCell(1,target,chanceIdx,trial,nn) = numel(find(YTest(T_index)==targ_category(target)))/numel(T_index);
                    end
                    clear target shuffle_TTest
                end
            end
        end
        clear nn
    
        %% Training loop
        trainOpt = [numEpochs, miniBatchSize, learnRate];

        % Items are trained sequentially
        for trainIdx = 1:numel(trainOrder)
            currClsIdx = trainOrder(trainIdx);
            
            % Permute the labels of 'non-target' items
            clear tempImdsTrain
            tempImdsTrain = copy(imdsTrain);
            
            % Create a shuffled set of non-target labels
            clear clsImdIdx nclsImdIdx nClsLab noncurrClsLab addIdx
            clsImdIdx = trainImdIdx{currClsIdx};
            noncurrClsIdx = setdiff(1:numCategories,currClsIdx);
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
            clsImdIdx = find(tempImdsTrain.Labels==targ_category(currClsIdx));
            nclsImdIdx = setdiff(1:numel(tempImdsTrain.Labels),clsImdIdx);
            
            % Re-label the non-target items
            tempImdsTrain.Labels(nclsImdIdx) = categorical(targ_category(noncurrClsLab));
            tempImdsTrain.Labels = removecats(tempImdsTrain.Labels,setdiff(total_category,targ_category));
            tempImdsTrain.Labels = reordercats(tempImdsTrain.Labels,targ_category);
            
            % Train different networks parallely
            for nn = 1:nwNum
                netCell{nn} = trainModel(trainOpt,netCell{nn},initNet,tempImdsTrain,flexibility(nn,:),s,FClayersIdx,epochwiseTest);
            end

            % Measure the performance of networks after training an item
            for nn = 1:nwNum
                freqVarPerformanceCell(trainIdx+1,:,trial,nn) = testModel(netCell{nn},imdsClassTest,string(targ_category));
                if verbal
                    if nn == 1; disp('Conventional network performance: ');
                    else if nn == 2; disp('Hybrid network performance: ');
                    else if nn == 3; disp('Stable network performance: ');
                    end; end; end
                    disp(freqVarPerformanceCell(1:trainIdx+1,:,trial,nn))
                end
                if calculateShuffleAccuracy
                    % Calculate shuffled-accuracy
                    [YTest, TTest] = calculateChance(netCell{nn},imdsClassTest,string(targ_category));
                    for chanceIdx = 1:shuffleNum
                        shuffle_TTest = TTest;
                        shuffle_TTest = shuffle_TTest(randperm(numel(shuffle_TTest)));
                        for target = 1:size(freqVarPerformanceCell,2)
                            T_index = find(shuffle_TTest == targ_category(target));
                            shuffleAccCell(trainIdx+1,target,chanceIdx,trial,nn) = numel(find(YTest(T_index)==targ_category(target)))/numel(T_index);
                        end
                        clear target shuffle_TTest
                    end
                end
            end
            if verbal; disp(['Class-' num2str(trainIdx) '-trial-' num2str(trial) '-running']); end
            clear label tempImdsTrain imdIdx tempSeq
        end
        
        if saveLastNetwork; save(loadPath+"results\networks\learning-frequency-varying-networks-trial-"+num2str(trial),"initNet","netCell","flexibility"); end
    end
end

if calculateShuffleAccuracy
    save(strcat(loadPath,"results\learning-frequency-varying.mat"),"freqVarPerformanceCell","shuffleAccCell","freqVarCategoryCell","trainOrderList");
    save(strcat(loadPath,"results\info\info-learning-frequency-varying.mat"),"numTrial","numCategories","numEpochs","miniBatchSize","learnRate","s","nwNum","shuffleNum");
else
    save(strcat(loadPath,"results\learning-frequency-varying.mat"),"freqVarPerformanceCell","freqVarCategoryCell","trainOrderList");
    save(strcat(loadPath,"results\info\info-learning-frequency-varying.mat"),"numTrial","numCategories","numEpochs","miniBatchSize","learnRate","s","nwNum");
end
