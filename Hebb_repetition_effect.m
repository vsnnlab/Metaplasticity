%%%%%%%%%%%%%
% Code for generating "Hebb repetition effect" and "data poisoning attack" results (Fig 4)

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

load(strcar(loadPath,"hyperparameters.m"))
if numCategories == 10; s = s_item_10;
else if numCategories == 30; s = s_item_30; end; end

% Already have networks from "serial position effect" simulations
if ~exist(loadPath+"results\networks\serial-position-effect-networks-trial-"+num2str(numTrial), 'dir')
    disp("You have data from serial position effect simulation.")
    disp("In this case, you can use this data to start running this code from second repetition.")
    while true
        prompt = "Do you want to load the data you already have? Y/N [Y]: ";
        txt = input(prompt,"s");
        if isempty(txt)
            txt = 'Y';
        end
        if txt == 'Y'; loadNetwork = 1; break;
        else if txt == 'N'; loadNetwork = 0; break; 
        else; disp("You've entered a wrong input. Please try again"); continue;
        end; end
    end
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
performanceCell = zeros((repTime)*numCategories+1,numCategories,numTrial,nwNum); % precision of each class after training each item
categoryCell = strings(numTrial,(repTime)*numCategories); % trained catagories on each trial
if calculateShuffleAccuracy; shuffleAccCell = zeros((repTime)*numCategories+1,numCategories,shuffleNum,numTrial,nwNum); end

if loadNetwork == 1
    performanceCell(1:numCategories+1,:,:,:) = load(loadPath+"results\serial-position-effect.mat").performanceCell;
    shuffleAccCell(1:numCategories+1,:,:,:,:) = load(loadPath+"results\serial-position-effect.mat").shuffleAccCell;
    categoryCell(:,1:numCategories) = load(loadPath+"results\serial-position-effect.mat").categoryCell;
else
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
    
    initNet = dlnetwork(layers);
end

FClayersIdx = [17, 19, 21]; % Indices of fully connected layers

%%
for trial = 1:numTrial
    % Select categories to be trained on networks
    if loadNetwork == 1
        targ_category = load(loadPath+"results\serial-position-effect.mat").categoryCell;
        targ_category = targ_category(trial,:);
    else
        targ_category = total_category(randperm(numel(total_category),numCategories));
    end
    
    % Actual traninig sequence order; a sequence of items will be
    % repetitively trained multiple times.
    trainOrder = [];
    for loop = 1:repTime
        trainOrder = [trainOrder, 1:numCategories];
    end

    categoryCell(trial,:) = targ_category(trainOrder);
    
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

    if loadNetwork == 1
        initNet = load(loadPath+"results\networks\serial-position-effect-networks-trial-"+num2str(trial)).initNet;
        netCell = load(loadPath+"results\networks\serial-position-effect-networks-trial-"+num2str(trial)).netCell;
        flexibility = load(loadPath+"results\networks\serial-position-effect-networks-trial-"+num2str(trial)).flexibility;
    else
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
                if nn==1; flexibility{nn,jj} = ones(numW); end
                if nn==2; tempFlexibility = rand(numW); flexibility{nn,jj} = tempFlexibility; end % Here, the entire NW has random value       
                if nn==3; flexibility{nn,jj} = 0.3*ones(numW); end
            end
            clear numW tempFlexibility jj nn
        end
    end

    % If training from scratch, measure the initial precision of networks
    % before the training starts
    if loadNetwork == 0
        for nn = 1:nwNum
            performanceCell(1,:,trial,nn) = testModel(netCell{nn},imdsClassTest,targ_category);
            if verbal
                if nn == 1; disp('Conventional network performance: ');
                else if nn == 2; disp('Hybrid network performance: ');
                else if nn == 3; disp('Stable network performance: ');
                end; end; end
            end
            disp(performanceCell(1,:,trial,nn))
            
            if calculateShuffleAccuracy
                % Calculate shuffled-accuracy
                [YTest, TTest] = calculateChance(netCell{nn},imdsClassTest,targ_category);
                for chanceIdx = 1:shuffleNum
                    shuffle_TTest = TTest;
                    shuffle_TTest = shuffle_TTest(randperm(numel(shuffle_TTest)));
                    for target = (numCategories*(repTime-1)+1):size(performanceCell,2)
                        T_index = find(shuffle_TTest == targ_category(target));
                        shuffleAccCell(1,target,chanceIdx,trial,nn) = numel(find(YTest(T_index) == targ_category(target)))/numel(T_index);
                    end
                    clear target shuffle_TTest T_index
                end
            end
        end
        clear nn
    end
    
    %% Training loop
    trainOpt = [numEpochs, miniBatchSize, learnRate];
    
    if loadNetwork == 1; startIdx = numCategories+1;
    else; startIdx = 1; end
    
    % Items are trained sequentially
    for trainIdx = startIdx:numel(trainOrder)
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
            performanceCell(trainIdx+1,:,trial,nn) = testModel(netCell{nn},imdsClassTest,string(targ_category));
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
                    for target = (numCategories*(repTime-1)+1):size(performanceCell,2)
                        T_index = find(shuffle_TTest == targ_category(target));
                        shuffleAccCell(trainIdx+1,target,chanceIdx,trial,nn) = numel(find(YTest(T_index)==targ_category(target)))/numel(T_index);
                    end
                    clear target shuffle_TTest T_index
                end
                [memorizedCnt,memorizedIdx] = countMemorizedItems(numTrial, numCategories, repTime, performanceCell, shuffleAccCell);
            end
        end
        if verbal; disp(['Class-' num2str(trainIdx) '-trial-' num2str(trial) '-running']); end
        clear label tempImdsTrain imdIdx tempSeq
    end
    if calculateShuffleAccuracy
        [memorizedCnt,memorizedIdx] = countMemorizedItems(numTrial, numCategories, repTime, performanceCell, shuffleAccCell);
    end
    if saveLastNetwork; save(loadPath+"results\networks\Hebb-repetition-effect-networks-trial-"+num2str(trial),"initNet","netCell","flexibility"); end
    
    %% Data poisoning attack
    if dataPoisoningAttack
        attackPerformanceCell = zeros(numCategories,numCategories,numTrial,nwNum);
        if calculateShuffleAccuracy; attackShuffleAccCell = zeros(numCategories,numCategories,numTrial,shuffleNum,nwNum); end
        attackCategoryCell = zeros(trial,numCategories);

        % Create fake shuffled labels to create poisoned data
        while true
            fakeClsLab = randperm(numCategories);
            for cc = 1:numCategories
                if cc == fakeClsLab(cc); continue
                end
            end 
            break
        end
        attackCategoryCell(trial,:) = fakeClsLab;
        clear cc

        for trainIdx = 1:numCategories
            currClsIdx = trainOrder(trainIdx);
            fakeClsIdx = fakeClsLab(trainIdx);

            clear tempImdsTrain
            tempImdsTrain = copy(imdsTrain);
            
            % Create a mixed set of non-target labels
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
            tempImdsTrain.Labels(clsImdIdx) = categorical(targ_category(ones(1,numel(clsImdIdx))*fakeClsIdx));
            tempImdsTrain.Labels(nclsImdIdx) = categorical(targ_category(noncurrClsLab));
            tempImdsTrain.Labels = removecats(tempImdsTrain.Labels,setdiff(total_category,targ_category));
            tempImdsTrain.Labels = reordercats(tempImdsTrain.Labels,targ_category);
            

            % Train different networks parallely
            for nn = attackTargetNetwork
                netCell{nn} = trainModel(trainOpt,netCell{nn},initNet,tempImdsTrain,flexibility(nn,:),s,FClayersIdx,epochwiseTest);
            end
            
            % Measure the precision of networks after training an item
            for nn = attackTargetNetwork
                attackPerformanceCell(trainIdx,:,trial,nn) = testModel(netCell{nn},imdsClassTest,string(targ_category));
                if verbal
                    if nn == 1; disp('Conventional network performance: ');
                    else if nn == 2; disp('Hybrid network performance: ');
                    else if nn == 3; disp('Stable network performance: ');
                    end; end; end
                    disp(attackPerformanceCell(1:trainIdx,:,trial,nn))
                end
                if calculateAttackShuffleAccuracy
                    % Calculate shuffled-accuracy
                    [YTest, TTest] = calculateChance(netCell{nn},imdsClassTest,string(targ_category));
                    for chanceIdx = 1:shuffleNum
                        shuffle_TTest = TTest;
                        shuffle_TTest = shuffle_TTest(randperm(numel(shuffle_TTest)));
                        for target = 1:size(attackPerformanceCell,2)
                            T_index = find(shuffle_TTest == targ_category(target));
                            attackShuffleAccCell(trainIdx,target,chanceIdx,trial,nn) = numel(find(YTest(T_index)==targ_category(target)))/numel(T_index);
                        end
                        clear target shuffle_TTest
                    end
                end
            end
            if verbal; disp(['Data poisoning attack: Class-' num2str(trainIdx) '-trial-' num2str(trial) '-running']); end
            clear label tempImdsTrain imdIdx tempSeq
        end
    end
    if calculateAttackShuffleAccuracy
        [memorizedAttackCnt,memorizedAttackIdx] = countMemorizedItems(numTrial, numCategories, 1, attackPerformanceCell, attackShuffleAccCell);
    end
    if saveLastNetwork; save(loadPath+"results\networks\data-poisoning-attack-networks-trial-"+num2str(trial),"initNet","netCell","flexibility"); end
end

if calculateShuffleAccuracy
    save(strcat(loadPath,"results\Hebb-repetition-effect.mat"),"performanceCell","memorizedCnt","memorizedIdx","categoryCell");
    save(strcat(loadPath,"results\info\info-Hebb-repetition-effect.mat"),"numTrial","numCategories","repTime","numEpochs","miniBatchSize","learnRate","s","nwNum","shuffleNum");
else
    save(strcat(loadPath,"results\Hebb-repetition-effect.mat"),"performanceCell","categoryCell");
    save(strcat(loadPath,"results\info\info-Hebb-repetition-effect.mat"),"numTrial","numCategories","repTime","numEpochs","miniBatchSize","learnRate","s","nwNum");
end

if dataPoisoningAttack
    if calculateAttackShuffleAccuracy
    save(strcat(loadPath,"results\data-poisoning-attack.mat"),"attackPerformanceCell","memorizedAttackCnt","memorizedAttackIdx","attackCategoryCell","numCategories","numTrial");
    save(strcat(loadPath,"results\info\info-data-poisoning-attack.mat"),"numTrial","numCategories","repTime","numEpochs","miniBatchSize","learnRate","s","nwNum","shuffleNum");
    else
        save(strcat(loadPath,"results\data-poisoning-attack.mat"),"attackPerformanceCell","attackCategoryCell","numCategories","numTrial");
        save(strcat(loadPath,"results\info\info-data-poisoning-attack.mat"),"numTrial","numCategories","repTime","numEpochs","miniBatchSize","learnRate","s","nwNum");
    end
end
