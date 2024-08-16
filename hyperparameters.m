numTrial = 10;
shuffleNum = 1000;

numEpochs = 5;
miniBatchSize = 1024;
learnRate = 1e-4;
s_item_10 = 4e5;
s_item_30 = 2e5;

verbal = 1;
saveLastNetwork = 1;

% For the Hebb repetition effect results
repTime = 9;

% For data poisoning attack results
dataPoisoningAttack = 1;
attackTargetNetwork = [1,2]; % Attack only the "conventional" and "hybrid" networks

% For learning frequency-dependent
numInitNetwork = 1;
numFreqProfile = 2;

save(strcat(loadPath,"hyperparameters.mat"),"numTrial","shuffleNum",...
    "numEpochs","miniBatchSize","learnRate","s_item_10","s_item_30",...
    "verbal","saveLastNetwork",...
    "repTime",...
    "dataPoisoningAttack","attackTargetNetwork",...
    "numInitNetwork","numFreqProfile");
