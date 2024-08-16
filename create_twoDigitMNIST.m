addpath('catast_function\')

oldpath = addpath(fullfile(strcat(pwd,"\MNIST_raw")));
newpath = 'twoDigitMNIST\';

%% Load MNIST files
filenameImagesTrain = 'train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'train-labels-idx1-ubyte.gz';
filenameImagesTest = 't10k-images-idx3-ubyte.gz';
filenameLabelsTest = 't10k-labels-idx1-ubyte.gz';

XrawTrain = processImagesMNIST(filenameImagesTrain);
YTrain = processLabelsMNIST(filenameLabelsTrain);

XrawTest = processImagesMNIST(filenameImagesTest);
YTest = processLabelsMNIST(filenameLabelsTest);

%%
cat = []; % possible two-digit catagories
for ii = 0:8
    for jj = ii:9
        cat = [cat string(strcat(num2str(ii),num2str(jj)))];
    end
end
label = string(cat);

%% Create training dataset
numSamp = 5000;

idxDigit = cell(1,10);
for digit = 0:9
    idxDigit{digit+1} = find(YTrain == digit);
end

for nth = 1:numel(label)
    t = char(label(nth));
    if ~exist([newpath,'training\',t], 'dir')
        mkdir([newpath,'training\',t])
        
        if length(t) == 1; ii = 0; jj = str2num(t(1));
        else ii = str2num(t(1)); jj = str2num(t(2));
        end

        for kk = 1:numSamp
            dataOut = zeros(56,56);
            iIdx = idxDigit{ii+1}(randi(numel(idxDigit{ii+1})));
            jIdx = idxDigit{jj+1}(randi(numel(idxDigit{jj+1})));

            t1 = XrawTrain(:,:,1,iIdx);
            t2 = XrawTrain(:,:,1,jIdx);

            dataOut(15:42,1:28) = reshape(t1(:),28,28);
            dataOut(15:42,29:56) = reshape(t2(:),28,28);
            imwrite(dataOut, strcat(newpath,'training\',string(t),"\",num2str((nth-1)*numSamp+kk),'.png'));
        end
    end
    clear kk
end

%% Create test dataset
numSamp = 1000;

idxDigit = cell(1,10);
for digit = 0:9
    idxDigit{digit+1} = find(YTest == digit);
end

for nth = 1:numel(label)
    t = char(label(nth));
    if ~exist([newpath,'testing\',t], 'dir')
        mkdir([newpath,'testing\',t])

        if length(t) == 1; ii = 0; jj = str2num(t(1));
        else; ii = str2num(t(1)); jj = str2num(t(2));
        end

        for kk = 1:numSamp
            dataOut = zeros(56,56);
            iIdx = idxDigit{ii+1}(randi(numel(idxDigit{ii+1})));
            jIdx = idxDigit{jj+1}(randi(numel(idxDigit{jj+1})));

            t1 = XrawTest(:,:,1,iIdx);
            t2 = XrawTest(:,:,1,jIdx);

            dataOut(15:42,1:28) = reshape(t1(:),28,28);
            dataOut(15:42,29:56) = reshape(t2(:),28,28);
            imwrite(dataOut, strcat(newpath,'testing\',string(t),"\",num2str((nth-1)*numSamp+kk),'.png'));
        end
        clear kk
    end
end