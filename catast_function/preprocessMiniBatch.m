function [X,T] = preprocessMiniBatch(dataX,dataT)

% Preprocess predictors.
X = cat(4,dataX{1:end});

% Extract label data from cell and concatenate.
T = cat(2,dataT{1:end});

% One-hot encode labels.
T = onehotencode(T,1);

end