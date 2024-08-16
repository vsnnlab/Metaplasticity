function X = preprocessMiniBatchPredictors(dataX)

% Concatenate.
X = cat(4,dataX{1:end});

end
