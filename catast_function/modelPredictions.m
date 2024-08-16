function Y = modelPredictions(net,mbq,classes)

Y = [];

    % Loop over mini-batches.
    while hasdata(mbq)
        X = next(mbq);
    
        % Make prediction.
        scores = predict(net,X);
    
        % Decode labels and append to output.
        labels = onehotdecode(scores,classes,1);
        Y = [Y, labels];
    end
end