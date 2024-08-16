% This function simply resizes the images to fit in AlexNet
% Copyright 2017 The MathWorks, Inc.
function dataOut = imgAugmentation(filename)
% Resize the images to the size required by the network.
data = imread(filename);
data = imresize(data, [227 227]);

if size(data,3)<3
    dataOut = zeros(227,227,3);
    
    for ii = 1:3
        dataOut(:,:,ii) = data;
    end
else; dataOut = data;
end
end
