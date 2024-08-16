function [meanPerf,stdPerf] = calculateMeanStd(clsAccCell,numTrial,dim)

if (nargin<3); dim = 3; end
outputArrSize = size(clsAccCell); outputArrSize(dim) = [];

meanPerf = zeros(outputArrSize);
stdPerf = zeros(outputArrSize);

for nn = 1:outputArrSize(end)
   meanPerf(:,:,nn) = mean(clsAccCell(:,:,1:numTrial,nn),3);
   stdPerf(:,:,nn) = std(clsAccCell(:,:,1:numTrial,nn),0,3);
end

end

