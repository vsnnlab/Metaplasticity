function [memorizedCnt,memorizedIdx] = countMemorizedItems(numTrial, numCategories, targRep, performanceCell, shuffleAccCell)

memorizedCnt = zeros(size(performanceCell,4),size(performanceCell,1)-1,size(performanceCell,3));
memorizedIdx = cell(size(performanceCell,4),size(performanceCell,1)-1,size(performanceCell,3));

targP = 0.05;

for trial = 1:numTrial
    for nn = 1:3
        for rep = 1:targRep
            for targCat = 1:numCategories
                if rep == 1; idxRange = 1:targCat; else; idxRange = 1:numCategories; end
                for idx = idxRange
                    samp = performanceCell(numCategories*(rep-1)+targCat+1,idx,trial,nn);
                    chances = shuffleAccCell(numCategories*(rep-1)+targCat+1,idx,:,trial,nn);
                    if numel(find(chances>samp))/numel(chances) < targP
                        memorizedCnt(nn,numCategories*(rep-1)+targCat,trial) = memorizedCnt(nn,numCategories*(rep-1)+targCat,trial) + 1;
                        memorizedIdx{nn,numCategories*(rep-1)+targCat,trial} = [memorizedIdx{nn,numCategories*(rep-1)+targCat,trial}, idx];
                    end
                    clear samp chances
                end
            end
        end
    end
end

end

