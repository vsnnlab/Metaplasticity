%% Generate plots with existing data

while true
    prompt = "Enter the number of the figure you want to plot: ";
    figNum = input(prompt,"n");
    if (figNum > 6 || figNum < 2); disp("You've entered a wrong input."); continue;
    else; break; end
end

stbColor = [162,89,51]/256; conventionalColor = [29,55,102]/256; hybridColor = [0.8 0 0]; chanceColor = [.7 .7 .7];
initX = 30; initY = 16; Xdistance = 7;


%% Figure 2
if figNum == 2
    load(loadPath+"results\serial-position-effect-10.mat")
    load(loadPath+"results\info\info-serial-position-effect-10.mat")
    
    % Fig 2d, g, j
    [epochMeanPerf,epochStdPerf] = calculateMeanStd(epochPerformanceCell,numTrial);
    lastItem = 7;
    x = 0:numEpochs*lastItem;
    lineColor = [230,44,138;150,150,150;150,150,150;150,150,150;...
        150,150,150;150,150,150;22,156,59]./256;
    
    for nn = [1,3,2]

        if nn == 1; figTitle = "Figure 2d";
        else if nn == 2; figTitle = "Figure 2j"; 
        else if nn == 3; figTitle = "Figure 2g";
        end; end; end

        figure('Units','centimeters','Position',[initX+nn*1 initY+nn*1 8.4 3.8]);

        subplot(1,3,1:2)
        hold on

        plot(0:lastItem*numEpochs+1,ones(1,lastItem*numEpochs+2)*1/numCategories,'Color',chanceColor,'LineStyle','--','LineWidth',2)
        for cc = 1:lastItem
            shaded_errorbar(epochMeanPerf(1:lastItem*numEpochs+1,cc,nn),epochStdPerf(1:lastItem*numEpochs+1,cc,nn),lineColor(cc,:),x,0);
        end
        
        xlim([0 lastItem*numEpochs]); ylim([0 1.1])
        ylabel('Performance'); xlabel('Training epoch');
        yticks(0:0.2:1.1);
        xticks(0:5:lastItem*numEpochs); xticklabels(["0"," ","10"," ","20"," ","30"," "])
        set(gca,'TickDir','out'); 
        
        subplot(1,3,3)
        hold on
        targetData = squeeze(performanceCell(lastItem*numEpochs+1,[1 lastItem],1:numTrial,nn))';
        % [h, p, ~, stat] = ttest(squeeze(performanceCell(lastItem*5+1,1,1:targTrial,nn)), 1/numCategories, "Tail", "right")
        % [h, p, ~, stat] = ttest(squeeze(performanceCell(lastItem*5+1,lastItem,1:targTrial,nn)), 1/numCategories, "Tail", "right")
        plot(0:3,1/numCategories*[1 1 1 1],'Color',chanceColor,'LineStyle','--','LineWidth',2)
        
        ax = axes();
        hold(ax);
        itemIdx = [1 lastItem];
        for ii = 1:numel(itemIdx)
            boxchart(ii*ones(size(targetData(:,ii))),targetData(:,ii),'BoxFaceColor',lineColor(itemIdx(ii),:))
        end
        ylim([0 1.1]); xlim([0.7 2.3]);
        ylabel('Performance');
        yticks(0:0.2:1.1); xticks([1 2]);
        xticklabels(["1st", lastItem+"th"])
        set(gca,'TickDir','out');
                
        sgtitle(figTitle)

    end
    clear lastItem x lineColor cc targetData itemIdx

    % Fig 2e, h, k
    [meanPerf,stdPerf] = calculateMeanStd(performanceCell,numTrial);
    
    for nn = [1,3,2]

        if nn == 1; figTitle = "Figure 2e"; lineColor = conventionalColor;
        else if nn == 2; figTitle = "Figure 2k"; lineColor = hybridColor;
        else if nn == 3; figTitle = "Figure 2h"; lineColor = stbColor;
        end; end; end

        figure('Units','centimeters','Position',[initX+Xdistance+nn*1 initY+nn*1 3.9 3.8]); hold on
        targTimepoint = numCategories*1+1;

        compareChance = zeros(1,numCategories);
        for cc = 1:numCategories
            compareChance(cc) = ttest(performanceCell(targTimepoint,cc,:,nn),1/numCategories,"Tail","right");
        end

        plot(0:numCategories,ones(1,numCategories+1)*1/numCategories,'Color',chanceColor,'LineStyle','--','LineWidth',2);
        shaded_errorbar(meanPerf(targTimepoint,1:numCategories,nn),stdPerf(targTimepoint,1:numCategories,nn),lineColor,1:numCategories,1,find(compareChance == 0));

        xlim([0 numCategories]); ylim([0 1.1])
        ylabel('Performance'); xlabel('Item order');
        yticks(0:0.2:1.1); xticks(0:2:numCategories)
        set(gca,'TickDir','out'); 

        title(figTitle)
    end
    clear meanPerf stdPerf cc compareChance figTitle

    % Fig 2f, i, l
    meanCnt = mean(memorizedCnt,3); stdCnt = std(memorizedCnt,0,3);

    for nn = [1,3,2]

        if nn == 1; figTitle = "Figure 2e"; lineColor = conventionalColor;
        else if nn == 2; figTitle = "Figure 2k"; lineColor = hybridColor;
        else if nn == 3; figTitle = "Figure 2h"; lineColor = stbColor;
        end; end; end

        figure('Units','centimeters','Position',[initX+2*Xdistance+nn*1 initY+nn*1 3.8 3.8]); hold on

        plot(1:numCategories,1:numCategories,'Color',chanceColor,'LineStyle','--','LineWidth',2);
        shaded_errorbar(meanCnt(nn,:),stdItems(nn,:),lineColor,x,1);

        xlim([0 numCategories]); ylim([0 numCategories])
        ylabel('N(memorized)'); xlabel('N(trained)');
        yticks(0:2:numCategories);
        xticks(0:2:numCategories);
        set(gca,'TickDir','out'); 
        
        title(figTitle)
    end
    clear meanCnt stdCnt nn figTitle lineColor
end

%% Figure 3
if figNum == 3
    load(loadPath+"results\serial-position-effect-30.mat")
    load(loadPath+"results\info\info-serial-position-effect-30.mat")
    
    % Fig 3b
    [meanShuff,stdShuff] = calculateMeanStd(shuffleAccCell,numTrial);

    figTitle = "Figure 3b";

    figure('Units','centimeters','Position',[initX initY 38 50]);
    
    sampleTrial = 1;
    
    subplot(2,1,1); hold on
    nn = 2;
    shaded_errorbar(meanShuff(numCategories+1,1:numCategories,sampleTrial,nn),stdShuff(numCategories+1,1:numCategories,sampleTrial,nn),...
        chanceColor,1:numCategories,0,[],'--');
    shaded_errorbar(performanceCell(numCategories+1,1:numCategories,sampleTrial,nn),zeros(1,numCategories,1,1),...
        hybridColor,1:numCategories,1,setdiff(1:numCategories,memorizedIdx{nn,numCategories,sampleTrial}));

    xlim([0 numCategories]); ylim([0 1])
    ylabel('Performance (Si)');
    yticks(0:0.5:1);
    xticks(0:10:numCategories);
    set(gca,'TickDir','out'); 
    title("Hybrid");

    subplot(2,1,2); hold on
    nn = 3;
    shaded_errorbar(meanShuff(numCategories+1,1:numCategories,sampleTrial,nn),stdShuff(numCategories+1,1:numCategories,sampleTrial,nn),...
        chanceColor,1:numCategories,0,[],'--');
    shaded_errorbar(performanceCell(numCategories+1,1:numCategories,sampleTrial,nn),zeros(1,numCategories,1,1),...
        conventionalColor,1:numCategories,1,setdiff(1:numCategories,memorizedIdx{nn,numCategories,sampleTrial}));

    xlim([0 numCategories]); ylim([0 1])
    ylabel('Performance (Si)'); xlabel('Item order (i)')
    yticks(0:0.5:1);
    xticks(0:10:numCategories);
    set(gca,'TickDir','out'); 
    title("Conventional")

    sgtitle(figTitle)

    clear meanShuff stdShuff sampleTrial nn figTitle

    % Fig 3c
    meanCnt = mean(memorizedCnt,3); stdCnt = std(memorizedCnt,0,3);
    
    targetPoint = [3, numCategories];

    figTitle = "Figure 3c";

    figure('Units','centimeters','Position',[initX+Xdistance initY 8.1 3.9]);
    subplot(1,2,1); hold on
    plot(1:numCategories,1:numCategories,'Color',chanceColor,'LineStyle','--','LineWidth',2);
    for nn = 1:2

        if nn == 1; lineColor = conventionalColor;
        else if nn == 2; lineColor = hybridColor;
        end; end
        
        shaded_errorbar(meanCnt(nn,:),stdItems(nn,:),lineColor,1:numCategories,0);
        plot(targetPoint(1),meanCnt(nn,targetPoint(1)),'o','Color',lineColor,'LineStyle','none');
        plot(targetPoint(2),meanCnt(nn,targetPoint(2)),'^','Color',lineColor,'LineStyle','none');

    end

    xlim([0 numCategories]); ylim([0 numCategories])
    ylabel('N(memorized)'); xlabel('N(trained)');
    yticks(0:10:numCategories);
    xticks(0:10:numCategories);
    set(gca,'TickDir','out');

    subplot(1,2,2); hold on

    nn = 2;
    errorbar([0.9,1.9],mean(squeeze(memorizedCnt(nn,targetPoint,:)),2),std(squeeze(memorizedCnt(nn,targetPoint,:)),0,2),'.','Color',hybridColor);
    plot(0.9,mean(squeeze(memorizedCnt(nn,targetPoint(1),:)),2),'o','Color',hybridColor,'LineStyle','none');
    plot(1.9,mean(squeeze(memorizedCnt(nn,targetPoint(2),:)),2),'^','Color',hybridColor,'LineStyle','none');

    nn = 1;
    errorbar([1.1,2.1],mean(squeeze(memorizedCnt(nn,targetPoint,:)),2),std(squeeze(memorizedCnt(nn,targetPoint,:)),0,2),'.','Color',conventionalColor);
    plot(1.1,mean(squeeze(memorizedCnt(nn,targetPoint(1),:)),2),'o','Color',conventionalColor,'LineStyle','none');
    plot(2.1,mean(squeeze(memorizedCnt(nn,targetPoint(2),:)),2),'^','Color',conventionalColor,'LineStyle','none');
    
    xlim([0.5 2.5]); ylim([0 1]);
    ylabel('Ratio'); xlabel('N(trained)');
    yticks(0:0.25:1);
    xticks([1,2]); xticklabels(string(targetPoint));
    set(gca,'TickDir','out'); 

    sgtitle(figTitle)

    clear meanCnt stdCnt nn lineColor figTitle targetPoint figTitle


    % Fig 3d
    figTitle = "Figure 3d";

    meanAvg = zeros(3,numCategories); stdAvg = zeros(3,numCategories);    
    for nn = 1:2
        for cc = 1:numCategories
            hbtarg = [];
            for ll = 1:numTrial
                data = squeeze(performanceCell(cc+1,1:cc,ll,nn));
                hbtarg = [hbtarg; data(:)];
            end
            clear data
            meanAvg(nn,cc) = mean(hbtarg,"all");
            stdAvg(nn,cc) = std(hbtarg,0,"all");
            clear hbtarg
        end
    end

    figure('Units','centimeters','Position',[initX+1*Xdistance initY 3.9 4.2]); hold on
    
    for nn = 1:2
        if nn == 1; lineColor = conventionalColor;
        else if nn == 2; lineColor = hybridColor;
        end; end

        shaded_errorbar(meanAvg(nn,1:end),stdAvg(nn,1:end),lineColor,1:numCategories,0);
    end

    ylim([0 1]); xlim([0 numCategories]);
    ylabel("Average performance"); xlabel("N(trained)");
    yticks(0:0.2:1); xticks(0:10:numCategories);
    set(gca,'TickDir','out'); 

    sgtitle(figTitle)

    clear meanAvg stdAvgnn cc lineColor figTitle


    % Fig 3e
    figTitle = "Figure 3e";

    figure('Units','centimeters','Position',[initX+1*Xdistance initY 3.9 4.2]);
    meanSum = zeros(3,numCategories); stdSum = zeros(3,numCategories);    
    for nn = 1:2
        for cc = 1:numCategories
            hbtarg = [];
            for ll = 1:numTrial
                data = squeeze(performanceCell(cc+1,1:cc,ll,nn));
                hbtarg = [hbtarg; sum(data(:),"all")];
            end
            clear data
            meanSum(nn,cc) = mean(hbtarg,"all");
            stdSum(nn,cc) = std(hbtarg,0,"all");
            clear hbtarg
        end
    end

    figure('Units','centimeters','Position',[39 42 190 16]); hold on
    
    for nn = 1:2
        if nn == 1; lineColor = conventionalColor;
        else if nn == 2; lineColor = hybridColor;
        end; end

        shaded_errorbar(meanSum(nn,1:end),stdSum(nn,1:end),lineColor,1:numCategories,0);
    end

    ylim([0 5]); xlim([0 numCategories]);
    ylabel("Gross memory"); xlabel("N(trained)");
    yticks(0:1:5); xticks(0:10:numCategories);
    set(gca,'TickDir','out'); 

    sgtitle(figTitle)

    clear meanSum stdSum nn cc figTitle

end

%% Figure 4
if figNum == 4
    load(loadPath+"results\Hebb-repetition-effect.mat")
    load(loadPath+"results\info\info-Hebb-repetition-effect.mat")

    % Fig 4c
    [meanPerf,stdPerf] = calculateMeanStd(performanceCell,numTrial);

    figTitle = "Figure 4c";

    figure('Units','centimeters','Position',[initX initY 6.6 3.9]);

    repInterest = [1,3,6,9];

    for nn = 1:2
        subplot(1,2,nn); hold on
        if nn == 1; baseColor = conventionalColor;
        else if nn == 2; baseColor = hybridColor;
        end; end

        lineColor = [linspace(0,baseColor(1),numel(repInterest))',linspace(0,baseColor(2),numel(repInterest))',linspace(0,baseColor(3),numel(repInterest))'];
        
        plot(0:numCategories,ones(1,numCategories+1)*1/numCategories,'Color',chanceColor,'LineStyle','--','LineWidth',2);
        rr=0;
        for rep = repInterest
            rr = rr+1;
            shaded_errorbar(meanPerf(numCategories*rep+1,:,nn),stdPerf(numCategories*rep+1,:,nn),lineColor(rr,:),1:numCategories,0);
        end
        clear rr

        ylim([0 1]); xlim([0 numCategories]);
        ylabel("Item order"); xlabel("Performance");
        yticks(0:0.2:1); xticks(0:2:numCategories);
        set(gca,'TickDir','out'); 
    
    end
    sgtitle(figTitle)

    clear baseColor lineColor rr

    % Fig 4d
    figTitle = "Figure 4d";

    figure('Units','centimeters','Position',[36 39 initX+Xdistance initY]); hold on
    
    repInterest = 9;

    minData = squeeze(min(performanceCell(numCategories*(1:repInterest)+1,:,:,:),[],2));
    minMean = squeeze(mean(minData,2)); minStd = squeeze(std(minData,0,2));

    for nn = 1:2
        if nn == 1; lineColor = conventionalColor;
        else if nn == 2; lineColor = hybridColor;
        end; end

        plot(1:repInterest,squeeze(mean(meanAcc(numCategories*(1:repInterest)+1,:,nn),2)),'Color',lineColor,'LineStyle','--');
        shaded_errorbar(minMean(:,nn),minStd(:,nn),lineColor,1:repInterest,1);

    end
    
    xlim([0.7 repInterest+0.3]); ylim([0 0.5]);
    xticks(1:repInterest); yticks(0:0.1:0.5);
    ylabel("Performance"); xlabel("Repeated trial"); set(gca,'TickDir','out');
    
    title(figTitle)

    clear minData minMean nn lineColor

    % Fig 4f
    load(loadPath+"results\data-poisoning-attack")
    load(loadPath+"results\info\info-data-poisoning-attack")
    
    [meanAttackPerf,stdAttackPerf] = calculateMeanStd(attackPerformanceCell,numTrial);

    figTitle = "Figure 4f";
    figure('Units','centimeters','Position',[initX+2*Xdistance initY 3.6 3.9]); hold on
    
    for nn = 1:2

        if nn == 1; lineColor = conventionalColor;
        else if nn == 2; lineColor = hybridColor;
        end; end

        targTimepoint = numCategories*1;

        compareChance = zeros(1,numCategories);
        for cc = 1:numCategories
            compareChance(cc) = ttest(attackPerformanceCell(targTimepoint,cc,:,nn),1/numCategories,"Tail","right");
        end

        plot(0:numCategories,ones(1,numCategories+1)*1/numCategories,'Color',chanceColor,'LineStyle','--','LineWidth',2);
        shaded_errorbar(meanAttackPerf(targTimepoint,1:numCategories,nn),stdAttackPerf(targTimepoint,1:numCategories,nn),lineColor,1:numCategories,1,find(compareChance == 0));

    end

    xlim([0 numCategories]); ylim([0 1])
    ylabel('Performance'); xlabel('Item order');
    yticks(0:0.2:1); xticks(0:2:numCategories)
    set(gca,'TickDir','out');

    title(figTitle)

    clear meanAttackPerf stdAttackPerf nn targetTimepoint compareChance cc


    % Fig 4g
    figTitle = "Figure 4g";

    avgData = squeeze(mean(attackPerformanceCell(end,:,:,:),2));

    figure('Units','centimeters','Position',[36 39 initX+4*Xdistance initY]); hold on
    
    plot(0:3,numCategories*[1 1 1],'Color',chanceColor,'LineStyle','--','LineWidth',2);
    ax = axes();
    hold(ax);
    for nn = 1:2
        if nn == 1; lineColor = conventionalColor;
        else if nn == 2; lineColor = hybridColor;
        end; end
        boxchart(nn*ones(size(avgData(:,nn))),avgData(:,nn),'BoxFaceColor',lineColor)
    end
    
    xlim([.5 2.5]); ylim([0 0.5])
    ylabel('Performance');
    yticks(0:0.1:0.5); xticks([1 2])
    xticklabels("Conventional","Hybrid")
    set(gca,'TickDir','out');

    title(figTitle)

    clear avgData ax nn lineColor figTitle

    % Fig 4h
    figTitle = "Figure 4h";
    
    memorizedCntData = squeeze(memorizedAttackCnt(nn,end,:))';
    figure('Units','centimeters','Position',[initX+5*Xdistance initY 3.6 3.9]); hold on
    
    ax = axes();
    hold(ax);
    for nn = 1:2
        if nn == 1; lineColor = conventionalColor;
        else if nn == 2; lineColor = hybridColor;
        end; end
        boxchart(nn*ones(size(memorizedCntData(:,nn))),memorizedCntData(:,nn),'BoxFaceColor',lineColor)
    end
    
    xlim([.5 2.5]); ylim([0 numCategories])
    ylabel('N(memorized)');
    yticks(0:1:numCategories); xticks([1 2])
    xticklabels("Conventional","Hybrid")
    set(gca,'TickDir','out');

    title(figTitle)

    clear memorizedCntData ax nn lineColor figTitle

end

%% Figure 5
if figNum == 5
    load(loadPath+"results\learning-frequency-varying.mat")
    load(loadPath+"results\info\info-learning-frequency-varying.mat")
    
    itemOrder = zeros(numTrial,numCategories);
    clsOrderAcc = zeros(size(performanceCell,2),size(performanceCell,3),size(performanceCell,4));
    
    for tt = 1:numTrial
        for cc = 1:numCategories
            idxOfItem = find(trainOrderList(tt,:)==cc);
            S = sort(idxOfItem);
            itemOrder(tt,cc) = S(end);
            clear S cc
        end
        [~,I] = sort(itemOrder(tt,:));
        for nn = 1:size(performanceCell,4)
            clsOrderAcc(:,tt,nn) = performanceCell(end,I,tt,nn);
        end
        clear tt
    end
    
    % Fig 5b
    figTitle = "Figure 5b";
    sampleTrial = 20;
    [~,I] = sort(itemOrder(sampleTrial,:));

    catInfo = trainOrderList(sampleTrial,:);
    sampleFrequency = zeros(1,numCategories);
    for cc = 1:numCategories
        sampleFrequency(cc) = sum(catInfo==cc);
    end

    figure('Units','centimeters','Position',[47 39 initX+Xdistance initY]); hold on
    
    % yyaxis left
    for nn = 1:2
        if nn == 1; lineColor = conventionalColor;
        else if nn == 2; lineColor = hybridColor;
        end; end
        x = 1:numCategories; y = performanceCell(end,I,sampleTrial,nn);
        plot(x,y,'color',lineColor);
        c = polyfit(x,y,1); xfit = 0:0.1:numCategories; yfit = polyval(c,xfit);
        plot(x,y,'-',xfit,yfit,'--','Color',lineColor)
        % [r,p] = corr(x',y', 'Type','Spearman');
    end
    
    ylabel('Performance'); yticks(0.0:0.2:1);
    
    yyaxis right
    bar(1:numCategories,sampleFrequency(I),'LineStyle',"none",'FaceColor',[.8 .8 .8],'FaceAlpha',0.2);
    ylabel('Frequency'); yticks(0:2:10); ylim([0 11])
    xlabel('Item order'); xticks(2:2:10);xlim([0 11])
    set(gca,'TickDir','out');

    ax = gca;
    ax.YAxis(1).Color = [0 0 0]; ax.YAxis(2).Color = [0 0 0];
    ax.YAxis(1).FontSize = 6; ax.YAxis(2).FontSize = 6;
    ax.YAxis(1).LineWidth = .5; ax.YAxis(2).LineWidth = .5;
    ax.XAxis.FontSize = 6; ax.XAxis.LineWidth = .5;
    ax.TickDir = 'out';

    title(figTitle)
    
    
    % Fig. 5c
    figTitle = "Figure 5c";
    % Calculate correlation between the learning frqeuency and the order
    freqTable = zeros(numTrial, numCategories);
    clsRepAcc = zeros(size(freqVarPerformanceCell,1),size(freqVarPerformanceCell,2),numTrial,size(freqVarPerformanceCell,4));
    
    corrFreq = zeros(numTrial,size(freqVarPerformanceCell,4));
    
    for trial = 1:numTrial
        catInfo = trainOrderList(trial,:); temp = zeros(1,numCategories);
    
        for cc = 1:numCategories
            temp(cc) = sum(catInfo==cc);
        end
        
        [~, freqTable(trial,:)] = sort(temp);
        clsRepAcc(:,:,trial,:) = freqVarPerformanceCell(:,freqTable(trial,:),trial,:);
        for nn = 1:size(freqVarPerformanceCell,4)
            corrFreq(trial,nn) = corr((1:numCategories)', clsRepAcc(end,:,trial,nn)', 'Type','Spearman');
        end
        clear I cc catInfo temp
    end

    figure('Units','centimeters','Position',[initX+Xdistance initY 3.9 3.8]); hold on;
    data = clsRepAcc(end,:,:,2);
    s = scatter(repmat((1:10)-0.1,1,numTrial),data(:),2,conventionalColor,"filled");
    s.MarkerFaceAlpha = 0.2;
    data = clsRepAcc(end,:,:,1);
    s = scatter(repmat((1:10)+0.1,1,numTrial),data(:),2,[1 0 0],"filled");
    s.MarkerFaceAlpha = 0.2;
    xlabel('Frequency'); ylabel('Performance');
    xticks(0:2:10); yticks(0:.2:1);
    xlim([0 10])
    set(gca,'TickDir','out');
    
    clear data
    
    figure('Units','centimeters','Position',[initX+2*Xdistance initY 6.0 3.9]);

    subplot(1,2,1); hold on;
    nn=2;
    xx = repmat(1:numCategories,numTrial,1)'; yy = squeeze(clsOrderAcc(:,1:numTrial,nn));
    [c_conv,p_conv] = corr(xx,yy,'Type','Spearman');
    c_conv = c_conv(1,:); p_conv = p_conv(1,:);
    clear xx yy
    
    nn=1;
    xx = repmat(1:numCategories,numTrial,1)'; yy = squeeze(clsOrderAcc(:,1:numTrial,nn));
    [c_hybrid,p_hybrid] = corr(xx,yy,'Type','Spearman');
    c_hybrid = c_hybrid(1,:); p_hybrid = p_hybrid(1,:);
    clear xx yy

    corrData = [c_conv;c_hybrid]';

    ax = axes();
    hold(ax);
    for nn = 1:2
        if nn == 1; lineColor = conventionalColor;
        else if nn == 2; lineColor = hybridColor;
        end; end
        boxchart(nn*ones(size(corrData(nn,:))),corrData(nn,:),'BoxFaceColor',lineColor)
    end

    hybIdx = find(p_hybrid<0.05); convIdx = find(p_conv<0.05);
    scatter(1.7*ones(numel(hybIdx),1),c_hybrid(hybIdx),'Marker','o','MarkerFaceAlpha',.1,'MarkerFaceColor','k','MarkerEdgeAlpha',0);
    scatter(1.7*ones(numel(find(p_hybrid>=0.05)),1),c_hybrid(find(p_hybrid>=0.05)),'Marker','+','MarkerFaceAlpha',0,'MarkerEdgeColor','k','MarkerEdgeAlpha',0.5);
    
    scatter(0.7*ones(numel(convIdx),1),c_conv(convIdx),'Marker','o','MarkerFaceAlpha',.1,'MarkerFaceColor','k','MarkerEdgeAlpha',0);
    scatter(0.7*ones(numel(find(p_conv>=0.05)),1),c_conv(find(p_conv>=0.05)),'Marker','+','MarkerFaceAlpha',0,'MarkerEdgeColor','k','MarkerEdgeAlpha',0.5);

    ylabel('Correlation')
    xlim([.5 2.5]); ylim([-0.6 1.1]);
    xticks([1 2]); yticks(-0.5:0.5:1.0);
    xticklabels("Conventional","Hybrid");

    title("Order vs. Performance")


    subplot(1,2,2); hold on;
    nn = 1;
    xx = repmat(1:numCategories,numTrial,1)'; yy = squeeze(clsRepAcc(end,:,1:numTrial,nn));
    [c_conv,p_conv] = corr(xx,yy);
    c_conv = c_conv(1,:); p_conv = p_conv(1,:);
    clear xx yy
    
    nn = 2;
    xx = repmat(1:numCategories,numTrial,1)'; yy = squeeze(clsRepAcc(end,:,1:numTrial,nn));
    [c_hybrid,p_hybrid] = corr(xx,yy);
    c_hybrid = c_hybrid(1,:); p_hybrid = p_hybrid(1,:);
    clear xx yy

    corrData = [c_conv;c_hybrid]';

    ax = axes();
    hold(ax);
    for nn = 1:2
        if nn == 1; lineColor = conventionalColor;
        else if nn == 2; lineColor = hybridColor;
        end; end
        boxchart(nn*ones(size(corrData(nn,:))),corrData(nn,:),'BoxFaceColor',lineColor)
    end

    hybIdx = find(p_hybrid<0.05); convIdx = find(p_conv<0.05);
    scatter(1.7*ones(numel(hybIdx),1),c_hybrid(hybIdx),'Marker','o','MarkerFaceAlpha',.1,'MarkerFaceColor','k','MarkerEdgeAlpha',0);
    scatter(1.7*ones(numel(find(p_hybrid>=0.05)),1),c_hybrid(find(p_hybrid>=0.05)),'Marker','+','MarkerFaceAlpha',0,'MarkerEdgeColor','k','MarkerEdgeAlpha',0.5);
    
    scatter(0.7*ones(numel(convIdx),1),c_conv(convIdx),'Marker','o','MarkerFaceAlpha',.1,'MarkerFaceColor','k','MarkerEdgeAlpha',0);
    scatter(0.7*ones(numel(find(p_conv>=0.05)),1),c_conv(find(p_conv>=0.05)),'Marker','+','MarkerFaceAlpha',0,'MarkerEdgeColor','k','MarkerEdgeAlpha',0.5);

    ylabel('Correlation')
    xlim([.5 2.5]); ylim([-0.6 1.1]);
    xticks([1 2]); yticks(-0.5:0.5:1.0);
    xticklabels("Conventional","Hybrid");

    title("Frequency vs. Performance")
    
    sgtitle(figTitle)


end








%%%%%%%% Old plots

%% Memorized ratio
% [h, p, ~, stat] = ttest(squeeze(performanceCell(targCat*5+1,targCat,1:targTrial,nn)), 1/numCategories, "Tail", "right")

%% Figure 2. Memory performance in each epoch, iterAcc

%% Figure 2. Memory performance in each epoch, iterAcc - boxplot

%% Single item's precision in training timeline

%% Figure 2. Serial position - item order

%% Figure 3. memorized number of items, total item = 20 and 30

%% Figure 3d. Minimum performance in sequence for diff. number of trained items
meanMin = zeros(3,numCategories); stdMin = zeros(3,numCategories);

for cc = 1:numCategories
    hbtarg = squeeze(min(squeeze(performanceCell(cc+1,memorizedCntIdx{:,cc,:},1:numTrial,:)),[],1));
    meanMin(:,cc) = mean(hbtarg,1)';
    stdMin(:,cc) = std(hbtarg,0,1)';
end


figure; hold on; x=2:numCategories;
% nn=1;
% shaded_errorbar(meanMin(nn,2:end),stdMin(nn,2:end),stbColor,x,0);
% plot(1:numCategories,ones(1,numCategories).*1/30,'--','LineWidth',1,'Color',[0 0 0]);
nn=3;
shaded_errorbar(meanMin(nn,2:end),stdMin(nn,2:end)./sqrt(numCategories),conventionalColor,x,0);
nn=2;
shaded_errorbar(meanMin(nn,2:end),stdMin(nn,2:end)./sqrt(numCategories),[1 0 0],x,0);
xlabel("Total number of items in sequence"); ylabel("min. performance"); xlim([0 30]); ylim([0.0003 1]);
yticks([0.001 0.01 0.1 1.0])
set(gca, 'YScale', 'log');
set(gca,'TickDir','out');

%% Figure 3. Avg performance of the memorized items

%% Scatter plot of avg. performance vs. memorized item tradeoff
meanRes = mean(memorizedCnt,3);

figure; hold on
nn=3;
for cc = 1:numCategories
%     tempcolor = [unstbColor(1)-(cc-1)*(unstbColor(1))/(numCategories-1),...
%         unstbColor(2)-(cc-1)*(unstbColor(2))/(numCategories-1),...
%         unstbColor(3)-(cc-1)*(unstbColor(3))/(numCategories-1)];
%     endColor = [0 .8 .2];
    endColor = [0 0.2 0.4];
    tempcolor = [((cc-1)*endColor(1))/(numCategories-1),...
        ((cc-1)*endColor(2))/(numCategories-1),...
        ((cc-1)*endColor(3))/(numCategories-1)];
    scatter(meanRes(nn,cc),meanAvg(nn,cc),"filled",'MarkerFaceColor',tempcolor,'MarkerFaceAlpha',0.5);
end

nn=2;
for cc = 1:numCategories
%     endColor = [.9 .5 0];
    endColor = [1 0 0];
    scatter(meanRes(nn,cc),meanAvg(nn,cc),"filled",'MarkerFaceColor',...
        [((cc-1)*endColor(1))/(numCategories-1),...
        ((cc-1)*endColor(2))/(numCategories-1),...
        ((cc-1)*endColor(3))/(numCategories-1)],'MarkerFaceAlpha',0.5);
end

set(gca,'TickDir','out'); xticks(0:5:30)

%% Fig 3., accuracy capacity tradeoff, Sum of trained item performance
sumPerf = zeros(3,numCategories);
sumStd = zeros(3,numCategories);
for cc = 1:numCategories
    sumData = squeeze(sum(performanceCell(cc+1,1:cc,:,:),2));
    sumPerf(:,cc) = squeeze(mean(sumData,1))';
    sumStd(:,cc) = squeeze(std(sumData,0,1))';
end
figure; hold on; shaded_errorbar(sumPerf(2,:)./(1:30),sumStd(2,:)./(1:30),[1 0 0],1:numCategories,0);
shaded_errorbar(sumPerf(3,:)./(1:30),sumStd(3,:)./(1:30),conventionalColor,1:numCategories,0);
set(gca,'TickDir','out');

%% Fig 3., accuracy capacity tradeoff, Sum of memorized item performance
memorizedPerf = zeros(30,100,3);
for nn = 1:3; for cc = 1:30; for tt = 1:100
            memorizedPerf(cc,tt,nn) = squeeze(sum(performanceCell(cc+1,memorizedIdx{nn,cc,tt},tt,nn),2));
end;end;end

for nn = 1:3; for cc = 1:30; for tt = 1:100
            memorizedPerf(cc,tt,nn) = squeeze(sum(performanceCell(cc+1,1:cc,tt,nn),2));
end;end;end

figure; hold on;
shaded_errorbar(mean(memorizedPerf(:,:,3),2),std(memorizedPerf(:,:,3),0,2),conventionalColor, 1:30, 0);
shaded_errorbar(mean(memorizedPerf(:,:,2),2),std(memorizedPerf(:,:,2),0,2),[1 0 0], 1:30, 0);
ylim([0 5]); set(gca,'TickDir','out');

[h,p] = ttest2(squeeze(memorizedPerf(end,:,2)),squeeze(memorizedPerf(end,:,3)))

%% Fit the tradeoff data with y=1/x
fitX = meanRes; fitY = meanAvg;
% fitX = [1:numCategories;1:numCategories;1:numCategories];
% fitY = zeros(size(fitX));

nn = 2;
ft = fittype('a*(x-b)^(-1)+c');
[curve,info] = fit(fitX(nn,:)',fitY(nn,:)',ft);
% Plot the results
x = 1:numCategories;
figure; hold on; plot(fitX(nn,:)',fitY(nn,:)','o'); plot(x,(curve.a)./(x-curve.b)+curve.c);
xlim([0 30]); ylim([0 1]); set(gca,'TickDir','out');

%% Bar plot for avg. performance at a specific N_item
targCls = 10;

figure; hold on
bar(meanAvg(2:3,targCls)); errorbar(meanAvg(2:3,targCls),stdAvg(2:3,targCls)./sqrt(numTrial/numberTrial),'Color','k');
ylim([0 1]); yticks(0:0.2:1.0)
set(gca,'TickDir','out');

%% Fig 3. Plot different number of items in one figure
addpath('catast_function')

itemCount = 1:30;
% Color = [46/256 100/256 176/256];
Color = conventionalColor;

f = figure; hold on
for cc = itemCount

        nn=3;
        if cc==30;shaded_errorbar(meanAcc(cc+1,1:cc,nn),stdAcc(cc+1,1:cc,nn),(cc-1)/numCategories*Color,1:cc,1);
        else; plot(meanAcc(cc+1,1:cc,nn),'Color',Color); end
        ylim([0 1])

    xlabel('Item order')
    ylabel('Performance')
%     xlim([0.5 30])
    ylim([0 1]); set(gca,'TickDir','out');
end


%% Fig 4e. U shape curve from different repetition
addpath('catast_function')

%% Fig 4. Single item's mean performance change across repetitions
repInterest = 4;

targCls = [1 15 30]; nn = 2;
figure; hold on
for cc = targCls
    shaded_errorbar(meanAcc(numCategories.*(1:repInterest)+1,cc,nn),stdAcc(numCategories.*(1:repInterest)+1,cc,nn),conventionalColor,1:repInterest,1)
end
yticks(0:0.2:1.0); ylim([0 1]); xlim([.5 repInterest]); xticks(0:1:repInterest);
set(gca,'TickDir','out');
clear cc

%% Fig 4. Anova - single item thru repetition
data = [];
targCls = 5; nn = 2; repInterest = 3;

for rr = 1:repInterest
    data = [data, squeeze(performanceCell(numCategories*rr+1,targCls,:,nn))];
end

anova(data)


%% Fig 4d. Std across items - barchart
addpath('catast_function')

repInterest = 10;
meanStd = zeros(3,repInterest); stdStd = zeros(3,repInterest);
temp = zeros(100,2,4);

for rr = 1:repInterest
    data = squeeze(std(performanceCell(numCategories*rr+1,:,:,:),0,2));
    meanStd(:,rr) = mean(data,1)'; stdStd(:,rr) = std(data,0,1)';
    temp(:,:,rr) = data(:,2:3);
end

x=1:repInterest;
figure; hold on
nn=3;
shaded_errorbar(meanStd(nn,:),stdStd(nn,:),conventionalColor,x,1);
nn=2;
shaded_errorbar(meanStd(nn,:),stdStd(nn,:),[1 0 0],x,1);
xticks(1:repInterest); yticks(0.05:0.05:0.35); yticklabels(0.05:0.05:0.35); ylim([0.05 0.35]); xlim([0.5 repInterest+.5])
set(gca,'TickDir','out');
ylabel("SD(performance)"); xlabel("Repetition")

stdData = squeeze(std(performanceCell(numCategories*(1:10)+1,:,:,3),0,2));
[~,p] = ttest2(stdData(1,:),stdData(10,:))

%% Fig 5. Freq vs. Perf. (Indiv.)

addpath('catast_function')

freqTable = zeros(numTrial, numCategories);
clsRepAcc = zeros(size(performanceCell,1),size(performanceCell,2),numTrial,size(performanceCell,4));

corrFreq = zeros(numTrial,size(performanceCell,4));

for trial = 1:numTrial
    catInfo = trainOrderList(trial,:); temp = zeros(1,numCategories);

    for cc = 1:numCategories
        temp(cc) = sum(catInfo==cc);
    end
    
    [~, freqTable(trial,:)] = sort(temp);
    clsRepAcc(:,:,trial,:) = performanceCell(:,freqTable(trial,:),trial,:);
    for nn = 1:size(performanceCell,4)
        corrFreq(trial,nn) = corr([1:numCategories]', clsRepAcc(end,:,trial,nn)', 'Type','Spearman');
    end
    clear I cc catInfo temp
end

meanRepAcc = squeeze(mean(clsRepAcc,3));
stdRepAcc = squeeze(std(clsRepAcc,0,3));

targTrial  = 4;
figure('Units','centimeters','Position',[10 10 3.9 3.8]); hold on; x = 1:numCategories;
y = clsRepAcc(end,:,targTrial,2);
c = polyfit(x,y,1); xfit = 0:0.1:numCategories; yfit = polyval(c,xfit);
plot(x,y,'-',xfit,yfit,'--','Color',conventionalColor)

y = clsRepAcc(end,:,targTrial,1);
c = polyfit(x,y,1); xfit = 0:0.1:numCategories; yfit = polyval(c,xfit);
plot(x,y,'-',xfit,yfit,'--','Color',[1 0 0])

xlabel('Frequency')
ylabel('Performance')
xlim([1 numCategories])
ylim([0 1])

[r,p] = corr([1:10]',clsRepAcc(end,:,targTrial,1)', 'Type','Spearman')
[r,p] = corr([1:10]',clsRepAcc(end,:,targTrial,2)', 'Type','Spearman')

figure('Units','centimeters','Position',[10 10 3.9 3.8]); hold on;
data = clsRepAcc(end,:,:,2);
s = scatter(repmat((1:10)-0.1,1,numTrial),data(:),2,conventionalColor,"filled");
s.MarkerFaceAlpha = 0.2;
data = clsRepAcc(end,:,:,1);
s = scatter(repmat((1:10)+0.1,1,numTrial),data(:),2,[1 0 0],"filled");
s.MarkerFaceAlpha = 0.2;
xlabel('Frequency'); ylabel('Performance');
xticks(0:2:10); yticks(0:.2:1);
xlim([0 10])
set(gca,'TickDir','out');

clear data

%% Bar-Freq-Order plot
for targTrial = 20
    [~,I] = sort(itemOrder(targTrial,:));
    
    for tt = targTrial
        catInfo = trainOrderList(tt,:);
        temp = zeros(1,numCategories);
        for cc = 1:numCategories
            temp(cc) = sum(catInfo==cc);
        end
    end

    figure; hold on
    
    % yyaxis left
    nn = 2;
    x = 1:numCategories; y = performanceCell(end,I,targTrial,nn);
    plot(x,y,'color',conventionalColor);
%     shaded_errorbar(meanAcc(end,I,nn),stdAcc(end,I,nn),unstbColor,1:numCategories,0);
    c = polyfit(x,y,1); xfit = 0:0.1:numCategories; yfit = polyval(c,xfit);
    plot(x,y,'-',xfit,yfit,'--','Color',conventionalColor)
    [r,p] = corr(x',y', 'Type','Spearman');
    
    nn = 1;
    y = performanceCell(end,I,targTrial,nn);
    plot(x,y,'color',[1 0 0]);
%     shaded_errorbar(meanAcc(end,I,nn),stdAcc(end,I,nn),[1 0 0],1:numCategories,0);
    c = polyfit(x,y,1); xfit = 0:0.1:numCategories; yfit = polyval(c,xfit);
    plot(x,y,'-',xfit,yfit,'--','Color',[1 0 0])
    [r,p] = corr(x',y', 'Type','Spearman');
    ylabel('Performance'); yticks(0.0:0.2:1);
    
    yyaxis right
    bar(1:numCategories,temp(I),'LineStyle',"none",'FaceColor',[.8 .8 .8],'FaceAlpha',0.2);
    ylabel('Frequency','FontSize',7); yticks(0:2:10); ylim([0 11])
    
    xlabel('Item order'); xticks(2:2:10);
    
    xlim([0 11])
    
    set(gca,'TickDir','out');
    ax = gca;
    ax.YAxis(1).Color = [0 0 0]; ax.YAxis(2).Color = [0 0 0];
    ax.YAxis(1).FontSize = 6; ax.YAxis(2).FontSize = 6;
    ax.YAxis(1).LineWidth = .5; ax.YAxis(2).LineWidth = .5;
    ax.XAxis.FontSize = 6; ax.XAxis.LineWidth = .5;
    ax.TickDir = 'out';
    
end

%% Figure 5. Corr b/w Performance-Freq and Performance-CorrOrder


meanCorrFreq = mean(corrFreq,1);
stdCorrFreq = std(corrFreq,0,1);
meanCorrOrder = mean(corrOrder,1);
stdCorrOrder = std(corrOrder,0,1);

figure('Position', [100 100 50 180]); hold on;
% bar(1:2, meanCorrFreq(2:3));
% er = errorbar(1:2,meanCorrFreq(2:3),stdCorrFreq(2:3));er.Color = [0 0 0];er.LineStyle = 'none';
% for tt = 1:numTrial
%     plot(1:2,corrFreq(tt,2:3),'Marker','.','Color',[0 0 0 .5])
% end
boxplot(corrFreq(:,1:2));
xticks(1:2); yticks(0:0.2:1); set(gca,'TickDir','out'); ylim([-0.1 1.1])
ylabel("Correlation"); xticklabels(["Hybrid","Conventional"])

figure('Position', [200 200 50 180]); hold on;
% bar(1:2, meanCorrOrder(2:3));
% er = errorbar(1:2,meanCorrOrder(2:3),stdCorrOrder(2:3));er.Color = [0 0 0];er.LineStyle = 'none';
% for tt = 1:numTrial
%     plot(1:2,corrOrder(tt,2:3),'Marker','.','Color',[0 0 0 .5])
% end
boxplot(corrOrder(:,1:2));
xticks(1:2); yticks(0:0.2:1); set(gca,'TickDir','out'); ylim([-0.1 1.1])
ylabel("Correlation"); xticklabels(["Hybrid","Conventional"])

%% corrFreq vs. corrOrder - mean + std
figure('Position', [10 10 200 200]); hold on
nn=3;
plot(meanCorrFreq(nn),meanCorrOrder(nn),'o','Color',conventionalColor)
er = errorbar(meanCorrFreq(nn),meanCorrOrder(nn),stdCorrFreq(nn),'horizontal');er.Color = conventionalColor;er.LineStyle = 'none';
er = errorbar(meanCorrFreq(nn),meanCorrOrder(nn),stdCorrOrder(nn));er.Color = conventionalColor;er.LineStyle = 'none';
nn=2;
plot(meanCorrFreq(nn),meanCorrOrder(nn),'o','Color',[1 0 0])
er = errorbar(meanCorrFreq(nn),meanCorrOrder(nn),stdCorrFreq(nn),'horizontal');er.Color = [1 0 0];er.LineStyle = 'none';
er = errorbar(meanCorrFreq(nn),meanCorrOrder(nn),stdCorrOrder(nn));er.Color = [1 0 0];er.LineStyle = 'none';
plot([0 1],[0 1],'--','Color',[0 0 0]);
xlim([.2 1]); ylim([.2 1]); set(gca,'TickDir','out'); xticks(.2:.2:1)
xlabel("Correlation with freq."); ylabel("Correlation with order.");

%% corrFreq vs. corrOrder - scatter plot

figure('Position', [10 10 250 250]); hold on;
nn=2; scatter(corrFreq(:,nn),corrOrder(:,nn),'filled','Color',[1 0 0],'MarkerFaceAlpha',.7);
nn=3; scatter(corrFreq(:,nn),corrOrder(:,nn),'filled','Color',conventionalColor,'MarkerFaceAlpha',.7);

plot([-.2 1],[-.2 1],'--','Color',[0 0 0]);
xlim([-.2 1]); ylim([-.2 1]); set(gca,'TickDir','out'); xticks(-0.2:.2:1)
xlabel("Correlation with freq."); ylabel("Correlation with order");
%% Supple? Learning sequence vs. Item order
itemOrder = zeros(numTrial,10);

for tt = 1:numTrial
    for cc = 1:numCategories
        idxOfItem=find(trainOrderList(tt,:)==cc);
        itemOrder(tt,cc) = idxOfItem(end);
        clear idxOfItem
    end
end

orderData = zeros(size(trainOrderList));
for tt = 1:numTrial
    [~,I] = sort(itemOrder(tt,:));
    for cc = 1:numel(I)
        class = I(cc);
        idx = find(trainOrderList(tt,:)==class);
        orderData(tt,idx) = cc;
        clear class idx
    end
    clear tt
end

figure;
for tt = 1:numTrial
    subplot(10,10,tt); hold on;
    plot(1:size(trainOrderList,2),orderData(tt,:),'.','MarkerSize',10);
    xlim([1 size(trainOrderList,2)]); xlabel("Learning sequence"); ylabel("Item order")
    title("Trial "+num2str(tt))
end


%% Fig 5b. Sample plot for freq vs performance 

%% Supple. 2, Unified flx. plot

addpath('catast_function')
numTrial = numTrial;
f = figure; hold on

x = 1:numCategories;

initColor = stbColor; endColor = conventionalColor;
colorMat = [linspace(initColor(1),endColor(1),size(meanAcc,3));
    linspace(initColor(2),endColor(2),size(meanAcc,3));
    linspace(initColor(3),endColor(3),size(meanAcc,3))];

% plot(x,ones(1,numCategories)*1/numCategories,'Color',[0 0 0],'LineStyle','--');
% 
% plot(x,chanceMeanAcc(numCategories+1,1:numCategories,2),'--','Color',[1 0 0],LineWidth = 2)

for nn = 1:size(meanAcc,3)
    shaded_errorbar(meanAcc(numCategories+1,1:numCategories,nn),stdAcc(numCategories+1,1:numCategories,nn),colorMat(:,nn)',x,0);
end
%xticks(5:5:30); yticks(0:0.2:1.0);
xlabel('Item order')
ylabel('Performance')
xlim([0 numCategories])
ylim([-0.1 1])
set(gca,'TickDir','out');

% [h,p] = ttest(performanceCell(numCategories+1,5,:,2),1/numCategories,"Tail","right")

%% Supple. 2, first-last item plot

x = [0.3 0.5 0.8 0.9 1.0];
figure; hold on;
shaded_errorbar(squeeze(meanAcc(numCategories+1,1,:)),squeeze(stdAcc(numCategories+1,1,:)),stbColor,x,0);
shaded_errorbar(squeeze(meanAcc(numCategories+1,end,:)),squeeze(stdAcc(numCategories+1,end,:)),conventionalColor,x,0);

%xticks(5:5:30); yticks(0:0.2:1.0);
xlabel('Item order')
ylabel('Performance')
xlim([0.2 1])
ylim([-0.1 1])
set(gca,'TickDir','out');


%% Supple. 3, forgetting curve and the asymptots
numCycle = 10; nn = 2;
highPerf = zeros(numCycle-1,numCategories,numTrial,3);
lowPerf = zeros(numCycle-1,numCategories,numTrial,3);
xx = zeros(2*(numCycle-1),numCategories,numTrial,3);
% asymp = zeros(2,numCategories);

% figure; 
% ft = fittype('a*(x-b)^(-1)+c');
for nn = 1:3
for cc = 1:numCategories
    for cyc = 1:numCycle-1
        xx(2*(cyc-1)+1,cc) = numCategories*(cyc-1)+cc+1;
%         highPerf(cyc,cc,nn) = meanAcc(numCategories*(cyc-1)+cc+1,cc,nn);
        highPerf(cyc,cc,:,nn) = performanceCell(numCategories*(cyc-1)+cc+1,cc,:,nn);

        xx(2*(cyc-1)+2,cc) = numCategories*(cyc-1)+cc+numCategories;
        lowPerf(cyc,cc,nn) = meanAcc(numCategories*(cyc-1)+cc+numCategories,cc,nn);
        lowPerf(cyc,cc,:,nn) = performanceCell(numCategories*(cyc-1)+cc+numCategories,cc,:,nn);
    end
%     subplot(3,4,cc); hold on
%     [curve,~] = fit(xx(1:2:end,cc),highPerf(:,cc),ft); asymp(1,cc) = curve.c;
%     plot(xx(1:2:end,cc),highPerf(:,cc),'o'); plot(curve);
%     [curve,~] = fit(xx(2:2:end,cc),lowPerf(:,cc),ft); asymp(2,cc) = curve.c;
%     plot(xx(2:2:end,cc),lowPerf(:,cc),'o'); plot(curve);
%     xlim([1 101]); ylim([0 1])
end
end


meanHigh = mean(highPerf,[2,3]); stdHigh = std(highPerf,0,[2,3]);
meanLow = mean(lowPerf,[2,3]); stdLow = std(lowPerf,0,[2,3]);
figure('position',[680,597,356,158]); hold on;

nn = 2;
shaded_errorbar(meanHigh(:,nn),stdHigh(:,nn),conventionalColor,1:(numCycle-1),1);
shaded_errorbar(meanLow(:,nn),stdLow(:,nn),conventionalColor,1:(numCycle-1),1);
nn = 3;
shaded_errorbar(meanHigh(:,nn),stdHigh(:,nn),[1 0 0],1:(numCycle-1),1);
shaded_errorbar(meanLow(:,nn),stdLow(:,nn),[1 0 0],1:(numCycle-1),1);
xlabel('Repetition cycle'); ylabel('Performance difference')
xlim([0 numCycle-1]); ylim([0.0 1.0])
set(gca,'TickDir','out');

diffPerf = highPerf-lowPerf; meanDiff = mean(diffPerf,[2,3]); stdDiff = std(diffPerf,0,[2,3]);
figure('position',[680,597,356,158]); hold on;

shaded_errorbar(meanDiff(:,2),stdDiff(:,2),conventionalColor,1:(numCycle-1),1);
shaded_errorbar(meanDiff(:,3),stdDiff(:,3),[1 0 0],1:(numCycle-1),1);
xlabel('Repetition cycle'); ylabel('Performance difference')
xlim([0 numCycle-1]); ylim([0.1 1.0]); xticks(0:1:9); set(gca, 'YScale', 'log')
set(gca,'TickDir','out');


%resizeDiff = reshape(diffPerf(:,:,:,2),9,1000);
% mm = squeeze(mean(diffPerf,3));
% cumKendallTest(mm(:,:,2),(1:9)','lt')
% 
% anova(reshape(diffPerf(:,:,:,3),9,1000)')
% cumKendallTest(reshape(diffPerf(:,:,:,3),9,1000),(1:9)','lt')

reshapedDiff = reshape(diffPerf(:,:,:,3),9,1000);
[h,p,~,stat] = ttest(reshapedDiff(1,:),reshapedDiff(2,:))


reshapedDiff = reshape(diffPerf(:,:,:,2),9,1000);
[h,p,~,stat] = ttest(reshapedDiff(1,:),reshapedDiff(2,:))
