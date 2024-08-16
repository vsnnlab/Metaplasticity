function shaded_errorbar(meanVal,errorVal,colorLine,x_axis,marker,emptyMarkerIdx,lineStyle)
    if (nargin<6); emptyMarkerIdx = [];
    else if (nargin<7); lineStyle = '-';
    end
     
    curve1 = meanVal + errorVal;
    curve2 = meanVal - errorVal;
    x2 = [x_axis, fliplr(x_axis)];
    inBetween = [curve1(:)', fliplr(curve2(:)')];
    fill(x2, inBetween, colorLine,'FaceAlpha',0.2,'LineStyle','none');
    hold on
    if marker
        plot(x_axis, meanVal, '-o','MarkerFaceColor',colorLine,'color', colorLine,'LineWidth',2);
        plot(emptyMarkerIdx, meanVal(emptyMarkerIdx), 'o','MarkerFaceColor',[1 1 1],'MarkerEdgeColor',colorLine,'LineWidth',2);
    else
        plot(x_axis, meanVal, 'color', colorLine, 'LineWidth', 2, 'LineStyle', lineStyle);
    end    

end

