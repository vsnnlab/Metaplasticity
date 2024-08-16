function [netUpdate, lim, Weights, Biases] = initializeWeight(net, ver, stdfac, rand_layers_ind)
netUpdate = net.saveobj;
netTemp = netUpdate.LayerGraph.saveobj;

Weights = cell(1,length(rand_layers_ind));
Biases = cell(1,length(rand_layers_ind));
for ind_tl = 1:length(rand_layers_ind)
    targetlayer_ind = rand_layers_ind(ind_tl);
    weight_conv = net.Layers(targetlayer_ind ,1).Weights;
    bias_conv = net.Layers(targetlayer_ind ,1).Bias;
    
    fan_in = size(weight_conv,2);
    fan_out = size(weight_conv,1);
    
    if ver == 1
        lim(ind_tl) = sqrt(2/(fan_in+fan_out));
        Wtmp = stdfac*randn(size(weight_conv))*sqrt(2/(fan_in+fan_out)); % Glorot initializaation
        Btmp = randn(size(bias_conv));
    elseif ver == 2
        lim(ind_tl) = sqrt(6/(fan_in+fan_out));
        Wtmp = stdfac*(rand(size(weight_conv))-0.5)*2*sqrt(6/(fan_in+fan_out)); % Glorot uniform initializaation
        Btmp = randn(size(bias_conv));
    end

    %% change network parameters
    
    weight_conv_randomize = single(1*Wtmp);
    bias_conv_randomize = single(0*Btmp);
    
    Weights{ind_tl} = weight_conv_randomize;
    Biases{ind_tl} = bias_conv_randomize;
    
    netTemp.Layers(targetlayer_ind).Weights = weight_conv_randomize;
    netTemp.Layers(targetlayer_ind).Bias = bias_conv_randomize;
end
netUpdate.LayerGraph = netUpdate.LayerGraph.loadobj(netTemp);
netUpdate = dlnetwork(netUpdate.LayerGraph);
end