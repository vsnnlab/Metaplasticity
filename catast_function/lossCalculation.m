function [loss,gradients,state] = lossCalculation(net,X,T)

% Forward data through network.
[Y,state] = forward(net,X);

% Calculate cross-entropy loss.
loss = crossentropy(Y,T);

% Calculate gradients of loss with respect to learnable parameters.
learnableIdx = find(contains(net.Learnables.Layer, "fc")); % & net.Learnables.Parameter=="Weights");
gradients = dlgradient(loss,net.Learnables(learnableIdx,:));

end