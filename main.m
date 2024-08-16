clc; clear; close all;
loadPath = "D:\SC\test\";

%% Create two-digit MNIST dataset

if ~exist(loadPath+"twoDigitMNIST", 'dir')
   create_twoDigitMNIST;
end

%% Generate all neccessary data, from beginning to the end

clear; clc; close all;
epochwiseTest = 1;
numCategories = 10;
calculateShuffleAccuracy = 1;
serial_position_effect % Fig 2
disp("code for the serial position effect - done"); disp(" ")

clear; clc; close all;
epochwiseTest = 0;
numCategories = 30;
calculateShuffleAccuracy = 1;
serial_position_effect % Fig 3
disp("code for the serial position effect - done"); disp(" ")

clear; clc; close all;
epochwiseTest = 0;
numCategories = 10;
nwNum = 2;
calculateShuffleAccuracy = 0;
calculateAttackShuffleAccuracy = 1;
Hebb_repetition_effect % Fig 4
disp("code for the Hebb repetition effect - done"); disp(" ")

clear; clc; close all;
epochwiseTest = 0;
numCategories = 10;
calculateShuffleAccuracy = 0;
nwNum = 2;
learning_frequency_varying % Fig 5
disp("code for the learning frequency varying - done"); disp(" ")

