clc;
clear;

% Load the image dataset
imds = imageDatastore('/Users/aaryanraj/Documents/Final_year_project/ecgdataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Step 1: Split 80% for training (will further split this) and 20% for testing
[imgsTrain, imgsTest] = splitEachLabel(imds, 0.8, 'randomize');

% Step 2: Split 80% of imgsTrain for actual training, 20% for validation
[trainSet, valSet] = splitEachLabel(imgsTrain, 0.8, 'randomize');
% Final distribution: 1152 train (384/class), 288 val (96/class), 360 test (120/class)

% Image data augmentation for training set
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

inputSize = [227 227 3];

% Prepare augmented image datastores
augimdsTrain = augmentedImageDatastore(inputSize, trainSet, 'DataAugmentation', augmenter);
augimdsValidation = augmentedImageDatastore(inputSize, valSet);
augimdsTest = augmentedImageDatastore(inputSize, imgsTest);

% Load pretrained SqueezeNet
net = squeezenet;
lgraph = layerGraph(net);

% Remove last layers (to add our custom classifier)
layersToRemove = {'drop9', 'conv10', 'relu_conv10', 'pool10', 'prob', 'ClassificationLayer_predictions'};
lgraph = removeLayers(lgraph, layersToRemove);

% Add new classification layers
numClasses = 3;
newLayers = [
    dropoutLayer(0.5, 'Name', 'new_dropout')
    convolution2dLayer(1, numClasses, 'Name', 'new_conv10', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    globalAveragePooling2dLayer('Name', 'new_gapool')
    softmaxLayer('Name', 'new_prob')
    classificationLayer('Name', 'new_classoutput')
];

% Connect new layers to the graph
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'fire9-concat', 'new_dropout');

% Define training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 5e-5, ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 20, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

% Evaluate on test set
YPred = classify(trainedNet, augimdsTest);
YTest = imgsTest.Labels;
accuracy = mean(YPred == YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Plot confusion matrix
figure;
plotconfusion(YTest, YPred);
title('Confusion Matrix for SqueezeNet ECG Classification');
