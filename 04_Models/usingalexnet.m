DatasetPath='/Users/aaryanraj/Documents/Final_year_project/ecgdataset';

%Reading Digit Images From Images Database Folder
images=imageDatastore(DatasetPath,'IncludeSubFolders',true,'LabelSource','foldernames');

%Distributing Images in the set of Training and Testing
numTrainFiles=480;
[TrainImages,TestImages]=splitEachLabel(images,numTrainFiles,'randomize');

net=alexnet;
layersTransfer=net.Layers(1:end-3);
numClasses=3;
layers=[layersTransfer
       fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
       softmaxLayer 
       classificationLayer];

options = trainingOptions('sgdm', ...
     'MiniBatchSize',20, ...
     'MaxEpochs',25,...
     'InitialLearnRate',1e-4, ...
     'Shuffle','every-epoch', ...
     'ValidationData',TestImages, ...
     'ValidationFrequency', 10, ...
     'Verbose', false, ...
     'Plots', 'training-progress');

netTransfer=trainNetwork(TrainImages,layers,options);

YPred=classify(netTransfer,TestImages);
YValidation=TestImages.Labels;

accuracy=sum(YPred == YValidation)/nume1(YValidation);
plotconfusion (YValidation,YPred)
