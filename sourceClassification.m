
inputFolder = 'data';

resultFolder = 'results\';

    if ~exist(resultFolder, 'dir')
       mkdir(resultFolder)
    end

net = resnet50;

prefiks={'vz'; 'tzz'; 'txz'; 'vx';  'txx'}
image_layer = [1,1,1,1,1];

for ind = 1:size(prefiks,1)

    tableFileName = [resultFolder 'pred_' inputFolder  '_' char(prefiks{ind}) '.xls'];
    matFileName =   [resultFolder inputFolder  '_'  char(prefiks{ind}) '.mat'];
    dataStoreFolder = ['..\' inputFolder '\images_' char(prefiks{ind})];
        
    image_layers = image_layer(ind);

    imds = imageDatastore(dataStoreFolder,...
    'IncludeSubfolders',true,'FileExtensions','.jpeg','LabelSource','foldernames')

    n=size(imds.Files,1);

    labelCount = countEachLabel(imds);
    img = readimage(imds,1);

    numTrainFiles = ceil(0.8*min(labelCount.Count));

    [imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
    
    labelCountTrain = countEachLabel(imdsTrain);
    labelCountValidation = countEachLabel(imdsValidation);
    disp('Train count: ')
    labelCountTrain.Count
    disp('Validation count: ')
    labelCountValidation.Count


inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net);
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

inputSize(3)=image_layer(ind);
firstLayer = imageInputLayer(inputSize, 'name', 'data');
lgraph = replaceLayer(lgraph,net.Layers(1).Name,firstLayer);

Weights = sum(net.Layers(2).Weights,3);
b = net.Layers(2).Bias;
Stride = net.Layers(2).Stride;
name = 'conv1A';

filterSize = 7;
numFilters = 64;
secondLayer = convolution2dLayer(filterSize,numFilters, ...
    'Weights',Weights, ...
    'Bias',b, ...
    'name', name, ...
    'Stride', 2, ...
    'Padding', [3 3]);

lgraph = replaceLayer(lgraph,net.Layers(2).Name,secondLayer);

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(augimdsTrain,lgraph,options);

h= findall(groot,'Type','Figure');
    h.MenuBar = 'figure';
    imageResultFolder = [resultFolder 'tp_' char(prefiks{ind}) '.jpeg'];
    saveas(h, imageResultFolder);
close(h)

[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

save(matFileName, 'net', 'options');

h = figure('units','normalized','outerposition',[0 0 1 1]);
confusionchart(imdsValidation.Labels,YPred, 'Title',"Confusion Matrix for Validation Data", 'FontSize',30);
imageResultFolder = [resultFolder 'cm_' char(prefiks{ind}) '.jpeg'];
saveas(h, imageResultFolder);
close(h)

cm = confusionmat(imdsValidation.Labels,YPred);


extendedConfusionMatrix= myConf2D(cm(1,1), cm(1,2), cm(2,1), cm(2,2));
imageResultFolder = [resultFolder 'cm_' char(prefiks{ind}) '.txt'];
dlmwrite(imageResultFolder, extendedConfusionMatrix);

[res indi] = find(imdsValidation.Labels~=YPred);

output_file = [];
    for i2 = 1:length(imdsValidation.Labels)
        temp1 = split(imdsValidation.Files{i2},'\');
        temp2 = temp1{end};
        temp1 = split(temp2, '.');

        output_file = [output_file;str2double(temp1{1})];
    end


    T = table(output_file,imdsValidation.Labels); 
   
    writetable(T,tableFileName,'Sheet','sourceData','WriteVariableNames',true);



end








