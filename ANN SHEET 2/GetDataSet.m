function [classificationData, trainingData] = GetDataSet(wineDataFile,numberOfDimensions,numberOfInputs)

rawData = importdata(wineDataFile);
classificationData = rawData(:,1);
trainingData = zeros(numberOfInputs,numberOfDimensions);

for i = 1:numberOfDimensions
    trainingData(:,i) = zscore(rawData(:,i+1));
end    
end