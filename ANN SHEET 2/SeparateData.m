function [trainingData2,validationData]=SeparateData(trainingData)

k=length(trainingData);
index = randperm(k);
trainingIndex = index(1:0.7*k);
validationIndex =index(0.7*k+1:end);
trainingData2=trainingData(trainingIndex,:);
validationData=trainingData(validationIndex,:);



end