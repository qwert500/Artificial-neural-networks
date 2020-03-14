clc, clear all

learningRate=0.01;
beta=1/2;
numberOfTrainingExperiments=100;
numberOfIterations=2*10^5;
numberOfInputVariables=2;


trainingSet=trainingSet2016; %calls upon a function with the training data given
validationSet=validationSet2016;%calls upon a function with the validation data given
normalizedTrainingSet=zeros(length(trainingSet),2);
normalizedValidationSet=zeros(length(validationSet),2);
w=zeros(2,1);
bias=zeros(1,1);
classificationErrorTS=zeros(numberOfIterations, numberOfTrainingExperiments);
classificationErrorVS=zeros(numberOfIterations, numberOfTrainingExperiments);
minimumErrorTS=zeros(numberOfTrainingExperiments,1);
minimumErrorVS=zeros(numberOfTrainingExperiments,1);
avgMinimumErrorTS=0;
avgMinimumErrorVS=0;
b=zeros(numberOfInputVariables,1);

for i=1:size(trainingSet-1,2) %normalizing indata patterns
  normalizedTrainingSet(:,i)=(trainingSet(:,i)-mean(trainingSet(:,i)))/(std(trainingSet(:,i)));
  normalizedValidationSet(:,i)=(validationSet(:,i)-mean(validationSet(:,i)))/(std(validationSet(:,i)));
end

for i=1:numberOfInputVariables %Generates initial values of the weights
  w(i)=-0.2+0.4*rand(1);
end

for i=1:1 %Generates initial values of the biases
  bias(i)=-1+2*rand(1);
end

%===================Initializing Training=========================

for trainingExperiment=1:numberOfTrainingExperiments %Training experiments begins
      O=zeros(length(trainingSet),1);
      Ov=zeros(length(trainingSet),1);
  for iteration=1:numberOfIterations

    mu=randi([1 length(trainingSet)]); %picks random mu for output calc. and vise versa
        O(mu)=0;
        if mu<length(validationSet)+1
        Ov(mu)=0;
        end
    for i=1:numberOfInputVariables %calculating output 
      O(mu)=w(i)*normalizedTrainingSet(mu,i)-bias+O(mu);
      if mu<length(validationSet)+1
      Ov(mu)=w(i)*normalizedValidationSet(mu,i)-bias+O(mu);
      end
    end

    for i=1:numberOfInputVariables %calculating b
      b(i)=w(i)*normalizedTrainingSet(mu,i)-bias;
    end
    
    for i=1:numberOfInputVariables %updating bias
      bias=learningRate*(trainingSet(mu,size(trainingSet,2))-O(mu))*(-1)*beta*...
        (1-(tanh(beta*b(i)))^2)+bias;
    end
    
    for i=1:numberOfInputVariables %weight update
      w(i)=w(i)+learningRate*(trainingSet(mu,size(trainingSet,2))-O(mu))...
        *normalizedTrainingSet(mu,i);
    end
    
    
    %evaluating classification error for training set
    for mu=1:length(trainingSet)
      classificationErrorTS(iteration, trainingExperiment)=1/(2*length(trainingSet))...
        *abs(trainingSet(mu,size(trainingSet,2))-sign(O(mu)))...
        +classificationErrorTS(iteration, trainingExperiment);
    end
    %evaluating classification error for validation set

    for mu=1:length(validationSet)
      classificationErrorVS(iteration, trainingExperiment)=1/(2*length(validationSet))...
        *abs(validationSet(mu,size(validationSet,2))-sign(Ov(mu)))...
        +classificationErrorVS(iteration, trainingExperiment);
    end
    
  end
end



for trainingExperiment=1:numberOfTrainingExperiments
  
  minimumErrorTS(trainingExperiment)=min(classificationErrorTS(:,trainingExperiment));
  minimumErrorVS(trainingExperiment)=min(classificationErrorVS(:,trainingExperiment));
  
  
  avgMinimumErrorTS=sum(minimumErrorTS(trainingExperiment))/numberOfTrainingExperiments+avgMinimumErrorTS;
  avgMinimumErrorVS=sum(minimumErrorVS(trainingExperiment))/numberOfTrainingExperiments+avgMinimumErrorVS;
end

