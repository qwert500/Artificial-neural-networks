clc, clear all

learningRate=0.01;
beta=1/2;
numberOfTrainingExperiments=100;
numberOfIterations=2*10^5;
numberOfInputVariables=2;
numberOfNeuronsHiddenLayerVector=[2 4 8 16 32];

finalValuesVS=zeros(length(numberOfNeuronsHiddenLayerVector),1);
finalValuesTS=zeros(length(numberOfNeuronsHiddenLayerVector),1);
trainingSet=trainingSet2016; %calls upon a function with the training data given
validationSet=validationSet2016;%calls upon a function with the validation data given
normalizedTrainingSet=zeros(length(trainingSet),2);
normalizedValidationSet=zeros(length(validationSet),2);


for neuronDrive=1:length(numberOfNeuronsHiddenLayerVector)
numberOfNeuronsHiddenLayer=numberOfNeuronsHiddenLayerVector(neuronDrive);

w=zeros(numberOfInputVariables,numberOfNeuronsHiddenLayer);
bias1=zeros(1,1);
classificationErrorTS=zeros(numberOfIterations, numberOfTrainingExperiments);
classificationErrorVS=zeros(numberOfIterations, numberOfTrainingExperiments);
minimumErrorTS=zeros(numberOfTrainingExperiments,1);
minimumErrorVS=zeros(numberOfTrainingExperiments,1);
avgMinimumErrorTS=0;
avgMinimumErrorVS=0;
b=zeros(numberOfInputVariables,numberOfNeuronsHiddenLayer);
bv=zeros(numberOfInputVariables,numberOfNeuronsHiddenLayer);
B=zeros(numberOfNeuronsHiddenLayer,1);
Bv=zeros(numberOfNeuronsHiddenLayer,1);
W=zeros(numberOfNeuronsHiddenLayer,1);

for i=1:size(trainingSet-1,2) %normalizing indata patterns
  normalizedTrainingSet(:,i)=(trainingSet(:,i)-mean(trainingSet(:,i)))/(std(trainingSet(:,i)));
  normalizedValidationSet(:,i)=(validationSet(:,i)-mean(validationSet(:,i)))/(std(validationSet(:,i)));
end

%===================Initializing Training=========================


for trainingExperiment=1:numberOfTrainingExperiments %Training experiments begins
  V=zeros(length(trainingSet),numberOfNeuronsHiddenLayer);
  Vv=zeros(length(validationSet),numberOfNeuronsHiddenLayer);
  O=zeros(length(trainingSet),1);
  Ov=zeros(length(validationSet),1);
  
  for j=1:numberOfNeuronsHiddenLayer
    for i=1:numberOfInputVariables %Generates initial values of the weights for first layer
      w(i,j)=-0.2+0.4*rand(1);
    end
  end
  
  for j=1:numberOfNeuronsHiddenLayer %Generates initial values of weights in secound layer
    W(j)=-0.2+0.4*rand(1);
  end
  
  for j=1:numberOfNeuronsHiddenLayer %Generates initial values of the biases for first layer
    bias1(j)=-1+2*rand(1);
  end
  
  bias2=-1+2*rand(1); %bias for the output neuron
  
  for iteration=1:numberOfIterations
    
    mu=randi([1 length(trainingSet)]); %picks random mu for output calc. and vise versa
    
    for j=1:numberOfNeuronsHiddenLayer %calculating b for first layer
      for i=1:numberOfInputVariables
        b(i,j)=w(i,j)*normalizedTrainingSet(mu,i)-bias1(j);
        if mu<=length(validationSet)
          bv(i,j)=w(i,j)*normalizedValidationSet(mu,i)-bias1(j);
        end
      end
    end
    
    for j=1:numberOfNeuronsHiddenLayer
      for i=1:numberOfInputVariables %calculating state of neurons in hidden layer
        V(mu,j)=0;
        V(mu,j)=b(i,j)+V(mu,j);
        if mu<=length(validationSet)
          Vv(mu)=0;
          Vv(mu,j)=bv(i,j)+Vv(mu,j);
        end
      end
    end
    
    for j=1:numberOfNeuronsHiddenLayer %tanh(sum(b))
      V(mu,j)=tanh(V(mu,j));
      if mu<=length(validationSet)
        Vv(mu,j)=tanh(Vv(mu,j));
      end
    end
    
    for j=1:numberOfNeuronsHiddenLayer
      B(j)=W(j)*V(mu,j)-bias2;
      if mu<=length(validationSet)
        Bv(j)=W(j)*Vv(mu,j)-bias2;
      end
    end
    
    O(mu)=0;
    if mu <=length(validationSet)
      Ov(mu)=0;
    end
    
    for j=1:numberOfNeuronsHiddenLayer
      O(mu)=B(j)+O(mu);
      if mu <=length(validationSet)
        Ov(mu)=Bv(j)+Ov(mu);
      end
    end
    
    for j=1:numberOfNeuronsHiddenLayer %updating biases for first layer
      for i=1:numberOfInputVariables
        bias1(j)=learningRate*(trainingSet(mu,size(trainingSet,2))-O(mu))*beta*...
          (1-(tanh(beta*B(j)))^2)*W(j)*(-1)*beta*...
          (1-(tanh(beta*b(i,j)))^2)+bias1(j);
      end
    end
    
    bias2=learningRate*(trainingSet(mu,size(trainingSet,2))-O(mu))*(-1)*beta*...
      (1-(tanh(beta*B(j)))^2); %updating bias for secound layer
    
    for j=1:numberOfNeuronsHiddenLayer %weight update for secound layer
      W(j)=W(j)+learningRate*(trainingSet(mu,size(trainingSet,2))-O(mu))...
        *beta*(1-(tanh(beta*B(j)))^2)*V(mu,j);
    end
    
    for i=1:numberOfInputVariables
      for j=1:numberOfNeuronsHiddenLayer
        w(i,j)=w(i,j)+learningRate*(trainingSet(mu,size(trainingSet,2))-O(mu))*W(j)...
          *beta*(1-(tanh(beta*B(j)))^2)*normalizedTrainingSet(mu,i);
      end
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

finalValuesTS(neuronDrive)=avgMinimumErrorTS;
finalValuesVS(neuronDrive)=avgMinimumErrorVS;
end
xlabel('Number Of Neurons')
ylabel('Error')
plot(numberOfNeuronsHiddenLayerVector,finalValuesTS,'r',numberOfNeuronsHiddenLayerVector,finalValuesVS,'b')
legend('Training Set','Validation Set')