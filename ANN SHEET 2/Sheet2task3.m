clear all
%========Parameters=========
etaUnsupervised=0.01;
inputDimensions=2;
nDataPoints=2000;
radialBases=5;
unsupervisedIterations=10^5;
independentRuns=20;
beta=1/2;
dataSet = importdata('task3.txt');
supervisedIterations=3000;
etaSupervised=0.1;

unsupervisedWeights = -1*ones(radialBases,inputDimensions,independentRuns)+2*rand(radialBases,inputDimensions,independentRuns);
supervisedWeights = -1*ones(independentRuns,radialBases)+2*rand(independentRuns,radialBases);
Threshold = -1+2*rand(independentRuns,1);
classificationErrors = zeros(independentRuns,1);
radialBasisOutputs = zeros(radialBases,1);

for i = 1:independentRuns
  
  for t = 1:unsupervisedIterations
    v = randi(nDataPoints);
    randomPattern = dataSet(v,2:3);
    for j = 1:radialBases
      radialBasisOutputs(j) = radialBasisFunction(unsupervisedWeights(:,:,i),randomPattern,j);
    end
    [~,winIndex]=max(radialBasisOutputs);
    unsupervisedWeights(winIndex,:,i)=...
      unsupervisedWeights(winIndex,:,i)...
      +etaUnsupervised*...
      (randomPattern-unsupervisedWeights(winIndex,:,i));
  end
  [trainingData,validationData]=SeparateData(dataSet);
  
  nDataPoints2=size(trainingData,1);
  supervisedOutputs=zeros(nDataPoints2,1);
  
  for t = 1:supervisedIterations
    
    v = randi(nDataPoints2);
    randomPattern = trainingData(v,2:3);
    
    for j = 1:radialBases
      radialBasisOutputs(j) = ...
        radialBasisFunction(unsupervisedWeights(:,:,i),randomPattern,j);
    end
    b=(supervisedWeights(i,:)*radialBasisOutputs-Threshold(i));
    
    supervisedOutputs(v)=tanh(beta*b);
    
    delta=(trainingData(v,1)-supervisedOutputs(v))...
      *(sech(beta*b)^2)*beta;
    supervisedWeights(i,:)=supervisedWeights(i,:)...
      +etaSupervised*delta*radialBasisOutputs';
    
    Threshold(i)=Threshold(i)-etaSupervised*delta;
    
  end
  
  classificationErrors(i)=classificationError(validationData,...
    supervisedWeights(i,:),Threshold(i),unsupervisedWeights(:,:,i)...
    ,radialBases,beta);
end

[~,minIndex]=min(classificationErrors);
winningUnsupervisedWeights=unsupervisedWeights(:,:,minIndex);
winningSupervisedWeights=supervisedWeights(minIndex,:);
winningThresholds=Threshold(minIndex);

outputs=zeros(1000,1000);
%=======Generating Outputs=========
for i = 1:1000
  for j = 1:1000
    pattern=[-15+40*i/1000,-15+40*j/1000];
    for k = 1:radialBases
      radialBasisOutputs(k) = ...
        radialBasisFunction(winningUnsupervisedWeights,pattern,k);
    end
    b=(winningSupervisedWeights*...
      radialBasisOutputs-winningThresholds);
    outputs(i,j)=tanh(beta*b);
    if outputs(i,j)<0
      outputs(i,j)=-1;
    else
      outputs(i,j)=1;
    end
  end
end

minusData=[];
plusData=[];
%=========Sorting Data=======
for i = 1:size(dataSet,1)  
  if dataSet(i,1)==1
    plusData=[plusData;dataSet(i,2:3)];
  else
    minusData=[minusData;dataSet(i,2:3)];
  end
end
%==============Plot========
[X,Y]=meshgrid(-15+40*(1:1000)/1000,-15+40*(1:1000)/1000);
contour(X,Y,outputs','k');
hold on

scatter(plusData(:,1),plusData(:,2),'g');
hold on
scatter(minusData(:,1),minusData(:,2),'c');
hold on
scatter(winningUnsupervisedWeights(:,1),...
  winningUnsupervisedWeights(:,2),'b');





