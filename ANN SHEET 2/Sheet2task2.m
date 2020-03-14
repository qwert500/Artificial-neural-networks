clc, clear all
%======Parameters========%
numberOfDimensions=13;
numberOfInputs=178;
numberOfOutputs=20;
numberOfWeights=numberOfOutputs*numberOfOutputs;
timeOrderingPhase=1000;
timeConvergencePhase=2*10^4;
sigma0=30;
eta0=0.1;
tau=300;
etaConv=0.01;
sigmaConv=0.9;

[classificationData, trainingData] = GetDataSet('wine.data.txt',numberOfDimensions,numberOfInputs);
neighbourhood=zeros(numberOfOutputs,numberOfOutputs);
b=zeros(numberOfOutputs,numberOfOutputs);
deltaWeight=zeros(numberOfOutputs,numberOfOutputs,numberOfDimensions);
nOnes = [];
nTwos = [];
nThrees = [];

w=-0.2*ones(numberOfOutputs,numberOfOutputs,numberOfDimensions)+0.4*rand(numberOfOutputs,numberOfOutputs,numberOfDimensions);
%======Ordering Phase==========
for t=1:timeOrderingPhase
  sigma=sigma0*exp(-t/tau);
  eta=eta0*exp(-t/tau);
  
  k=randi(numberOfInputs);
  for i=1:numberOfOutputs
    for j=1:numberOfOutputs
      b(i,j)=norm(squeeze(w(i,j,:))-trainingData(k,:)');
    end
  end
  [bi0,iTmp]=min(b(:));
  [i0,j0]=ind2sub(size(b),iTmp);
  
  
  for i=1:numberOfOutputs
    for j=1:numberOfOutputs
      deltaWeight(i,j,:)=eta*NeighbourhoodFunction2([i,j],[i0,j0],sigma)*(trainingData(k,:)'-squeeze(w(i,j,:)));
    end
  end
  
  w=w+deltaWeight;
end

sigma=sigmaConv;
eta=etaConv;

%========Convergence Pase=======
for t=1:timeConvergencePhase
  
  k=randi(numberOfInputs);
  for i=1:numberOfOutputs
    for j=1:numberOfOutputs
      b(i,j)=norm(squeeze(w(i,j,:))-trainingData(k,:)');
    end
  end
  [bi0,iTmp]=min(b(:));
  [i0,j0]=ind2sub(size(b),iTmp);
  
  for i=1:numberOfOutputs
    for j=1:numberOfOutputs
      deltaWeight(i,j,:)=eta*NeighbourhoodFunction2([i,j],[i0,j0],sigma)*(trainingData(k,:)'-squeeze(w(i,j,:)));
    end
  end
  w=w+deltaWeight;
end

%======Grouping=========
for k = 1:numberOfInputs
  
  for i=1:numberOfOutputs
    for j=1:numberOfOutputs
      b(i,j)=norm(squeeze(w(i,j,:))-trainingData(k,:)');
    end
  end
  [bi0,iTmp]=min(b(:));
  [i0,j0]=ind2sub(size(b),iTmp);
  if (classificationData(k)==1)
    nOnes=[nOnes;[i0,j0]];
  elseif (classificationData(k)==2)
    nTwos=[nTwos;[i0,j0]];
  elseif (classificationData(k)==3)
    nThrees=[nThrees;[i0,j0]];
  end
  
end
%============plot=============
sz=200;
scatter(nOnes(:,1),nOnes(:,2),sz,'filled','r');
hold on
scatter(nTwos(:,1),nTwos(:,2),sz,'filled','g');
hold on
scatter(nThrees(:,1),nThrees(:,2),sz,'filled','b');

