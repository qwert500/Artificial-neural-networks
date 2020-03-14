clc, clear all
%======Parameters========%
numberOfDimensions=2;
numberOfInputs=1000;
numberOfOutputs=100;
timeOrderingPhase=1000;
timeConvergencePhase=5*10^4;
sigma0=5;
eta0=0.1;
tau=200;
etaConv=0.01;
sigmaConv=0.9;

neighbourhood=zeros(numberOfOutputs,1);
xi=zeros(numberOfInputs,numberOfDimensions);
b=zeros(numberOfOutputs,1);
deltaWeight=zeros(numberOfOutputs,numberOfDimensions);

for i=1:numberOfInputs
  xi(i,:)=GetInput;
end

w=-0.2*ones(numberOfOutputs,numberOfDimensions)+0.4*rand(numberOfOutputs,numberOfDimensions);
%========OrderingPhase===========%
for t=1:timeOrderingPhase
  sigma=sigma0*exp(-t/tau);
  eta=eta0*exp(-t/tau);
  k=randi(numberOfInputs);
  for i=1:numberOfOutputs
    b(i)=norm(w(i,:)-xi(k,:));
  end
  [bi0,i0]=min(b);
  
  for i=1:numberOfOutputs
    neighbourhood(i)=NeighbourhoodFunction(i,i0,sigma);
  end
  
  for i=1:numberOfOutputs
    for j=1:numberOfDimensions
      deltaWeight(i,j)=eta*neighbourhood(i)*(xi(k,j)-w(i,j));
    end
  end
  w=w+deltaWeight;
end
wOrderLast=w;

%======Plot Ordering Phase=======%
figure(1)
subplot(1,2,1)
GenerateEquilateralTriangle
hold on
plot(w(:,1),w(:,2),'-o')
xlabel('\xi1')
ylabel('\xi2')
title('Ordering phase')
hold off
%============ConvergancePhase=============%
sigma=sigmaConv;
eta=etaConv;

for t=1:timeConvergencePhase
  
  k=randi(numberOfInputs);
  for i=1:numberOfOutputs
    b(i)=norm(w(i,:)-xi(k,:));
  end
  [bi0,i0]=min(b);
  
  for i=1:numberOfOutputs
    neighbourhood(i)=NeighbourhoodFunction(i,i0,sigma);
  end
  
  for i=1:numberOfOutputs
    for j=1:numberOfDimensions
      deltaWeight(i,j)=eta*neighbourhood(i)*(xi(k,j)-w(i,j));
    end
  end
  w=w+deltaWeight;
end
%=========Plot for after convergance phase==========%
subplot(1,2,2)
GenerateEquilateralTriangle
hold on
plot(w(:,1),w(:,2),'-o')
xlabel('\xi1')
ylabel('\xi2')
title('Convergence phase')
hold off
wConvergenceLast=w;
