%%Sheet1Task3
clc, clear all
tic
beta=2;
N=250;
totalNumberOfTimeSteps=1000;
numberOfIterations=1;
S=zeros(N,N,totalNumberOfTimeSteps); %store S for every p (there are N p:s) and t!!
m1=zeros(totalNumberOfTimeSteps,N);%there is p=1,...N patterns, order parameter
g=zeros(totalNumberOfTimeSteps,N); 
b=zeros(N,totalNumberOfTimeSteps);
m1Mean=zeros(N,1);
transientLimit=100; %Starts avrage of m1 at t=transientLimit. 
meanMean=0;

for iteration=1:numberOfIterations
  
  for p=1:N %For p=1,...,N patterns
    w=zeros(N,N);
    ny=randi(p); % the ny:th pattern <=> feed pattern
    zetaStored=zeros(N,p);
    
    for j=1:p % Generates random stored patterns
      for i=1:N
        zetaStored(i,j)=randi(2)*2-3;
      end
    end
    
    for i=1:N %calculating weights according to Hebb's Rule
      for j=1:N
        if i==j
          w(i,j)=0;
        else
          for k=1:p
            w(i,j)=1/N*zetaStored(i,k)*zetaStored(j,k)+w(i,j);
          end
        end
      end
    end
    
    for t=1:totalNumberOfTimeSteps %Time steps begins
      if t==1 %generates S(t=0) (S(t=1) beacause of matrix storing)
        S(:,p,t)=zetaStored(:,ny);
      end
      b=zeros(N,totalNumberOfTimeSteps);
      for i=1:N %calculationg elements of b
        for j=1:N
          b(i,t)=w(i,j)*S(j,p,t)+b(i,t);
        end
      end
      
      for i=1:N
        m1(t,p)=1/N*S(i,p,t)*S(i,p,1)+m1(t,p);% S(t=1) for every p is zeta_i^1
      end
      for i=1:N % calculating probabilty for 1 and -1 for every bit 'i'.
        g(t,i)=1/(1+exp(-2*beta*b(i,t)));
      end
      
      for i=1:N %stochastic updating step in time
        r=rand;
        if r<g(t,i)
          S(i,p,t+1)=1;
        else
          S(i,p,t+1)=-1;
        end
      end
    end
  end
end

alpha=(1:N)/N;

for p=1:N % calculating the mean of m1(t)
  for t=transientLimit:totalNumberOfTimeSteps; %starts at 100 to cut of the intitaial transient
    m1Mean(p)=m1(t,p)/(totalNumberOfTimeSteps-transientLimit)+m1Mean(p);
  end
end

for p=transientLimit:N
meanMean=m1Mean(p)/(N-transientLimit)+meanMean;
end
figure(1) %plots mean of m1 over alpha
hold on
xlabel('alpha')
ylabel('<m1>')
title('<m1> over alpha')
plot(alpha,m1Mean(1:N))
plot(alpha,meanMean)
hold off
toc


