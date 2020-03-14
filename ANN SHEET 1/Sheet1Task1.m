%% Task 1
clc, clear all
tic
P=[10 20 30 40 50 75 100 150 200]; %number of patterns stored
numberOfBits=[100 200];% number of bits
pdivN=zeros(length(P),length(numberOfBits));
PerrorAnalytic=zeros(length(P),length(numberOfBits));
Perror=zeros(length(P),length(numberOfBits));
numberOfIterations=500;

for iteration=1:numberOfIterations
  for n=1:length(numberOfBits)
    N=numberOfBits(n);
    for z=1:length(P)
      w=zeros(N,N);
      p=P(z);
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
      
      S=sign(w*zetaStored(:,ny)); %one time step
      
      diffS=S-zetaStored(:,ny); % if same elemets => -2,2 ,wants -1,1
      numberOfErrors=sum(abs(diffS)./2);
      Perror(z,n)=numberOfErrors/(N*numberOfIterations)+Perror(z,n);
      PerrorAnalytic(z,n)=(1/2)*(1-erf((1/sqrt(2))*sqrt(N/p)));
      pdivN(z,n)=p/N;
      
    end
  end
end

for i=1:length(numberOfBits)
  hold on
  title('Error estimation for one time step')
  xlabel('p/N')
  ylabel('Perror')
  plot(pdivN(:,i),Perror(:,i),'gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
  plot(pdivN(:,i),PerrorAnalytic(:,i),'r')
  legend('PerrorCalculated','PerrorAnalytical')
end
toc