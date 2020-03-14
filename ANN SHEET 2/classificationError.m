function [error] = classificationError(dataSet,supervisedWeights,Threshold,unsupervisedWeights,radialBases,beta)

nPoints=length(dataSet);
b = zeros(nPoints,1);
radialBasisOutputs=zeros(radialBases,1);

for i = 1:nPoints 
  for j = 1:radialBases
    radialBasisOutputs(j) = radialBasisFunction(unsupervisedWeights,dataSet(i,2:3),j);
  end
  b(i)=supervisedWeights*radialBasisOutputs-Threshold;
end
output=tanh(beta*b);
diff=dataSet(:,1)-sign(output);

error=sum(abs(diff))/(2*nPoints);
end