function radialBasisOutput = radialBasisFunction(weights,randomPattern,index)

k=size(weights,1);
denominator=zeros(k,1);

for i = 1:k
    denominator(i)=exp((-norm(randomPattern-weights(i,:))^2)/2);
end

radialBasisOutput = exp((-norm(randomPattern-weights(index,:))^2)/2)/sum(denominator);

end