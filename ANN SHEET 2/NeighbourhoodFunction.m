function value=NeighbourhoodFunction(i,i0,sigma)
value=exp(-abs(i-i0)^2/(2*sigma^2));
end