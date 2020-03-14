function value=NeighbourhoodFunction2(i,i0,sigma)

value=exp(-norm(i-i0)^2/(2*sigma^2));
end