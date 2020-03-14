function input=GetInput
whileCondition=0;
while whileCondition==0
  x=rand(1,2);
  if (sqrt(3)*x(1)>=x(2)) && (-sqrt(3)*x(1)+sqrt(3)>=x(2))
    input=x;
    whileCondition=1;
  end
end
end
