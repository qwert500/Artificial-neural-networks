function GenerateEquilateralTriangle
triangle1=linspace(0,1);
triangle2=linspace(0,0.5);
triangle3=linspace(0.5,1);
hold on
plot(triangle1,0*triangle1,'r')
plot(triangle2,sqrt(3)*triangle2,'r')
plot(triangle3,-sqrt(3)*triangle3+sqrt(3),'r')