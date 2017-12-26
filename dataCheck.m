clc, clear, close all


x = linspace(0,100, 1000);
s = 10*sin(x);
c = 10*cos(x);

y = s.*c;
plot(y)

