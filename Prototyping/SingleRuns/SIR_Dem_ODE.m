

clear all;
close all;
myplot = @semilogy;

%Time unit is days
b = 2.0/7; % Infect 2 people a week
g = 1/7; % infecion for 1 week
k = 1/(10*365); %lifespane 

R_0 = b/(g+k)
% tspan = linspace(0,1.5/k,100000);
tspan = linspace(0,5000,100000);

[t,y] = ode45(@(t,y)sir(t,y,b,g,k),tspan,[0.99; 0.1; 0]);

A = 1/(k*(R_0-1));
G = 1/(k+g);

figure;
subplot(1,2,1)
hold on
plot(t,y(:,1),...
     t,y(:,2),...
     t,y(:,3),'linewidth',1.5)
% plot(tspan,exp(tspan*(-k*R_0/2)).*cos(tspan/sqrt(A*G)))
legend('s','i','r')
title('Deterministic')
hold off

subplot(1,2,2)
plot(y(:,1),y(:,2),'linewidth',1.5)
ylabel('I')
xlabel('S')



%%
function dydt = sir(t,y,b,g,k)

% y(1)->s; y(2)->i; y(3)->r

dydt = [-b*y(1)*y(2) + k - k*y(1);
        b*y(2)*y(1) - g*y(2) - k*y(2);
        g*y(2) - k*y(3)];
end