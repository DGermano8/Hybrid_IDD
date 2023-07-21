clear all;
close all;
addpath('Solver');
%   | mBirthA*A
%   | 
%   v       mDeathA*A*B  
%   -----      --->       -----       
%   | A |                 | B |      
%   -----      --->       -----      
%           mBirthB*A*B     |
%                           | mDeathB*B 
%                           V          
%   
%   A = Prey
%   B = Preditor

% These define the rates of the system
mDeathA = 0.01;
mBirthA = 1.1;

mDeathB = 0.8;
mBirthB = 0.01;


% These are the initial conditions
A0 = 10^2;
B0 = 5;

% How long to simulate for
tFinal = 100;

% These are solver options
dt = 10^(-3);
SwitchingThreshold = [0.2; 20];

% mDeathA rate parameters
kConsts = [mDeathA; mBirthB;   mBirthA;     mDeathB;];
kTime = @(p,t) [p(1); p(2); p(3); p(4)];
X0 = [A0;B0];

% stoichiometric matrix
nu = [-1, 1;
      -1, 1;
       1, 0;
      0, -1];

% propensity function
rates = @(X) k.*[(X(1)*X(2));
                 (X(1)*X(2));
                 X(1);
                 X(2)];

% identify which reactions are discrete and which are continuous
DoDisc = [0; 0];
EnforceDo = [1; 1];

%%
<<<<<<< HEAD:Examples/examplePreditorPrey.m
stoich = struct();
stoich.nu = nu;
stoich.DoDisc = DoDisc;
solTimes = 0:dt:tFinal;
myOpts = struct();
myOpts.EnforceDo = EnforceDo;
myOpts.dt = dt;
myOpts.SwitchingThreshold = SwitchingThreshold;
=======
CompartmentSystem  = struct();

CompartmentSystem.X0 =X0;
CompartmentSystem.tFinal = tFinal;
CompartmentSystem.kConsts = kConsts;
CompartmentSystem.kTime = kTime;
CompartmentSystem.rates = rates;
CompartmentSystem.nu = nu;
CompartmentSystem.DoDisc = DoDisc;
CompartmentSystem.EnforceDo = EnforceDo;
CompartmentSystem.dt = dt;
CompartmentSystem.SwitchingThreshold = SwitchingThreshold;
>>>>>>> main:Examples/PreditorPrey.m

tic;
% profile on
[X,TauArr] = cdsSimulator(X0, rates, stoich, solTimes, myOpts);
% profile off
% profile viewer
toc;

%%

figure;
subplot(1,2,1)
hold on;
plot(TauArr,sqrt(X(1,:)),'.','linewidth',1.5)
plot(TauArr,sqrt(X(2,:)),'.','linewidth',1.5)
legend('Prey','Preditor')
% axis([0 tFinal 0 1.1*N0])

hold off;

subplot(1,2,2)
plot(X(1,:),X(2,:),'.','linewidth',1.5)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel('Preditor')
xlabel('Prey')
