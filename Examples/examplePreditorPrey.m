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
mDeathA = 0.001;
mBirthA = 1.5;

mDeathB = 0.8;
mBirthB = 0.01;


% These are the initial conditions
A0 = 10^3;
B0 = 5;

% How long to simulate for
tFinal = 200;

% These are solver options
dt = 10^(-2);
SwitchingThreshold = [0.2; 10];

% Initial condition
X0 = [A0;B0];

% stoichiometric matrix
nu = [-1, 1;
      -1, 1;
       1, 0;
      0, -1];
nuProduct = [ 0, 1;
              0, 1;
              1, 0;
              0, 0];
nuReactant= [ 1, 0;
              1, 0;
              0, 0;
              0, 1];

% propensity function
k = [mDeathA; mBirthB;   mBirthA; mDeathB;];
rates = @(X, t) k.*[(X(1)*X(2));
                    (X(1)*X(2));
                    X(1);
                    X(2)];

% identify which reactions are discrete and which are continuous
DoDisc = [0; 0];
EnforceDo = [0; 0];

%%

stoich = struct();
stoich.nu = nu;
stoich.nuReactant = nuReactant;
stoich.nuProduct = nuProduct;

stoich.DoDisc = DoDisc;
solTimes = 0:dt:tFinal;
myOpts = struct();
myOpts.EnforceDo = EnforceDo;
myOpts.dt = dt;
myOpts.SwitchingThreshold = SwitchingThreshold;


tic;
% profile on
[X,TauArr] = JumpSwitchFlowSimulator(X0, rates, stoich, solTimes, myOpts);
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
