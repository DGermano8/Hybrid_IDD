% clear all;
close all;
addpath('Solver');

% randSeed = randSeed+1;
rng(26)

%
%   | mBirth*N
%   v
%   -----   mBeta*I*S/N     -----      mGamma*I     -----
%   | S |       --->        | I |       --->        | R |
%   -----                   -----                   -----
%   | mDeath*S              | mDeath*I              | mDeath*R
%   V                       V                       V
%

% These define the rates of the system
mBeta = 1.45/7; % Infect "___" people a week
mGamma = 0.4/7; % infecion for "___" weeks
mDeath = 1/(2*365); %lifespan
mBirth = mDeath;

R_0 = mBeta/(mGamma+mDeath);

% These are the initial conditions
N0 = 10^5;
I0 = 2;
R0 = 0;
S0 = N0-I0-R0;

% How long to simulate for
tFinal = 1000;

% These are solver options
dt = 10^(-3);
SwitchingThreshold = [0.2; 20];

% kinetic rate parameters
kConsts =      [mBeta; mGamma;   mBirth;     mDeath;      mDeath;      mDeath];
kTime = @(p,t) [p(1); p(2); p(3); p(4); p(5); p(6)];
X0 = [S0;I0;R0];


% reactant stoichiometries
nuMinus = [1,1,0;
           0,1,0;
           0,0,0;
           1,0,0;
           0,1,0;
           0,0,1];

% product stoichiometries
nuPlus = [0,2,0;
          0,0,1;
          1,0,0;
          0,0,0;
          0,0,0;
          0,0,0];
% stoichiometric matrix
nu = nuPlus - nuMinus;

% propensity function
% Rates :: X -> rates -> propensities
rates = @(X) k.*[(X(1)*X(2))/(X(1)+X(2)+X(3));
                 X(2);
                 X(1)+X(2)+X(3);
                 X(1);
                 X(2);
                 X(3)];

% identify which reactions are discrete and which are continuous
DoDisc = [0; 1; 0];
% allow S and I to switch, but force R to be continuous
EnforceDo = [0; 0; 1];
% allow I to switch, but force S and R to be continuous
% EnforceDo = [1; 0; 1];

%%
stoich = struct();
stoich.nu = nu;
stoich.DoDisc = DoDisc;
solTimes = 0:dt:tFinal;
myOpts = struct();
myOpts.EnforceDo = EnforceDo;
myOpts.dt = dt;
myOpts.SwitchingThreshold = SwitchingThreshold;


tic;
% profile on
[X,TauArr] = cdsSimulator(X0, rates, stoich, solTimes, myOpts);
% profile off
% profile viewer
% NOTE when I profiled this, it looked like calls to the rate function
% where the most expensive part of the evaluation so I don't think
% that the allocation of a dynamic array is going to be a problem.
% There are stacks and exponentially growing allocations if this ends
% up changing.
toc;

%%

figure;
subplot(1,2,1)
hold on;
plot(TauArr,X(1,:),'.','linewidth',1.5)
plot(TauArr,X(2,:),'.','linewidth',1.5)
plot(TauArr,X(3,:),'.','linewidth',1.5)
legend('S','I','R')
axis([0 tFinal 0 1.1*N0])

hold off;

subplot(1,2,2)
plot(X(1,:),X(2,:),'.','linewidth',1.5)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel('I')
xlabel('S')
