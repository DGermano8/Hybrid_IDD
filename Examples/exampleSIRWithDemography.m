% clear all;
% close all;
addpath('Solver');
% clc
% randSeed = randSeed+1;
rng(6)
%           --------------------------------------------------
%           |                   mWane*R                      |
%           v                                                |
% mBirth*N -----   mBeta*I*S/N     -----      mGamma*I     -----
%     ---> | S |       --->        | I |       --->        | R |
%          -----                   -----                   -----
%          | mDeath*S              | mDeath*I              | mDeath*R
%          V                       V                       V
%

% These define the rates of the system
mBeta = 2/7; % Infect "___" people a week
mGamma = 0.6/7; % infecion for "___" weeks
mDeath = 1/(2.0*365); %lifespan
mBirth = mDeath;
mWane = 0/(2.0*365);

R_0 = mBeta/(mGamma+mDeath)

% These are the initial conditions
N0 = 10^5;
I0 = 20;
R0 = 0;
S0 = N0-I0-R0;

% How long to simulate for
tFinal = 100;

% These are solver options
dt = 10^(-3);
SwitchingThreshold = [1; 1000];

% kinetic rate parameters
X0 = [S0;I0;R0];


% reactant stoichiometries
nuMinus = [1,1,0;
           0,1,0;
           0,0,0;
           1,0,0;
           0,1,0;
           0,0,1;
           0,0,1];

% product stoichiometries
nuPlus = [0,2,0;
          0,0,1;
          1,0,0;
          0,0,0;
          0,0,0;
          0,0,0;
          1,0,0];
% stoichiometric matrix
nu = nuPlus - nuMinus;

% propensity function
k = [mBeta; mGamma; mBirth; mDeath; mDeath; mDeath; mWane];
rates = @(X,t) k.*[(X(1)*X(2))/(X(1)+X(2)+X(3));
                 X(2);
                 X(1)+X(2)+X(3);
                 X(1);
                 X(2);
                 X(3);
                 X(3)];
             
DXDT = @(X,t) [-mBeta*(X(1)*X(2))/(X(1)+X(2)+X(3)) - mDeath*X(1) + mBirth*(X(1)+X(2)+X(3)) + mWane*X(3);
                mBeta*(X(1)*X(2))/(X(1)+X(2)+X(3)) - mGamma*X(2) - mDeath*X(2);
                mGamma*X(2)-mDeath*X(3) - mWane*X(3)];

% identify which reactions are discrete and which are continuous
DoDisc = [0; 0; 0];
% allow S and I to switch, but force R to be continuous
EnforceDo = [1; 1; 1];
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

randM = 9;
% profile on
tic;
rng(randM)
[X,TauArr] = JumpSwitchFlowSimulator_FE_Trap(X0, rates, stoich, solTimes, myOpts);
toc; 
% figure;
subplot(1,2,1)
hold on;
plot(TauArr(1:1:end),X(1,(1:1:end)),'.-','linewidth',1.5)
plot(TauArr(1:1:end),X(2,(1:1:end)),'.-','linewidth',1.5)
plot(TauArr(1:1:end),X(3,(1:1:end)),'.-','linewidth',1.5)
legend('S','I','R')
axis([0 tFinal 0 1.1*N0])
hold off;

subplot(1,2,2)
hold on;
plot(X(1,(1:1:end)),X(2,(1:1:end)),'.-','linewidth',1.0)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel('I')
xlabel('S')

tic
rng(randM)
[X,TauArr] = JumpSwitchFlowSimulator_RK4_Simps(X0, rates, stoich, solTimes, myOpts);
toc;

% profile off
% profile viewer
% NOTE when I profiled this, it looked like calls to the rate function
% where the most expensive part of the evaluation so I don't think
% that the allocation of a dynamic array is going to be a problem.
% There are stacks and exponentially growing allocations if this ends
% up changing.

%%

% figure;
subplot(1,2,1)
hold on;
plot(TauArr(1:1:end),X(1,(1:1:end)),'.','linewidth',2)
plot(TauArr(1:1:end),X(2,(1:1:end)),'.','linewidth',2)
plot(TauArr(1:1:end),X(3,(1:1:end)),'.','linewidth',2)
legend('S','I','R')
axis([0 tFinal 0 1.1*N0])
hold off;

subplot(1,2,2)
hold on;
plot(X(1,(1:1:end)),X(2,(1:1:end)),'.-','linewidth',1.0)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel('I')
xlabel('S')

% figure; histogram(diff(TauArr),[0:10^(-4):dt])
% set(gca, 'YScale', 'log')
drawnow
