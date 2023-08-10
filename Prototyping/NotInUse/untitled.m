% clear all;
% close all;
clc;
addpath('Solver');
% clc
% randSeed = randSeed+1;
rng(3)
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
mBeta = 0.5; % Infect "___" people a week


% These are the initial conditions
N0 = 10^4;
I0 = 20;
S0 = N0-I0;

% How long to simulate for
tFinal = 200;

% These are solver options
dt = 10^(-2);
SwitchingThreshold = [0.5; 1000];

% kinetic rate parameters
X0 = [S0;I0];


% reactant stoichiometries
nuReactant = [1,1];

% product stoichiometries
nuProduct = [0,2];
% stoichiometric matrix
nu = nuProduct - nuReactant;

% propensity function
k = [mBeta];
rates = @(X,t) k.*[(X(1)*X(2))/(X(1)+X(2))];
             
% identify which reactions are discrete and which are continuous
DoDisc = [1; 1];
% allow S and I to switch, but force R to be continuous
EnforceDo = [1; 1];
% allow I to switch, but force S and R to be continuous
% EnforceDo = [1; 0; 1];

%%
stoich = struct();
stoich.nu = nu;
stoich.nuReactant = nuReactant;
stoich.DoDisc = DoDisc;
solTimes = 0:dt:tFinal;
myOpts = struct();
myOpts.EnforceDo = EnforceDo;
myOpts.dt = dt;
myOpts.SwitchingThreshold = SwitchingThreshold;

% randM = 6;
% profile on
tic;
% rng(randM)
% [X,TauArr] = JumpSwitchFlowSimulator(X0, rates, stoich, solTimes, myOpts, "FE");
[X,TauArr] = JumpSwitchFlowSimulator_FE(X0, rates, stoich, solTimes, myOpts);
% profile off
% profile viewer

toc; 
% clf;
% figure;
subplot(1,2,1)
hold on;
plot(TauArr(1:1:end),X(1,(1:1:end)),'.-','linewidth',1.5)
plot(TauArr(1:1:end),X(2,(1:1:end)),'.-','linewidth',1.5)
legend('S','I')
axis([0 tFinal 0 1.1*N0])
hold off;

subplot(1,2,2)
hold on;
plot(X(1,(1:1:end)),X(2,(1:1:end)),'.-','linewidth',1.0)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel('I')
xlabel('S')

%%

% this writes the data to a .csv file to use for later.
% mex_WriteMatrix('Results/SIR_test_2.csv',[TauArr' X'],'%10.10f',',','w+');
% % free mex function
% clear mex_WriteMatrix;



%%

% tic
rng(randM)
[XG,TauArrG] = GillespieDirectMethod(X0, rates, stoich, solTimes, myOpts);
toc;

% profile off
% profile viewer
% NOTE when I profiled this, it looked like calls to the rate function
% where the most expensive part of the evaluation so I don't think
% that the allocation of a dynamic array is going to be a problem.
% There are stacks and exponentially growing allocations if this ends
% up changing.

%%

% % figure;
subplot(1,2,1)
hold on;
plot(TauArrG(1:1:end),XG(1,(1:1:end)),'.','linewidth',2)
plot(TauArrG(1:1:end),XG(2,(1:1:end)),'.','linewidth',2)
legend('S','I')
axis([0 tFinal 0 1.1*N0])
hold off;

subplot(1,2,2)
hold on;
plot(XG(1,(1:1:end)),XG(2,(1:1:end)),'.-','linewidth',1.0)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel('I')
xlabel('S')

% figure; histogram(diff(TauArr),[0:10^(-4):dt])
% set(gca, 'YScale', 'log')
drawnow
