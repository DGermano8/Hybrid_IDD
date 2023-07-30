% clear all;
% close all;
addpath('Solver');

% randSeed = randSeed+1;
rng(1)
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
mBeta = 1.0/7; % Infect "___" people a week
mGamma = 0.6/7; % infecion for "___" weeks
mDeath = 1/(3*365); %lifespan
mBirth = mDeath;

R_0 = mBeta/(mGamma+mDeath)

% These are the initial conditions
N0 = 10^7;
I0 = 2;
R0 = 0;
S0 = N0-I0-R0;

% How long to simulate for
tFinal = 500;

% These are solver options
dt = 10^(-2);
SwitchingThreshold = [10; round(10^(2))];

% kineticate parameters
X0 = [S0;I0;R0];

% kinetic rate parameters
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
k = [mBeta; mGamma; mBirth; mDeath; mDeath; mDeath];
rates = @(X,t) k.*[(X(1)*X(2))/(X(1)+X(2)+X(3));
                 X(2);
                 X(1)+X(2)+X(3);
                 X(1);
                 X(2);
                 X(3)];

% identify which reactions are discrete and which are continuous
DoDisc = [0; 0; 0];
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

f=figure;

tic;
[X,TauArr] = Copy_2_of_MovingFEMesh_cdsSimulator(X0, rates, stoich, solTimes, myOpts);

toc;

subplot(1,2,1)
hold on;
plot(TauArr,X(1,:),'.','linewidth',1.5,'color',[0.1 0.1 0.75])
plot(TauArr,X(2,:),'.','linewidth',1.5,'color',[0.75 0.1 0.1])
plot(TauArr,X(3,:),'.','linewidth',1.5,'color',[0.1 0.725 0.1])
legend('S','I','R')
axis([0 tFinal 0 1.1*N0])
hold off;

subplot(1,2,2)
hold on;
plot(X(1,:),X(2,:),'.-','linewidth',1.0)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel('I')
xlabel('S')



