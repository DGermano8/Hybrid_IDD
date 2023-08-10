% clear all;
% close all;
% addpath('Solver');

% randSeed = randSeed+1;
rng(2)
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
mDeath = 1/(1*365); %lifespan
mBirth = mDeath;

R_0 = mBeta/(mGamma+mDeath)

% These are the initial conditions
N0 = 10^5;
I0 = 2;
R0 = 0;
S0 = N0-I0-R0;

% How long to simulate for
tFinal = 1500;

% These are solver options
dt = 10^(-3);
SwitchingThreshold = [0.2; 20];

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

% tic;
[X,TauArr,DiscTimes] = MovingFEMesh_cdsSimulator(X0, rates, stoich, solTimes, myOpts);
% toc;
%%
subplot(2,3,1)
title(['S'])
plot(TauArr,X(1,:),'.','linewidth',1.5,'color',[0.1 0.1 0.75])
axis([254.5 255 44040 44075])

subplot(2,3,2)
title(['I'])
plot(TauArr,X(2,:),'.','linewidth',1.5,'color',[0.75 0.1 0.1])
axis([254.5 255 1315 1328])

subplot(2,3,3)
title(['R'])
plot(TauArr,X(3,:),'.','linewidth',1.5,'color',[0.1 0.725 0.1])
axis([254.5 255 54610 54636])


subplot(2,3,4)
title(['No. Discrete Events per 100 Continous'])
tau = DiscTimes(1,:);
[val,pos] = max(tau);
tau(pos+1:end) = [];
histogram(tau, 0:100*dt:tFinal)
axis([254.5 255 0 10])

subplot(2,3,5)
title(['No. Discrete Events per 100 Continous'])
tau = DiscTimes(2,:);
[val,pos] = max(tau);
tau(pos+1:end) = [];
histogram(tau, 0:100*dt:tFinal)
axis([254.5 255 0 12])

subplot(2,3,6)
title(['No. Discrete Events per 100 Continous'])
tau = DiscTimes(3,:);
[val,pos] = max(tau);
tau(pos+1:end) = [];
histogram(tau, 0:100*dt:tFinal)
axis([254.5 255 0 2])

% sgtitle('No. Discrete Events per 100 Continous')
% exportgraphics(f,'SIR_Dem_DiscTimes.png')%,'Resolution',500)

