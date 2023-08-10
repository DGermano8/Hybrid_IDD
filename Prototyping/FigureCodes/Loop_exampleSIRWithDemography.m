% clear all;
% close all;
% addpath('Solver');

% randSeed = randSeed+1;
% rng(3)
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
mDeath = 1/(1.75*365); %lifespan
mBirth = mDeath;

R_0 = mBeta/(mGamma+mDeath)

% These are the initial conditions
N0 = 10^5;
I0 = 2;
R0 = 0;
S0 = N0-I0-R0;

% How long to simulate for
tFinal = 1000;

% These are solver options
dt = 10^(-2);
SwitchingThreshold = [10; 1*round(10^(2))];

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
EnforceDo = [0; 0; 0];
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

numbOfNoTakeOff = 0;
numbOfFadeOut = 0;
numbOfEndemic = 0;
numbSims = 50;
tic;
for ii=1:numbSims
rng(ii)
% tic;
[X,TauArr] = JumpSwitchFlowSimulator_FE_Trap(X0, rates, stoich, solTimes, myOpts);
% toc;
ii
    if(X(2,end) > 0)
        TimePlot = 3;
        PhasePlot = 6;
        numbOfEndemic = numbOfEndemic + 1;
    elseif(max(X(2,:)) < 20)
        TimePlot = 1;
        PhasePlot = 4;
        numbOfNoTakeOff = numbOfNoTakeOff + 1;
    else
        TimePlot = 2;
        PhasePlot = 5;
        numbOfFadeOut = numbOfFadeOut + 1;
    end
    subplot(2,3,TimePlot)
    hold on;
    plot(TauArr(1:100:end),X(1,1:100:end),'.','linewidth',1.5,'color',[0.1 0.1 0.75])
    plot(TauArr(1:100:end),X(2,1:100:end),'.','linewidth',1.5,'color',[0.75 0.1 0.1])
    plot(TauArr(1:100:end),X(3,1:100:end),'.','linewidth',1.5,'color',[0.1 0.725 0.1])
    legend('S','I','R')
    axis([0 tFinal 0 1.1*N0])
    hold off;

    subplot(2,3,PhasePlot)
    hold on;
    plot(X(1,:),X(2,:),'.-','linewidth',1.0)
    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
    ylabel('I')
    xlabel('S')

    
end
toc;
subplot(2,3,1)
title(['Prob of Extinct = ' num2str(numbOfNoTakeOff/numbSims)])

subplot(2,3,2)
title(['Prob of Fade Out = ' num2str(numbOfFadeOut/numbSims)])

subplot(2,3,3)
title(['Prob of Endemic = ' num2str(numbOfEndemic/numbSims)])
sgtitle(['SIR with Demography, Jump threshold = ' num2str(SwitchingThreshold(2))])
% exportgraphics(f,'SIR_Dem.png')%,'Resolution',500)

