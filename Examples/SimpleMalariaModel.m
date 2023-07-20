clear all;
close all;
addpath('Solver');

% randSeed = randSeed+1;
rng(26)

%                           (3)
%                       mOmega*R_H
%      -------------------------------------------------
%      |                                               |
%      v          (1)                      (2)         v
%   ------  mB*mBeta*I_M*S_H   ------    mGamma*I_H    ------
%   | S_H |       --->         | I_H |      --->      | R_H |        
%   ------         ^            ------                 ------
%               ...:        	   :   
%               :                  .....               
%               :                      :         
%            ------   (4)     ------   v (5)   ------   (6)
%           | I_M |    <---  | E_M |  <-----  | S_M | <-- mBirth*N_M
%           ------   mG*E_M   ------  mB*I_H   ------   
%                 |              |               | 
%                 v              v               v
%             mDeath*I_M     mDeath*E_M      mDeath*S_M
%               (7)              (8)            (9)
%
% These define the rates of the system
mB = 1;          % 10 bites a day 
mBeta = 0.5;      % Successful transfer of parasite
mGamma = 1/7;     % 1 week to recover
mOmega = 1/(7); % 1 yeah of immunity
mG = 0.5;
mDeath = 0.25; 
mBirth = mDeath;

% These are the initial conditions
NH0 = 10^2;
RH0 = 0;
IH0 = 20;
SH0 = NH0-RH0-IH0;

NM0 = 10^4;
IM0 = 0.25*NM0;
EM0 = 0.25*NM0;
SM0 = 0.5*NM0;

X0 = [SH0;IH0;RH0;SM0;EM0;IM0];

% How long to simulate for
tFinal = 100;

% These are solver options
dt = 10^(-5);
SwitchingThreshold = [0.2; 20];

% kinetic rate parameters
%              [1   2      3       4       5   6       7     ]
kConsts =      [mB; mBeta; mGamma; mOmega; mG; mBirth; mDeath];
%              [1          2     3     4     5     6                7     8     9]
kTime = @(p,t) [p(1)*p(2); p(3); p(4); p(5); p(1); p(6)*(1+cos(t)); p(7); p(7); p(7)];

                     
% stoichiometric matrix
%   [S_H; I_H; R_H; S_M; E_M; I_M];
nu = [-1    1    0    0    0   0;  % 1
      0    -1    1    0    0   0;  % 2
      1     0   -1    0    0   0;  % 3
      0     0    0    0   -1   1;  % 4
      0     0    0   -1    1   0;  % 5
      0     0    0    1    0   0;  % 6
      0     0    0    0    0  -1;  % 7
      0     0    0    0   -1   0;  % 8
      0     0    0   -1    0   0]; % 9
    
% propensity function
% Rates :: X -> rates -> propensities
rates = @(X,k) k.*[(X(1)*X(6));   % 1
                X(2);             % 2
                X(3);             % 3
                X(5);             % 4
                X(6);             % 5
                X(4)+X(5)+X(6);   % 6
                X(6);             % 7
                X(5);             % 8
                X(4)];            % 9

% identify which reactions are discrete and which are continuous
DoDisc = [0; 0; 0; 0; 0; 0];
% allow S and I to switch, but force R to be continuous
EnforceDo = [1; 1; 1; 1; 1; 1];
% allow I to switch, but force S and R to be continuous
% EnforceDo = [1; 0; 1];

%%
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

tic;
[X,TauArr] = GeneralisedSolverSwitchingRegimes(CompartmentSystem);
toc;

%%

figure;
subplot(1,2,1)
hold on;
plot(TauArr,X(1,:),'.','linewidth',1.5)
plot(TauArr,X(2,:),'.','linewidth',1.5)
plot(TauArr,X(3,:),'.','linewidth',1.5)
legend('S_H','I_H','R_H')
axis([0 tFinal 0 1.1*NH0])
hold off;

subplot(1,2,2)
hold on;
plot(TauArr,X(4,:),'.','linewidth',1.5)
plot(TauArr,X(5,:),'.','linewidth',1.5)
plot(TauArr,X(6,:),'.','linewidth',1.5)
legend('S_M','E_M','I_M')
axis([0 tFinal 0 1.1*NM0])
hold off
