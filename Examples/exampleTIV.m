% clear all;
close all;
addpath('Solver');

% randSeed = randSeed+1;
rng(4)
%                   (1)                (2)
%             mRectuite_Z*I*Z ----- mDeath_Z*Z
%                        ---> | Z | --->
%     (3)                ^    -----
%   | mBirth_T           :      :
%   V          (4)       :      :
%   -----   mBeta*T*V   -----   V      (5)
%   | T |       --->    | I |  ---> mKilling*I*Z
%   -----         ^     -----                   
%   | mDeath_T*T    :     :   | mDeath_I*I              
%   V   (6)         :     :   V    (7) 
%                 -----   :
%                 | V | <----
%                 -----   mProd_V*I
%        mDeath_V*V |         (8)
%            (9)    V
%

% These define the rates of the system
mRectuite_Z = 1/2;
mDeath_Z = 1/14;
mBirth_T = 1/7;
mBeta = 0.05;
mKilling = 1/3;
mDeath_T = 1/7;
mDeath_I = 1/3.5;
mProd_V = 1/1;
mDeath_V = 1/2;

% These are the initial conditions
T0 = 10^4;
I0 = 0;
V0 = 10;
Z0 = 1;

% How long to simulate for
tFinal = 4*14;

% These are solver options
dt = 10^(-3);
SwitchingThreshold = [0.2; 20];

% kinetic rate parameters
X0 = [T0;I0;V0;Z0];

% stoichiometric matrix
%   [ T;    I;   V;   Z];
nu = [0     0    0    1;  % 1
      0     0    0   -1;  % 2
      1     0    0    0;  % 3
     -1     1    0    0;  % 4
      0    -1    0    0;  % 5
     -1     0    0    0;  % 6
      0    -1    0    0;  % 7
      0     0    1    0;  % 8
      0     0   -1    0]; % 9
  
% kinetic rate parameters
kConsts =      [mRectuite_Z; mDeath_Z; mBirth_T; mBeta; mKilling; mDeath_T; mDeath_I; mProd_V; mDeath_V];
%              [1     2     3     4     5     6     7     8     9]
k = @(p) [p(1); p(2); p(3); p(4); p(5); p(6); p(7); p(8); p(9)];

rates = @(X,t) k(kConsts).*[X(2)*X(4);
                   X(4);
                   1;
                   X(1)*X(3);
                   X(2)*X(4);
                   X(1);
                   X(2)
                   X(2)
                   X(3)];

% identify which reactions are discrete and which are continuous
DoDisc =    [0; 0; 0; 0];
EnforceDo = [0; 0; 0; 0];

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
[X,TauArr] = MovingFEMesh_cdsSimulator(X0, rates, stoich, solTimes, myOpts);
% [X,TauArr] = cdsSimulator(X0, rates, stoich, solTimes, myOpts);
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
subplot(3,1,1)
hold on;
plot(TauArr,X(1,:),'.','linewidth',1.5)
plot(TauArr,X(2,:),'.','linewidth',1.5)
hold off;
legend('T','I')
axis([0 tFinal 0 max(max(X(1:2,:)))])

subplot(3,1,2)
plot(TauArr,X(3,:),'.','linewidth',1.5)
legend('V')
axis([0 tFinal 0 max((X(3,:)))])

subplot(3,1,3)
plot(TauArr,X(4,:),'.','linewidth',1.5)
legend('Z')
axis([0 tFinal 0 max((X(4,:)))])

