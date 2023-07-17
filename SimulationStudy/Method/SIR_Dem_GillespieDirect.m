function [S,R, time , I,tau] = SIR_Dem_GillespieDirect(mBeta, mGamma, mDeath, N0, I0, t_final, RND_SEED)
%% Gillespie Direct Method

rng(RND_SEED);

% An exact stochastic simulation algorithm

% Original Author:
%   David J. Warne (david.warne@qut.edu.au)
%         School of Mathematical Sciences
%         Queensland University of Technology

mBirth = mDeath;
k = [mBeta; mGamma;   mBirth;     mDeath;      mDeath;      mDeath];
S0 = N0-I0;
R0 = 0;

bcrn = struct();
% kinetic rate parameters
bcrn.k = k;       
% number of reactions
bcrn.M = length(k);                
% number of chemical species
bcrn.N = 3;                        
% reactant stoichiometries
bcrn.nu_minus = [1,1,0;
                 0,1,0;
                 0,0,0;
                 1,0,0;
                 0,1,0;
                 0,0,1];
% product stoichiometries
bcrn.nu_plus = [0,2,0;
                0,0,1;
                1,0,0;
                0,0,0;
                0,0,0;
                0,0,0];
% stoichiometric matrix
bcrn.nu = bcrn.nu_plus - bcrn.nu_minus;
% initial copy numbers
bcrn.X0 = [S0;I0;R0];
% propensity function
bcrn.a = @(X,k) k.*[(X(1)*X(2))/(X(1)+X(2)+X(3));
                    X(2);
                    X(1)+X(2)+X(3);
                    X(1);
                    X(2);
                    X(3)];

% initialise
X = [bcrn.X0];
t = [0];
while true
    % compute propensities
    a = bcrn.a(X(:,end),bcrn.k);
    % sample exponential waiting time
    dt = exprnd(1/sum(a));
    % check if the simulation is finished
    if t(end) + dt <= t_final
        % sample the next reaction event
        j = randsample(bcrn.M,1,true,a);
        % update copy numbers  
        X = [X,X(:,end) + bcrn.nu(j,:)'];
        t = [t, t(end) + dt];
    else
        break;
    end
end
S = X(1,:);
R = X(3,:);
time = t;
I = X(2,:);
tau = t;
