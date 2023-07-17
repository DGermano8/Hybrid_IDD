function  [S,R, time , I,tau] = SIR_Dem_ModifiedNextReaction(mBeta, mGamma, mDeath, N0, I0, t_final, RND_SEED)
%% Modified Next Reaction Method
rng(RND_SEED);

% An exact stochastic simulation algorithm
%
% Inputs:
%    bcrn - a biochemical reaction network struct
%    T    - the end time of the simulation
% Outputs:
%    X    -  time series of copy number vectors
%    t    -  vector of reaction times
%
% Author:
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

%initialise
X = [bcrn.X0];
t = [0];
T_r = zeros(bcrn.M,1);
% generate M unit-time exponential variates
P = exprnd(1,[bcrn.M,1]);
while true
    % compute propensities
    a = bcrn.a(X(:,end),bcrn.k);
    % determine which reaction channel fires next
    dt = (P - T_r) ./ a;
    dt(a <= 0) = Inf;
    [delta,mu] = min(dt);
    if t(end) + delta <= t_final
        %update copy numbers
        X = [X,X(:,end) + bcrn.nu(mu,:)'];
        t = [t,t(end) + delta];
        T_r = T_r + a*delta;
        % update next reaction time for the firing channel
        P(mu) = P(mu) + exprnd(1);
    else
        break;
    end
end

S = X(1,:);
R = X(3,:);
time = t;
I = X(2,:);
tau = t;
