function  [S,R, time , I,tau] = SIR_Dem_TauLeaping(mBeta, mGamma, mDeath, N0, I0, dt, t_final, RND_SEED)
%% The Tau Leaping Method
rng(RND_SEED);
% An approximate stochastic simulation algorithm
%
% Inputs:
%    bcrn - a biochemical reaction network struct
%    T    - the end time of the simulation
%    tau   - the timestep
% Outputs:
%    Z    -  time series of copy number vectors
%    t    -  vector of times
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


% initialise
Nt = floor(t_final/dt);
Z = zeros(length(bcrn.X0),Nt+1);
t = zeros(1,Nt+1);
Z(:,1) = bcrn.X0;

for i=1:Nt
    % compute propensities
    a = bcrn.a(Z(:,i),bcrn.k); a(a < 0 ) = 0;
    % generate poisson variates
    Y = poissrnd(a*dt);
    % update copy numbers
    Z(:,i+1) = Z(:,i) + (bcrn.nu') * Y;
    t(i+1) = t(i) + dt;
end
S = Z(1,:);
R = Z(3,:);
time = t;
I = Z(2,:);
tau = t;
