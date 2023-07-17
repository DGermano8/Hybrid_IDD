mBeta = 2.0/7; % Infect 2 people a week
mGamma = 0.75/7; % infecion for 1 week
mDeath = 2/(2*365); %lifespan
mBirth = mDeath;

%   mBeta,mGamma,mBirthS,mDeathS,mDeathI,mDeathR
k = [mBeta; mGamma;   mBirth;     mDeath;      mDeath;      mDeath];
N0 = 10000;
I0 = 0.1*N0;
S0 = N0-I0;
R0 = 0;
TFinal = 1000;

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
                


%%
tic;
% [X,t] = GillespieDirectMethod(bcrn,TFinal);
[X,t] = ModifiedNextReactionMethod(bcrn,TFinal);
% [X,t] = TauLeapingMethod(bcrn,TFinal,0.1);
toc;

% Plot
figure;
Xn = reshape([X;X],size(X).*[1,2]); Xn(:,end) = [];
tn = reshape([t;t],[1,2*length(t)]); tn(1) = [];
plot(tn,Xn,'LineWidth',2); xlim([0,TFinal]); ylim([0,max(sum(Xn(:,:),1))]);
xlabel('t'); ylabel('Number of people');
legend({'S','I','R'});