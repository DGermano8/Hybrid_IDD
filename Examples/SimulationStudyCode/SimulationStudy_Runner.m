% clear all;
% close all;
addpath('Solver');

%
%
% mBirth*N -----   mBeta*I*S/N     -----      mGamma*I     -----
%     ---> | S |       --->        | I |       --->        | R |
%          -----                   -----                   -----
%          | mDeath*S              | mDeath*I              | mDeath*R
%          V                       V                       V
%

% These define the rates of the system
mBeta = 2/7; % Infect "___" people a week
mGamma = 0.6/7; % infecion for "___" weeks
mDeath = 1/(2.0*365); %lifespan
mBirth = mDeath;

R_0 = mBeta/(mGamma+mDeath);

% These are the initial conditions
N0 = 10^6;
I0 = 2;
R0 = 0;
S0 = N0-I0-R0;

% How long to simulate for
tFinal = 1500;

% These are solver options
dt = 10^(-2);
SwitchingThreshold = [0.5; 1000];

% kinetic rate parameters
X0 = [S0;I0;R0];

% reactant stoichiometries
nuReactant = [1,1,0;
              0,1,0;
              1,0,0;
              0,1,0;
              0,0,1;
              1,0,0;
              0,1,0;
              0,0,1];

% product stoichiometries
nuProduct = [0,2,0;
             0,0,1;
             2,0,0;
             1,1,0;
             1,0,1;
             0,0,0;
             0,0,0;
             0,0,0];
% stoichiometric matrix
nu = nuProduct - nuReactant;

% propensity function
k = [mBeta; mGamma; mBirth; mBirth; mBirth; mDeath; mDeath; mDeath];
% rates = @(X,t) k.*[(X(1)*X(2))/(X(1)+X(2)+X(3));
% rates = @(X,t) k.*[(X(1)*X(2));
rates = @(X,t) k.*[(X(1)*X(2))/(X(1)+X(2)+X(3));
                 X(2);
                 X(1);
                 X(2);
                 X(3);
                 X(1);
                 X(2);
                 X(3)];
             
% identify which reactions are discrete and which are continuous
DoDisc = [0; 0; 0];
% allow S and I to switch, but force R to be continuous
EnforceDo = [0; 0; 0];

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

% These are the initial conditions

I0 = 2;
R0 = 0;


N0_Vector = [2:0.25:6.25];
HowManySimsToDo = 100;

TimerData = [];
dtObserve = 10^(-2);
TimeObserve = 0:dtObserve:tFinal;
XObserve = zeros(3,length(TimeObserve));

%%
masterpath = 'CleanedData6';
% mex_WriteMatrix([masterpath,'/META.csv'],[] ,'%10.10f',',','w+');
% clear mex_WriteMatrix;

for N0ii = 4:0.25:6.25
    for ii=1:HowManySimsToDo
        N0 = round(10^(N0ii));
        S0 = N0-I0-R0;
        X0 = [S0;I0;R0];
        
%         rng(ii)
%         tic;
%         [XTau,TauArrTau] = TauLeapingMethod(X0, rates, stoich, solTimes, myOpts);
%         TauTimer = toc;
%         mex_WriteMatrix([masterpath,'/Tau/Tau_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv'],[TimeObserve' XTau'],'%10.10f',',','w+');
% 
%         clear mex_WriteMatrix;
        
        tic;
        rng(ii)
        [XJSF,TauArrJSG] = Copy_of_JumpSwitchFlowSimulator_FE(X0, rates, stoich, solTimes, myOpts);
        JumpSwitchFlowTimer = toc;
        
        [UniqueT, UniqueIndecies] = unique(TauArrJSG);
        XObserve = zeros(3,length(TimeObserve));
        for jj=1:3
            UniqueX = XJSF(jj,UniqueIndecies);
            XObserve(jj,:) = interp1(UniqueT,UniqueX, TimeObserve,'previous');
        end
        mex_WriteMatrix([masterpath,'/JumpSwitchFlow/JSF_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv'],[TimeObserve' XObserve'],'%10.10f',',','w+');
        clear mex_WriteMatrix;
% 
% 
%         tic
%         rng(ii)
%         [XG,TauArrG] = GillespieDirectMethod(X0, rates, stoich, solTimes, myOpts);
%         GillespieTimer = toc;
%         
% 
%         [UniqueT, UniqueIndecies] = unique(TauArrG);
%         XObserve = zeros(3,length(TimeObserve));
%         for jj=1:3
%             UniqueX = XG(jj,UniqueIndecies);
%             XObserve(jj,:) = interp1(UniqueT,UniqueX, TimeObserve,'previous');
%         end
%         mex_WriteMatrix([masterpath,'/Gillespie/GIL_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv'],[TimeObserve' XObserve'],'%10.10f',',','w+');
%         clear mex_WriteMatrix;
%         
%         TimerData = [TimerData; N0ii ii JumpSwitchFlowTimer GillespieTimer];
%         [N0ii ii]
%         
        [N0ii ii JumpSwitchFlowTimer TauTimer]
        mex_WriteMatrix([masterpath,'/META.csv'],[N0ii ii JumpSwitchFlowTimer TauTimer],'%10.10f',',','a+');
        clear mex_WriteMatrix;
    end

end





