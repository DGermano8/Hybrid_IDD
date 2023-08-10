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
tFinal = 1000;

% These are solver options
dt = 10^(-3);
SwitchingThreshold = [0.5; 1000];

% kinetic rate parameters
X0 = [S0;I0;R0];

% stoichiometric matrix
nu = [-1, 1, 0;
       0,-1, 1;
       1, 0, 0;
      -1, 0, 0;
       0,-1, 0;
       0, 0,-1];
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

%%
stoich = struct();
stoich.nu = nu;
stoich.DoDisc = DoDisc;
solTimes = 0:dt:tFinal;
myOpts = struct();
myOpts.EnforceDo = EnforceDo;
myOpts.dt = dt;
myOpts.SwitchingThreshold = SwitchingThreshold;

% These are the initial conditions

I0 = 2;
R0 = 0;


N0_Vector = [2:1:5];
HowManySimsToDo = 1;

TimerData = [];
dtObserve = 10^(-2);
TimeObserve = 0:dtObserve:tFinal;
XObserve = zeros(3,length(TimeObserve));

%%
for N0ii = N0_Vector
    for ii=1:HowManySimsToDo
        N0 = round(10^(N0ii));
        S0 = N0-I0-R0;
        X0 = [S0;I0;R0];


        tic;
        rng(ii)
        [XJSF,TauArrJSG] = JumpSwitchFlowSimulator_FE(X0, rates, stoich, solTimes, myOpts);
        JumpSwitchFlowTimer = toc;
        
%         XObserve = zeros(3,length(TimeObserve));
%         jj=1;kk=0;
%         for t = TauArrJSG
%             kk=kk+1;
%             if(jj> length(TimeObserve))
%                 break;
%             else
%                 if(TimeObserve(jj) <= t)
%                     XObserve(:,jj) = XJSF(:,kk);
%                     jj=jj+1;
%                 end
%             end
%         end

        mex_WriteMatrix(['SimulationStudyResults2/JumpSwitchFlow/JFS_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv'],[TauArrJSG' XJSF'],'%10.10f',',','w+');
        clear mex_WriteMatrix;

        tic
        rng(ii)
        [XG,TauArrG] = GillespieDirectMethod(X0, rates, stoich, solTimes, myOpts);
        GillespieTimer = toc;
        
%         XObserve = zeros(3,length(TimeObserve));
%         jj=1;kk=0;
%         for t = TauArrG
%             kk=kk+1;
%             if(jj> length(TimeObserve))
%                 break;
%             else
%                 if(TimeObserve(jj) <= t)
%                     XObserve(:,jj) = XG(:,kk);
%                     jj=jj+1;
%                 end
%             end
%         end

        mex_WriteMatrix(['SimulationStudyResults2/Gillespie/GIL_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv'],[TauArrG' XG'],'%10.10f',',','w+');
        clear mex_WriteMatrix;
        
        TimerData = [TimerData; N0ii ii JumpSwitchFlowTimer GillespieTimer];
        [N0ii ii]
    end

end

mex_WriteMatrix('SimulationStudyResults2/META.csv',TimerData,'%10.10f',',','w+');
clear mex_WriteMatrix;



