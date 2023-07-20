clear all;
close all;

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
mB = 0.1;          % 10 bites a day 
mBeta = 0.1;      % Successful transfer of parasite
mGamma = 1/7;     % 1 week to recover
mOmega = 1/(365); % 1 yeah of immunity
mG = 0.1;
mDeath = 1.0; 
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
tFinal = 2;

% These are solver options
dt = 10^(-6);
SwitchingThreshold = [0.2; 20];

% kinetic rate parameters
%              [1   2      3       4       5   6       7     ]
kConsts =      [mB; mBeta; mGamma; mOmega; mG; mBirth; mDeath];
%              [1          2     3     4     5     6     7     8     9]
kTime = @(p,t) [p(1)*p(2); p(3); p(4); p(5); p(1); p(6); p(7); p(7); p(7)];

                     
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


%%

%%%%%%%%%%%%%%%%% Initilise %%%%%%%%%%%%%%%%%
X0 = CompartmentSystem.X0;
tFinal = CompartmentSystem.tFinal;
kConsts = CompartmentSystem.kConsts;
kTime = CompartmentSystem.kTime;
rates = CompartmentSystem.rates;
nu = CompartmentSystem.nu;
DoDisc = CompartmentSystem.DoDisc;
EnforceDo = CompartmentSystem.EnforceDo;
dt = CompartmentSystem.dt;
SwitchingThreshold = CompartmentSystem.SwitchingThreshold;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DoCont = ~DoDisc;

numberRates = length(nu);
numberCompartments = length(X0);

% identify which compartment is in which reaction:
compartInNu = nu~=0;
discCompartment = compartInNu*(DoDisc);
contCompartment = ~discCompartment;

% initialise discrete sum compartments
sumTimes = zeros(numberRates,1);
RandTimes = rand(numberRates,1);
tauArray = zeros(numberRates,1);

TimeMesh = 0:dt:tFinal;
overFlowAllocation = round(2.5*length(TimeMesh));

% initialise solution arrays
X = zeros(numberCompartments,overFlowAllocation); X(:,1) = X0;
TauArr = zeros(1,overFlowAllocation);
iters = 1;

% Track Absolute time
AbsT = dt; 

Xprev = X0; Xcurr = zeros(numberCompartments,1);
for ContT=TimeMesh(2:end)
    iters = iters + 1;

    Xprev = X(:,iters-1);
    % identify which compartment is to be modelled with Discrete and continuous dynamics
    [DoDisc, DoCont, discCompartment, contCompartment, sumTimes, RandTimes, XIsDiscrete] = IsDiscrete(Xprev,nu,rates,kTime,kConsts,dt,AbsT,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes);
    
    if( sum(Xprev == XIsDiscrete)<length(Xprev))
        X(:,iters) = XIsDiscrete;
        TauArr(iters) = ContT;
        iters = iters + 1;
    end

    % compute propensities
    Props = rates(Xprev,kTime(kConsts,AbsT-dt));

    % Perform the Forward Euler Step
    dXdt = sum(Props.*(contCompartment.*nu),1)';
    X(:,iters) = X(:,iters-1) + dt*dXdt.*DoCont;
    TauArr(iters) = ContT;

    Dtau = dt;
    stayWhile = true;
    TimePassed = 0;    
    % Perform the Stochastic Loop
    while stayWhile

        Xprev = X(:,iters-1);
        Xcurr = X(:,iters);

        % Integrate the cummulative wait times using trapazoid method
        TrapStep = Dtau*0.5*(rates(Xprev,kTime(kConsts,AbsT-Dtau)) + rates(Xcurr,kTime(kConsts,AbsT)));
        sumTimes = sumTimes+TrapStep;

        % identify which events have occured 
        IdEventsOccued = (RandTimes < (1 - exp(-sumTimes))).*discCompartment;
        if( sum(IdEventsOccued) > 0)
            tauArray = zeros(numberRates,1);
            for kk=1:length(IdEventsOccued)

                if(IdEventsOccued(kk))
                    % calculate time tau until event using linearisation of integral:
                    % u_k = 1-exp(- integral_{ti}^{t} f_k(s)ds )
                    ExpInt = exp(-(sumTimes(kk)-TrapStep(kk)));
                    Props = rates(Xprev,kTime(kConsts,AbsT-Dtau));
                    tauArray(kk) = log((1-RandTimes(kk))/ExpInt)/(1*Props(kk));
                end
            end
            % identify which reaction occurs first
            if(sum(tauArray) > 0)
                tauArray(tauArray==0.0) = Inf;
                % Identify first event occurance time and type of event
                [Dtau1,pos] = min(tauArray);

                TimePassed = TimePassed + Dtau1;
                AbsT = AbsT + Dtau1;

                % implement first reaction
                iters = iters + 1;
                X(:,iters) = X(:,iters-1) + nu(pos,:)';
                Xprev = X(:,iters-1);
                TauArr(iters) = AbsT;

                % Bring compartments up to date
                sumTimes = sumTimes - TrapStep;
                TrapStep = Dtau1*0.5*(rates(Xprev,kTime(kConsts,AbsT-Dtau1)) + rates(Xprev + (Dtau1*(~DoDisc)).*dXdt,kTime(kConsts,AbsT)));
                sumTimes = sumTimes+TrapStep;

                % reset timers and sums
                RandTimes(pos) = rand;
                sumTimes(pos) = 0.0;

                % execute remainder of Euler Step
                Dtau = Dtau-Dtau1;

            else
                stayWhile = false;
            end
        else
            stayWhile = false;
        end

        if((AbsT > ContT) || (TimePassed >= dt))
            stayWhile = false;
        end


    end

    AbsT = ContT;
end

if(iters < overFlowAllocation)
    X(:,(iters+1:end)) = [];
    TauArr((iters+1:end)) = [];
end



%%

figure;
subplot(1,2,1)
hold on;
plot(TauArr,X(1,:),'.','linewidth',1.5)
plot(TauArr,X(2,:),'.','linewidth',1.5)
plot(TauArr,X(3,:),'.','linewidth',1.5)
legend('S_H','I_H','R_H')
axis([0 tFinal 0 1.1*max([max(X(1,:)) max(X(2,:)) max(X(3,:))])])
hold off;

subplot(1,2,2)
hold on;
plot(TauArr,X(4,:),'.','linewidth',1.5)
plot(TauArr,X(5,:),'.','linewidth',1.5)
plot(TauArr,X(6,:),'.','linewidth',1.5)
legend('S_M','E_M','I_M')
axis([0 tFinal 0 1.1*max([max(X(4,:)) max(X(5,:)) max(X(6,:))])])
hold off


%%


function [DoDisc, DoCont, discCompartmentTmp, contCompartmentTmp, sumTimes,RandTimes, Xprev] = IsDiscrete(X,nu,rates,kTime,kConsts,dt,AbsT,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes)
    
    Xprev = X;
    OriginalDoDisc = DoDisc;
    OriginalDoCont = DoCont;
    for ii=1:length(X)
        if(~EnforceDo(ii))
            dX_ii = dt*sum(abs(nu(:,ii)).*rates(X,kTime(kConsts,AbsT)));

            if(dX_ii >= SwitchingThreshold(1))
                DoCont(ii) = 1;
                DoDisc(ii) = 0;
            else
                DoCont(ii) = 0;
                DoDisc(ii) = 1;
            end
            
            if(OriginalDoCont(ii) && DoDisc(ii))
                % This needs a better solution \TODO
                if(Xprev(ii) < SwitchingThreshold(2))
                	Xprev(ii) = round(X(ii));
                end
            end
        end
        
    end
    discCompartmentTmp = zeros(size(compartInNu,1),1);
    contCompartmentTmp = ones(size(compartInNu,1),1);
    for ii=1:length(X)
        if(~EnforceDo(ii))
            for jj = 1:size(compartInNu,1)
                if(DoDisc(ii) && compartInNu(jj,ii))
                    discCompartmentTmp(jj) = 1;
                    if(~OriginalDoDisc(ii))
                        sumTimes(jj) = 0.0;
                        RandTimes(jj) = rand;
                    end
                end
            end
        end
    end
    for ii=1:length(X)
        if(EnforceDo(ii))
            for jj = 1:size(compartInNu,1)
                if(OriginalDoDisc(ii) && compartInNu(jj,ii))
                    discCompartmentTmp(jj) = 1;
                end
            end
        end
    end
    contCompartmentTmp = contCompartmentTmp - discCompartmentTmp;

end
