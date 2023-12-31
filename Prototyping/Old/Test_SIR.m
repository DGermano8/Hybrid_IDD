% clear all;
close all;

% randSeed = 20;
rng(randSeed)

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
mBeta = 1.45/7; % Infect "___" people a week
mGamma = 0.4/7; % infecion for "___" weeks
mDeath = 1/(2*365); %lifespan
mBirth = mDeath;

R_0 = mBeta/(mGamma+mDeath);

% These are the initial conditions
N0 = 10^5;
I0 = 2;
R0 = 0;
S0 = N0-I0-R0;

% How long to simulate for
tFinal = 1000;

% These are solver options
dt = 10^(-3);
SwitchingThreshold = 0.2;

% kinetic rate parameters
k = [mBeta; mGamma;   mBirth;     mDeath;      mDeath;      mDeath];
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
% Rates :: X -> rates -> propensities
rates = @(X,k,t) k.*[(X(1)*X(2))/(X(1)+X(2)+X(3));
                X(2);
                X(1)+X(2)+X(3);
                X(1);
                X(2);
                X(3)];

% identify which reactions are discrete and which are continuous
DoDisc = [0; 1; 0];
% allow S and I to switch, but force R to be continuous
EnforceDo = [0; 0; 1];
% allow I to switch, but force S and R to be continuous
% EnforceDo = [1; 0; 1];

%%
CompartmentSystem  = struct();

CompartmentSystem.X0 =X0;
CompartmentSystem.tFinal = tFinal;
CompartmentSystem.k = k;
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
k = CompartmentSystem.k;
rates = CompartmentSystem.rates;
nu = CompartmentSystem.nu;
DoDisc = CompartmentSystem.DoDisc;
EnforceDo = CompartmentSystem.EnforceDo;
dt = CompartmentSystem.dt;
SwitchingThreshold = CompartmentSystem.SwitchingThreshold;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DoCont = ~DoDisc;

numberRates = length(k);
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
StochTime = [];
Xprev = X0; Xcurr = zeros(numberCompartments,1);
for ContT=TimeMesh(2:end)
    iters = iters + 1;

    Xprev = X(:,iters-1);
    % identify which compartment is to be modelled with Discrete and continuous dynamics
    [DoDisc, DoCont, discCompartment, contCompartment, sumTimes, RandTimes, XIsDiscrete] = IsDiscrete(Xprev,nu,rates,k,dt,AbsT,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes);
                                                                                                          
    if( sum(Xprev == XIsDiscrete)<length(Xprev))
        X(:,iters) = XIsDiscrete;
        TauArr(iters) = ContT;
        iters = iters + 1;
    end
    
    % compute propensities
    Props = rates(Xprev,k,AbsT-dt);

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
        TrapStep = Dtau*0.5*(rates(Xprev,k,AbsT-Dtau) + rates(Xcurr,k,AbsT));
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
                    Props = rates(Xprev,k,AbsT-Dtau);
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
                StochTime =  [StochTime AbsT];

                % Bring compartments up to date
                sumTimes = sumTimes - TrapStep;
                TrapStep = Dtau1*0.5*(rates(Xprev,k,AbsT-Dtau1) + rates(Xprev + (Dtau1*(~DoDisc)).*dXdt,k,AbsT));
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
subplot(2,2,1)
hold on;
plot(TauArr,X(1,:),'.','linewidth',1.5)
plot(TauArr,X(2,:),'.','linewidth',1.5)
plot(TauArr,X(3,:),'.','linewidth',1.5)
legend('S','I','R')
axis([0 tFinal 0 1.1*N0])

hold off;

subplot(2,2,2)
plot(X(1,:),X(2,:),'.','linewidth',1.5)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel('I')
xlabel('S')

subplot(2,2,3)
histogram(StochTime,0:100*dt:tFinal);
%%


function [DoDisc, DoCont, discCompartmentTmp, contCompartmentTmp, sumTimes,RandTimes, Xprev] = IsDiscrete(X,nu,rates,k,dt,AbsT,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes)
    
    Xprev = X;
    OriginalDoDisc = DoDisc;
    OriginalDoCont = DoCont;
    for ii=1:length(X)
        if(~EnforceDo(ii))
            dX_ii = dt*sum(abs(nu(:,ii)).*rates(X,k,AbsT));

            if(dX_ii >= SwitchingThreshold)
                DoCont(ii) = 1;
                DoDisc(ii) = 0;
            else
                DoCont(ii) = 0;
                DoDisc(ii) = 1;
            end
            
            if(OriginalDoCont(ii) && DoDisc(ii))
                % This needs a better solution \TOFIX
                if(Xprev(ii) < 50)
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
