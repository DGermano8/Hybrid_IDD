clear all;
close all;

rng(4)

mBeta = 1.45/7; % Infect "___" people a week
mGamma = 0.4/7; % infecion for "___" weeks
mDeath = 1/(2*365); %lifespan
mBirth = mDeath;

R_0 = mBeta/(mGamma+mDeath)

N0 = 10^5;
I0 = 2;
R0 = 0;
S0 = N0-I0-R0;

dttThreshold = 10^(-5); t_iter_max = 100;
dt = 10^(-3);
tFinal = 1000;

numberCompartments = 3;
numberRates = 6;
SwitchingThreshold = 0.25;
%%

% kinetic rate parameters
k = [mBeta; mGamma;   mBirth;     mDeath;      mDeath;      mDeath];

                     
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
% initial copy numbers
X0 = [S0;I0;R0];
% propensity function
rates = @(X,k) k.*[(X(1)*X(2))/(X(1)+X(2)+X(3));
                X(2);
                X(1)+X(2)+X(3);
                X(1);
                X(2);
                X(3)];
            
% identify which compartment is in which reaction:
compartInNu = [1 1 1;
               0 1 0;
               1 0 0;
               1 0 0;
               0 1 0;
               0 0 1];
                 
% identify which reactions are discrete and which are continuous
DoDisc = [0; 1; 0];
DoCont = [1; 0; 1];
EnforceDo = [0; 0; 1];
discCompartment = [1; 1; 0; 0; 1; 0];
contCompartment = [1; 1; 1; 1; 1; 1] - discCompartment;

% DoDisc = [1; 1; 1];
% DoCont = [0; 0; 0];
% discCompartment = [1; 1; 1; 1; 1; 1];
% contCompartment = [1; 1; 1; 1; 1; 1] - discCompartment;

% DoDisc = [0; 0; 0];
% DoCont = [1; 1; 1];
% discCompartment = [0; 0; 0; 0; 0; 0];
% contCompartment = [1; 1; 1; 1; 1; 1] - discCompartment;

% initialise discrete sum compartments
sumTimes = zeros(numberRates,1);
RandTimes = rand(numberRates,1);
tauArray = zeros(numberRates,1);

TimeMesh = 0:dt:tFinal;
% initialise solution arrays
X = zeros(numberCompartments,length(TimeMesh)); X(:,1) = X0;
TauArr = zeros(numberCompartments,length(TimeMesh));
iters = 2*ones(numberCompartments,1);

% Track Absolute time
AbsT = 0; 

tic;
Xprev = X0; Xcurr = zeros(numberCompartments,1);
for ContT=TimeMesh(2:end)
    
    [DoDisc, DoCont, discCompartment, contCompartment, sumTimes, RandTimes] = IsDiscrete(Xprev,nu,rates,k,dt,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes);
    
    %%% Computes 
    % compute propensities
    for iter=1:numberCompartments
        if( (DoDisc(iter)))
            Xprev(iter) = X(iter,iters(iter)-1);
        elseif(DoCont(iter))
            Xprev(iter) = X(iter,iters(iter)-1);
        end
    end
    
    Props = rates(Xprev,k);
    dXdt = sum(Props.*(contCompartment.*nu),1)';
    for iter=1:numberCompartments
        if(DoCont(iter))
            X(iter,iters(iter)) = X(iter,iters(iter)-1) + dt*dXdt(iter);
            TauArr(iter,iters(iter)) = ContT;
        end
    end
    
    Dtau = dt;
    stayWhile = true;
    TimePassed = 0;    

    while stayWhile
        
        for iter=1:numberCompartments
            if( (DoDisc(iter)))
                Xprev(iter) = X(iter,iters(iter)-1);
                Xcurr(iter) = X(iter,iters(iter)-1);
            elseif(DoCont(iter))
                Xprev(iter) = X(iter,iters(iter)-1);
                Xcurr(iter) = X(iter,iters(iter));
            end
        end
        TrapStep = Dtau*0.5*(rates(Xprev,k) + rates(Xcurr,k));
        sumTimes = sumTimes+TrapStep;
        
        % identify which events have occured 
        IdEventsOccued = (RandTimes < (1 - exp(-sumTimes))).*discCompartment;
        if( sum(IdEventsOccued) > 0)
            tauArray = zeros(numberRates,1);
            for kk=1:length(IdEventsOccued)

                if(IdEventsOccued(kk))
                    % calculate time tau
                    ExpInt = exp(-(sumTimes(kk)-TrapStep(kk)));
                    Props = rates(Xprev,k);
                    tauArray(kk) = log((1-RandTimes(kk))/ExpInt)/(-1*Props(kk));
                end
            end
            if(sum(tauArray) > 0)
                tauArray(tauArray==0.0) = Inf;
                % Identify first event occurance time and type of event
                [Dtau1,pos] = min(tauArray);

                TimePassed = TimePassed + Dtau1;
                AbsT = AbsT + Dtau1;

                for iter=1:numberCompartments
                    if( (DoDisc(iter)))
                        X(iter,iters(iter)) = X(iter,iters(iter)-1) + nu(pos,iter);
                        Xprev(iter) = X(iter,iters(iter)-1);
                        TauArr(iter,iters(iter)) = AbsT;
                        iters(iter) = iters(iter) + 1;
                    elseif(DoCont(iter))
                        X(iter,iters(iter)) = X(iter,iters(iter)) + nu(pos,iter);
                        Xprev(iter) = X(iter,iters(iter));
                    end
                end
                RandTimes(pos) = rand;
                sumTimes = sumTimes - TrapStep;

                TrapStep = Dtau1*0.5*(rates(Xprev,k) + rates(Xprev + (Dtau1*(~DoDisc)).*dXdt,k));
                sumTimes = sumTimes+TrapStep;

                sumTimes(pos) = 0.0;

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
    
    for iter=1:numberCompartments
        if( (DoCont(iter)))
            iters(iter) = iters(iter) + 1;
        end
    end

    AbsT = ContT;
end
toc;

%%

figure;
subplot(2,2,1)
hold on;
plot(TauArr(1,:),X(1,:),'.','linewidth',1.5)
plot(TauArr(2,:),X(2,:),'.','linewidth',1.5)
plot(TauArr(3,:),X(3,:),'.','linewidth',1.5)
legend('S','I','R')
axis([0 tFinal 0 1.1*N0])

hold off;

subplot(2,2,2)
TauArr_i = TauArr(1,:);
TauArr_ii = TauArr_i(TauArr_i > 0);
h = histogram(TauArr_ii, 0:100*dt:tFinal);
axis([0 tFinal 0 1.25*max(h.BinCounts)])
ylabel('# Stochastic Events per 100dt')
xlabel('Time')

subplot(2,2,3)
TauArr_i = TauArr(2,:);
TauArr_ii = TauArr_i(TauArr_i > 0);
h = histogram(TauArr_ii, 0:100*dt:tFinal);
axis([0 tFinal 0 1.25*max(h.BinCounts)])
ylabel('# Stochastic Events per 100dt')
xlabel('Time')

subplot(2,2,4)
TauArr_i = TauArr(3,:);
TauArr_ii = TauArr_i(TauArr_i > 0);
h = histogram(TauArr_ii, 0:100*dt:tFinal);
axis([0 tFinal 0 1.25*max(h.BinCounts)])
ylabel('# Stochastic Events per 100dt')
xlabel('Time')


%%

function [DoDisc, DoCont, discCompartmentTmp, contCompartmentTmp, sumTimes,RandTimes] = IsDiscrete(X,nu,rates,k,dt,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes)

    OriginalDoDisc = DoDisc;
    OriginalDoCont = DoCont;
    for ii=1:length(X)
        if(~EnforceDo(ii))
            dX_ii = dt*sum(abs(nu(:,ii)).*rates(X,k));

            if(dX_ii >= SwitchingThreshold)
                DoCont(ii) = 1;
                DoDisc(ii) = 0;
            else
                DoCont(ii) = 0;
                DoDisc(ii) = 1;
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
