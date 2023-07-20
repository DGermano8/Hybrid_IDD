%%
function [X,TauArr] = GeneralisedSolverSwitchingRegimes(CompartmentSystem)

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

end

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
