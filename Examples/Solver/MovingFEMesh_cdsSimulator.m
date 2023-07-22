%CDSSIMULATOR  Sample from CD-switching process.
%   [X,TAUARR] = cdsSimulator(X0, RATES, STOICH, TIMES, OPTIONS)
%   simulates from the continuous-discrete-switching process with flow
%   RATES and stoichiometry STOICH, starting from initial condition
%   X0, returning the process states at the times in TIMES. The output
%   X is a matrix with as many columns as there are species, and as
%   many rows as there are time points in the output. The output
%   TAUARR is a vector of time points at which the output is sampled.
%
%   OPTIONS is a structure with the following fields:
%   - dt: the time step used for the Euler-Maruyama discretisation of
%     the continuous dynamics. Default: 0.01.
%   - EnforceDo: a boolean indicating whether to enforce the discrete or
%     cont inuous dynamics to be active at all times. Default: false.
%   - SwitchingThreshold: a threshold for switching between discrete
%     and continuous dynamics. Default: 0.1.
%
% TODO Currently the times are only not actually used beyond using
% TIMES(END) as the final time point.
%
% TODO Currently the documentation is confused about the name of the
% function provided and the default values for options are not
% respected.
%
% Author: Domenic P.J. Germano (2023).
function [X,TauArr] = MovingFEMesh_cdsSimulator(x0, rates, stoich, times, options)

%%%%%%%%%%%%%%%%% Initilise %%%%%%%%%%%%%%%%%
X0 = x0;
nu = stoich.nu;
DoDisc = stoich.DoDisc;
DoCont = ~DoDisc;

tFinal = times(end);
dt = options.dt;
EnforceDo = options.EnforceDo;
SwitchingThreshold = options.SwitchingThreshold;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nRates,nCompartments] = size(nu);

% identify which compartment is in which reaction:
compartInNu = nu~=0;
discCompartment = compartInNu*(DoDisc);
contCompartment = ~discCompartment;

% initialise discrete sum compartments
sumTimes = zeros(nRates,1);
RandTimes = rand(nRates,1);
tauArray = zeros(nRates,1);

TimeMesh = 0:dt:tFinal;
overFlowAllocation = round(2.5*length(TimeMesh));

% initialise solution arrays
X = zeros(nCompartments,overFlowAllocation);
X(:,1) = X0;
TauArr = zeros(1,overFlowAllocation);
iters = 1;

% Track Absolute time
AbsT = dt;
ContT = 0;
Xprev = X0;
Xcurr = zeros(nCompartments,1);
while ContT < tFinal
    ContT = ContT + dt;
    iters = iters + 1;

    Xprev = X(:,iters-1);
    Dtau = dt;

    % identify which compartment is to be modelled with Discrete and continuous dynamics
    if((sum(EnforceDo) ~= length(EnforceDo)))

        [NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment] = IsDiscrete(Xprev,nu,rates,dt,AbsT,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes);

        % move Euler mesh to ensure the new distcrete compartment is integer
        if(nnz(NewDoDisc) > nnz(DoDisc))
            % this ^ identifies a state has switched to discrete
            % identify which compartment is the switch
            [~,pos] = max(NewDoDisc-DoDisc);

            % compute propensities
            Props = rates(Xprev,AbsT-Dtau);
            % Perform the Forward Euler Step
            dXdt = sum(Props.*(contCompartment.*nu),1)';

            Dtau = abs((round(Xprev(pos)) - Xprev(pos))/dXdt(pos));

            if(dt < Dtau)
                % your threshold is off, stay continuous
                Dtau = dt;
            else
                ContT = ContT - dt + Dtau;
                AbsT = AbsT  - dt + Dtau;

                contCompartment = NewcontCompartment;
                discCompartment = NewdiscCompartment;
                DoCont = NewDoCont;

                for ii=1:length(Xprev)
                    if(~EnforceDo(ii))
                        for jj = 1:size(compartInNu,1)
                            if(NewDoDisc(ii) && compartInNu(jj,ii))
                                discCompartment(jj) = 1;
                                if(~DoDisc(ii))
                                    sumTimes(jj) = 0.0;
                                    RandTimes(jj) = rand;
                                end
                            end
                        end
                    end
                end
                DoDisc = NewDoDisc;

            end

        else
            contCompartment = NewcontCompartment;
            discCompartment = NewdiscCompartment;
            DoCont = NewDoCont;
            DoDisc = NewDoDisc;
        end    
    end

    % compute propensities
    Props = rates(Xprev,AbsT-Dtau);

    % Perform the Forward Euler Step
    dXdt = sum(Props.*(contCompartment.*nu),1)';
    X(:,iters) = X(:,iters-1) + Dtau*dXdt.*DoCont;
    
    TauArr(iters) = ContT;

    % Dont bother doing anything discrete if its all continuous
    stayWhile = (true)*(sum(DoCont)~=length(DoCont));
    TimePassed = 0;
    % Perform the Stochastic Loop
    while stayWhile

        Xprev = X(:,iters-1);
        Xcurr = X(:,iters);

        % compute propensities
        Props = rates(Xprev,AbsT-dt);

        % Integrate the cummulative wait times using trapazoid method
        TrapStep = Dtau*0.5*(Props + rates(Xcurr,AbsT));
        sumTimes = sumTimes+TrapStep;

        % identify which events have occured
        IdEventsOccued = (RandTimes < (1 - exp(-sumTimes))).*discCompartment;
        if( sum(IdEventsOccued) > 0)
            tauArray = zeros(nRates,1);
            for kk=1:length(IdEventsOccued)

                if(IdEventsOccued(kk))
                    % calculate time tau until event using linearisation of integral:
                    % u_k = 1-exp(- integral_{ti}^{t} f_k(s)ds )
                    ExpInt = exp(-(sumTimes(kk)-TrapStep(kk)));
%                     Props = rates(Xprev,AbsT-Dtau);
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
                TrapStep = Dtau1*0.5*(rates(Xprev,AbsT-Dtau1) + rates(Xprev + (Dtau1*(~DoDisc)).*dXdt,AbsT));
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


function [DoDisc, DoCont, discCompartmentTmp, contCompartmentTmp] = IsDiscrete(X,nu,rates,dt,AbsT,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes)

    Xprev = X;
    OriginalDoDisc = DoDisc;
    OriginalDoCont = DoCont;
    trigger = 0;
    for ii=1:length(X)
        if(~EnforceDo(ii))
            dX_ii = dt*sum(abs(nu(:,ii)).*rates(X,AbsT));

            if(dX_ii >= SwitchingThreshold(1))
                DoCont(ii) = 1;
                DoDisc(ii) = 0;
            elseif(Xprev(ii) < SwitchingThreshold(2))
                DoCont(ii) = 0;
                DoDisc(ii) = 1;
                Xprev(ii) = round(X(ii));
                trigger = 1;
            else
                DoCont(ii) = 0;
                DoDisc(ii) = 1;
            end

%             if(OriginalDoCont(ii) && DoDisc(ii))
%                 % This needs a better solution \TODO
%                 if(Xprev(ii) < SwitchingThreshold(2))
%                 	Xprev(ii) = round(X(ii));
%                 end
%             elseif(DoDisc(ii) && OriginalDoDisc(ii) && ~isinteger(Xprev(ii)))
%                 Xprev(ii) = round(X(ii));
%             end
            
        end

    end
    discCompartmentTmp = zeros(size(compartInNu,1),1);
    contCompartmentTmp = ones(size(compartInNu,1),1);
    for ii=1:length(X)
        if(~EnforceDo(ii))
            for jj = 1:size(compartInNu,1)
                if(DoDisc(ii) && compartInNu(jj,ii))
                    discCompartmentTmp(jj) = 1;
                    if(trigger)
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
