clear all;
close all;

% rng(3)

% These define the rates of the system
mAlpha = 2;
mBeta = 1;
mGamma = 20;
numberRates = 3;

% These are the initial conditions
A0 = 50;
B0 = 50;
X0 = [A0;
      B0];
numberCompartments = 2;

% How long to simulate for
tFinal = 5;

% These are solver options
dttThreshold = 10^(-5); t_iter_max = 100;
dt = 10^(-4);
SwitchingThreshold = 10;
%%

% kinetic rate parameters
k = [mAlpha; mBeta; mGamma];

                     
% reactant stoichiometries
nuMinus = [0,0;
           1,0;
           0,1];
% product stoichiometries
nuPlus = [1,0;
          0,1;
          0,0];
% stoichiometric matrix
nu = nuPlus - nuMinus;

% propensity function
% Rates :: X -> rates -> propensities
rates = @(X,k) k.*[X(1);
                X(1)*X(2);
                X(2)];
            
% identify which compartment is in which reaction:
compartInNu = nu~=0;
                 
% identify which reactions are discrete and which are continuous
DoDisc = [1; 0];
DoCont = [0; 1];
EnforceDo = [1; 1];



%%

discCompartment = [1; 1; 0;];
contCompartment = ~discCompartment;

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
    
    [DoDisc, DoCont, discCompartment, contCompartment] = IsDiscrete(Xprev,nu,rates,k,dt,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu);
    
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
                    DeltaTrap = log(1/(1-RandTimes(kk)))-(sumTimes(kk)-TrapStep(kk));

                    % this uses symbolic matlab - super slow!!! Think of
                    % something better!
%                     PropsT = rates(Xprev,k) + rates(Xprev + (t*(~DoDisc)).*dXdt,k);
%                     Rnumeric = vpa(solve(0.5*t*PropsT(kk)- DeltaTrap,t));
%                     Rnumeric = double(solve(0.5*t*PropsT(kk)- DeltaTrap,t));
%                     try
%                         tauArray(kk) = Rnumeric(Rnumeric > 0);
%                     end
                    % Numerically find the time!!!!
                    dtt = dttThreshold; t_ii = 0;
                    FindingT = true;
                    while FindingT
                        t_ii = t_ii + dtt;
                        PropsT = rates(Xprev,k) + rates(Xprev + (t_ii*(~DoDisc)).*dXdt,k);
                        Sigma_ii = 0.5*t_ii*PropsT(kk)- DeltaTrap;
                        if Sigma_ii > 0
                            tauArray(kk) = t_ii;
                            FindingT = false;
                        end
                        if t_ii > dt
                            FindingT = false;
                        end
                    end
                
%%%%%%
%                     t_iter = 1;
%                     Sigma_ii = 1;
%                     t_ii = 0.5*dt;
%                     while (abs(Sigma_ii) > dttThreshold)
%                         PropsT = rates(Xprev,k) + rates(Xprev + (t_ii*(~DoDisc)).*dXdt,k);
%                         Sigma_ii = 0.5*t_ii*PropsT(kk)- DeltaTrap;
%                         if(Sigma_ii > 0)
%                             t_ii = 0.5*t_ii;
%                         elseif(Sigma_ii < 0)
%                             t_ii = 1.5*t_ii;
%                         end
%                         
%                         if t_iter > t_iter_max
%                             break;
%                         end
%                         t_iter = t_iter + 1;
%                     end
%                     tauArray(kk) = t_ii;
                    
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
hold on;
plot(TauArr(1,:),sqrt(X(1,:)),'.','linewidth',1.5)
plot(TauArr(2,:),sqrt(X(2,:)),'.','linewidth',1.5)
legend('A','B')

axis([0 tFinal 0 1.1*sqrt(max(max(X)))])

%%

function [DoDisc, DoCont, discCompartmentTmp, contCompartmentTmp] = IsDiscrete(X,nu,rates,k,dt,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu);

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
