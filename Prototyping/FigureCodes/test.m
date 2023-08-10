% clear all;
close all;

% randSeed = randSeed+1;
% rng(3)
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
mBeta = 1.0/7; % Infect "___" people a week
mGamma = 0.6/7; % infecion for "___" weeks
mDeath = 1/(10*365); %lifespan
mBirth = mDeath;

R_0 = mBeta/(mGamma+mDeath)

% These are the initial conditions
N0 = 10^8;
I0 = 2;
R0 = 0;
S0 = N0-I0-R0;

% How long to simulate for
tFinal = 1500;

% These are solver options
dt = 10^(-2);
SwitchingThreshold = [0.2; 200];

% kinetic rate parameters
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
EnforceDo = [0; 0; 1];
% allow I to switch, but force S and R to be continuous
% EnforceDo = [1; 0; 1];

%%
stoich = struct();
stoich.nu = nu;
stoich.DoDisc = DoDisc;
solTimes = 0:dt:tFinal;
myOpts = struct();
myOpts.EnforceDo = EnforceDo;
myOpts.dt = dt;
myOpts.SwitchingThreshold = SwitchingThreshold;

%%

%%%%%%%%%%%%%%%%% Initilise %%%%%%%%%%%%%%%%%
X0 = X0;
nu = stoich.nu;
DoDisc = stoich.DoDisc;
DoCont = ~DoDisc;

tFinal = solTimes(end);
dt = myOpts.dt;
EnforceDo = myOpts.EnforceDo;
SwitchingThreshold = myOpts.SwitchingThreshold;
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
    
    Dtau = dt;
    Xprev = X(:,iters-1);
    
    correctInteger = 0;
    % identify which compartment is to be modelled with Discrete and continuous dynamics
    if((sum(EnforceDo) ~= length(EnforceDo)))

        [NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment, ~, ~, ~] = IsDiscrete(Xprev,nu,rates,dt,AbsT,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes);
%         [NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment] = IsDiscrete(Xprev,nu,rates,dt,AbsT,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes);

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
                % slow down the FE solver so dont overstep the change
                Dtau = 0.75*dt;
                ContT = ContT - dt + Dtau;
                AbsT = AbsT  - dt + Dtau;
            else
                correctInteger = 1;
                ContT = ContT - dt + Dtau;
                AbsT = AbsT  - dt + Dtau;

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
    
    if(correctInteger)
        contCompartment = NewcontCompartment;
        discCompartment = NewdiscCompartment;
        DoCont = NewDoCont;

        for ii=1:length(Xprev)
            if(~EnforceDo(ii))
                % shouldnt need to do this, but just to be safe
                X(ii,iters) = round(X(ii,iters));
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

    
    % Dont bother doing anything discrete if its all continuous
    stayWhile = (true)*(sum(DoCont)~=length(DoCont));
    
    TimePassed = 0;
    % Perform the Stochastic Loop
    while stayWhile

        Xprev = X(:,iters-1);
        Xcurr = X(:,iters);
        
        % Integrate the cummulative wait times using trapazoid method
        TrapStep = Dtau*0.5*(rates(Xprev,AbsT-Dtau) + rates(Xcurr,AbsT));
        sumTimes = sumTimes+TrapStep;

        % identify which events have occured
        IdEventsOccued = (RandTimes < (1 - exp(-sumTimes))).*discCompartment;
        
        if( sum(IdEventsOccued) > 0)
            tauArray = zeros(nRates,1);
            for kk=1:length(IdEventsOccued)

                if(IdEventsOccued(kk))
                    % calculate time tau until event using linearisation of integral:
                    % u_k = 1-exp(- integral_{ti}^{t} f_k(s)ds )
%                     ExpInt = exp(-(sumTimes(kk)-TrapStep(kk)));
                    ExpInt = exp(-(sumTimes(kk)-TrapStep(kk)));
                    Props = rates(Xprev,AbsT-Dtau);
                    
                    tau_val_1 = log((1-RandTimes(kk))/ExpInt)/(-1*Props(kk));
                    tau_val_2 = -1;
                    tau_val =  tau_val_1;
                    % were doing a linear approximation to solve this, so
                    % it may be off, in which case we just fix it to a
                    % small order here
                    if(tau_val < 0)

                        if(abs(tau_val_1) < dt^(2))
                            tau_val_2 = abs(tau_val_1);
                        end
                        tau_val_1 = 0;
                        tau_val = max(tau_val_1,tau_val_2);

                        Dtau = 0.5*Dtau;
                        sumTimes = sumTimes - TrapStep;
                    end
                    tauArray(kk) = tau_val;
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
%                 stayWhile = false;
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

% end

%%


subplot(1,2,1)
hold on;
plot(TauArr,X(1,:),'.','linewidth',1.5,'color',[0.1 0.1 0.75])
plot(TauArr,X(2,:),'.','linewidth',1.5,'color',[0.75 0.1 0.1])
plot(TauArr,X(3,:),'.','linewidth',1.5,'color',[0.1 0.725 0.1])
legend('S','I','R')
axis([0 tFinal 0 1.1*N0])
hold off;

subplot(1,2,2)
hold on;
plot(X(1,:),X(2,:),'.-','linewidth',1.0)
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ylabel('I')
xlabel('S')

%%


function [DoDisc, DoCont, discCompartmentTmp, contCompartmentTmp, sumTimes,RandTimes, Xprev] = IsDiscrete(X,nu,rates,dt,AbsT,SwitchingThreshold,DoDisc,DoCont, EnforceDo, discCompartment, contCompartment, compartInNu, sumTimes,RandTimes)

    Xprev = X;
    OriginalDoDisc = DoDisc;
    OriginalDoCont = DoCont;
    for ii=1:length(X)
        if(~EnforceDo(ii))
%             dX_ii = dt*sum(abs(nu(:,ii)).*rates(X,AbsT));
            dX_ii = (abs(nu(:,ii)).*rates(X,AbsT));
            
            % the fast reaction
%             if(dX_ii >= SwitchingThreshold(1))
            if(sum(1./dX_ii(dX_ii>0) < dt)>0)
                DoCont(ii) = 1;
                DoDisc(ii) = 0;
            else
                if(Xprev(ii) > SwitchingThreshold(2))
                    DoCont(ii) = 1;
                    DoDisc(ii) = 0;
                else
                    DoCont(ii) = 0;
                    DoDisc(ii) = 1;
                end
            end

%             if(OriginalDoCont(ii) && DoDisc(ii))
% 
%                 % This needs a better solution \TODO
%                 if(Xprev(ii) < SwitchingThreshold(2))
%                 	Xprev(ii) = round(X(ii));
%                 end
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
%                     if(~OriginalDoDisc(ii))
%                         sumTimes(jj) = 0.0;
%                         RandTimes(jj) = rand;
%                     end
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
