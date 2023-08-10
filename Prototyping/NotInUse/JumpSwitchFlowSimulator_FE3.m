
%JUMPSWITCHFLOWSIMULATOR  Sample from CD-switching process.
%   [X,TAUARR] = JumpSwitchFlowSimulator(X0, RATES, STOICH, TIMES, OPTIONS)
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
%   - EnforceDo: a boolean indicating whether to enforce the discrete
%     dynamics to be active at all times. Default: false.
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
function [X,TauArr] = JumpSwitchFlowSimulator_FE(x0, rates, stoich, times, options)

%%%%%%%%%%%%%%%%% Initilise %%%%%%%%%%%%%%%%%

X0 = x0;
nu = stoich.nu;
[nRates,nCompartments] = size(nu);

tFinal = times(end);

% Set default ODE time step
try dt = options.dt; catch dt = 10^(-3);
end

% Set default switching thresholds
try SwitchingThreshold = options.SwitchingThreshold; catch SwitchingThreshold = [0.2, 1000];
end

% Set default compartments to discrete
try DoDisc = stoich.DoDisc; catch DoDisc = ones(nCompartments,1);
end

% Set default dynamics to all switching
try EnforceDo = options.EnforceDo; catch EnforceDo = zeros(nCompartments,1);
end
DoCont = ~DoDisc;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

NewDiscCompartmemt = zeros(nCompartments,1);
VisitedHere = zeros(nCompartments,1);
while ContT < tFinal
    ContT = ContT + dt;  
    iters = iters + 1;
    
    Dtau = dt;
    Xprev = X(:,iters-1);
    
    correctInteger = 0;
    
%     NewDiscCompartmemt = zeros(nCompartments,1);
    % identify which compartment is to be modelled with Discrete and continuous dynamics
    if((sum(EnforceDo) ~= length(EnforceDo)))
         Props = rates(Xprev,AbsT-Dtau);
        [NewDoDisc, NewDoCont, NewdiscCompartment, NewcontCompartment] = IsDiscrete(Xprev,nu,Props,dt,SwitchingThreshold,DoDisc,DoCont,discCompartment,contCompartment, EnforceDo, compartInNu);

        % move Euler mesh to ensure the new distcrete compartment is integer
        if(nnz(NewDoDisc) > nnz(DoDisc))
            

            % this ^ identifies a state has switched to discrete
            % identify which compartment is the switch
            [~,pos] = max(NewDoDisc-DoDisc);
            % compute propensities
%             Props = rates(Xprev,AbsT-Dtau); %^computed above, reuse for speed
            % Perform the Forward Euler Step
            dXdt = sum(Props.*(contCompartment.*nu),1)';

            Dtau = min(dt,abs((round(Xprev(pos)) - Xprev(pos))/dXdt(pos)));

            if(Dtau < dt)
                % used to implement the remainder of the step, and then
                % switch to discrete later
                NewDiscCompartmemt = ((NewDoDisc - DoDisc) ==1);
                correctInteger = 1;
                ContT = ContT - dt + Dtau;
                AbsT = AbsT  - dt + Dtau;
                Props = rates(Xprev,AbsT-Dtau);

            end

        else
            contCompartment = NewcontCompartment;
            discCompartment = NewdiscCompartment;
            DoCont = NewDoCont;
            DoDisc = NewDoDisc;
        end
    else
        Props = rates(Xprev,AbsT-Dtau);
    end

    
    % compute propensities
%     Props = rates(Xprev,AbsT-Dtau);  %^computed above, reuse for speed
    % Perform the Forward Euler Step
    dXdt = sum(Props.*(contCompartment.*nu),1)';
    XTmp = X(:,iters-1) + Dtau*(dXdt.*DoCont);
        
    % switch a continuous compartment to integer if needed
    OriginalDoDisc = DoDisc;
    OriginalDoCont = DoCont;
    if(correctInteger)
        contCompartment = NewcontCompartment;
        discCompartment = NewdiscCompartment;
        DoCont = NewDoCont;
        DoDisc = NewDoDisc;
    end

    % Dont bother doing anything discrete if its all continuous
    stayWhile = (true)*(sum(DoCont)~=length(DoCont));
    
    TimePassed = 0;
    % Perform the Stochastic Loop
    Xcurr = XTmp;
    while stayWhile
        
        % Integrate the cummulative wait times using trapazoid method
        Props = rates(Xprev,AbsT-Dtau);
        TrapStep = Dtau*0.5*(Props + rates(Xcurr,AbsT));
        sumTimes = sumTimes+TrapStep;
        
        % if a compartment has /just/ become discrete, make sure it has
        % zero sumTimes and rest the randTimes
        if(sum(NewDiscCompartmemt) == 1)
            for ii=1:length(Xprev)
                if(NewDiscCompartmemt(ii)==1 &&EnforceDo(ii)==0)
                    for jj = 1:size(compartInNu,1)
                        if(compartInNu(jj,ii)==1)
                            discCompartment(jj) = 1;
                            sumTimes(jj) = 0.0;
                            RandTimes(jj) = rand;
                        end
                    end
                end
            end
        end

        % identify which events have occured
        IdEventsOccued = (RandTimes < (1 - exp(-sumTimes))).*discCompartment;
        
        if( sum(IdEventsOccued) > 0)
            tauArray = zeros(nRates,1);
            for kk=1:length(IdEventsOccued)

                if(IdEventsOccued(kk))
                    % calculate time tau until event using linearisation of integral:
                    % u_k = 1-exp(- integral_{ti}^{t} f_k(s)ds )
                    ExpInt = exp(-(sumTimes(kk)-TrapStep(kk)));
                    
%                     Props = rates(Xprev,AbsT-Dtau); %^computed above, reuse for speed
                    % This method explicitly assumes that the rates are
                    % varying slowly in time (more specifically, time 
                    % independant)

                    Integral = -1*log((1-RandTimes(kk))/ExpInt);
                    tau_val_1 = Integral/((Props(kk)));

                    % Try Newtons Method to find the time to more accuracy
                    Error = 1; %This ensures we do atleast one iteration 
                    howManyHere = 1;
                    while (abs(Error) > 10^(-10) && howManyHere<10)
                        howManyHere=howManyHere+1;
                        Props2 = rates(Xprev + (tau_val_1*(OriginalDoCont)).*dXdt,AbsT+tau_val_1);
                        Error = 0.5*tau_val_1*(Props2(kk)+Props(kk))-Integral;
                        tau_val_1 = tau_val_1 - 1./(Props2(kk))*(Error);

                    end
                    
                    tau_val_2 = -1;
                    tau_val =  tau_val_1;
                    % were doing a linear approximation to solve this, so
                    % it may be off, in which case we just fix it to a
                    % small order here
%                     if(tau_val < 0)
% 
%                         if(abs(tau_val_1) < dt^(2))
%                             tau_val_2 = abs(tau_val_1);
%                         end
%                         tau_val_1 = 0;
%                         tau_val = max(tau_val_1,tau_val_2);
% 
%                         Dtau = 0.5*Dtau;
%                         sumTimes = sumTimes - TrapStep;
%                     end
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
                
                X(:,iters) = X(:,iters-1) + nu(pos,:)';
                Xprev = X(:,iters-1);
                Xcurr = X(:,iters);
                TauArr(iters) = AbsT;
                iters = iters + 1;
                
                Props = rates(Xprev + (Dtau1*(OriginalDoCont)).*dXdt,AbsT);

                % Bring compartments up to date
                sumTimes = sumTimes - TrapStep;
                TrapStep = Dtau1*0.5*(rates(Xprev,AbsT-Dtau1) + Props);
                sumTimes = sumTimes+TrapStep;
                
                % reset timers and sums
                RandTimes(pos) = rand;
                sumTimes(pos) = 0.0;
                
                % Check if a compartment has become continuous. If so, update the system to this point and
                % move the FE mesh to this point and 
                [NewDoDisc, NewDoCont, ~, ~] = IsDiscrete(Xprev,nu,Props,dt,SwitchingThreshold,DoDisc,DoCont,discCompartment,contCompartment, EnforceDo, compartInNu);
                if(nnz(NewDoCont) > nnz(DoCont))
                    ContT = ContT - (Dtau-Dtau1);
                    stayWhile = false;
                else
                    % execute remainder of Euler Step
                    Dtau = Dtau-Dtau1;
                end
                
                
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
    X(:,iters) = Xcurr;
    TauArr(iters) = ContT;
    if(sum(NewDiscCompartmemt)==1)
        [~,pos] = max(NewDiscCompartmemt);
        X(pos,iters) = round(X(pos,iters));
        
        for jj = 1:size(compartInNu,1)
            if(NewDiscCompartmemt(pos) && compartInNu(jj,pos))
                discCompartment(jj) = 1;
                sumTimes(jj) = 0.0;
                RandTimes(jj) = rand;
            end
        end
        NewDiscCompartmemt(pos) = 0;
    end

    
    AbsT = ContT;
end

if(iters < overFlowAllocation)
    X(:,(iters+1:end)) = [];
    TauArr((iters+1:end)) = [];
end

end

%%
function [DoDisc, DoCont, discCompartmentTmp, contCompartmentTmp] = IsDiscrete(Xprev,nu,Props,dt,SwitchingThreshold,DoDisc,DoCont,discCompartment,contCompartment, EnforceDo, compartInNu)

    OriginalDoDisc = DoDisc;
    OriginalDoCont = DoCont;
    
%     dX_ii = dt*sum(abs(nu).*Props,1);
%     for ii=1:length(Xprev)
%         if(~EnforceDo(ii))
%             if(dX_ii(ii) >= SwitchingThreshold(1))
%                 DoCont(ii) = 1;
%                 DoDisc(ii) = 0;
%             else
%                 if(Xprev(ii) > SwitchingThreshold(2))
%                     DoCont(ii) = 1;
%                     DoDisc(ii) = 0;
%                 else
%                     DoCont(ii) = 0;
%                     DoDisc(ii) = 1;
%                 end
%             end
%         end
%     end
%     dX_ii = dt*sum(abs(nu).*Props,1);
%     condition1 = (Xprev  < SwitchingThreshold(2));
    DoDisc = (Xprev  < SwitchingThreshold(2)).*(EnforceDo==0);
    DoCont = (DoDisc==0);
    
    if(OriginalDoDisc == DoDisc)
        discCompartmentTmp = discCompartment;
        contCompartmentTmp = contCompartment;
    else
        discCompartmentTmp = zeros(size(compartInNu,1),1);
        contCompartmentTmp = ones(size(compartInNu,1),1);

        for ii=1:length(Xprev)
            for jj = 1:size(compartInNu,1)
                if(~EnforceDo(ii))
                    if(DoDisc(ii) && compartInNu(jj,ii))
                        discCompartmentTmp(jj) = 1;
                    end
                else
                    if(OriginalDoDisc(ii) && compartInNu(jj,ii))
                        discCompartmentTmp(jj) = 1;
                    end
                end
            end
        end

        contCompartmentTmp = contCompartmentTmp - discCompartmentTmp;
    end

end

%%