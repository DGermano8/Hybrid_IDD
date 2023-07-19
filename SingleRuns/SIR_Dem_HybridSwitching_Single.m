
close all;
clear all;
tic;

dt = 10^(-4);
threshold = 0.01;

rng(10);

mBeta = 1.5/7; % Infect "___" people a week
mGamma = 0.6/7; % infecion for "___" weeks
mDeath = 1/(2*365); %lifespan
mBirth = mDeath;

Params = struct();
Params.mBeta = mBeta;
Params.mGamma = mGamma;
Params.mDeath = mDeath;
Params.mBirth = mBirth;

R_0 = mBeta/(mGamma+mDeath)

% N0 = 10^5;
% I0 = N0*mBirth*(mBeta - mGamma - mBirth)/(mBeat*(mGamma + mBirth));
% R0 = 1000;
% S0 = N0-I0-R0;

N0 = 10^7;
I0 = N0*mBirth*(mBeta - mGamma - mBirth)/(mBeta*(mGamma + mBirth));

S0 = N0*(mGamma + mBirth)/mBeta + 5*10^6;
R0 = N0 - I0 - S0;
t_final = 1000;
TimeMesh = 0:dt:t_final;

% Initialising stochasitic info
sumILoss = 0; sumIGain = 0; sumIDeath = 0; 
uILoss = rand; uIGain = rand; uIDeath = rand;

% Initialising stochasitic compartment makes for shorter run time
I = zeros(length(TimeMesh),1); I(1) = I0;
tau = zeros(length(TimeMesh),1); tau(1) = 0.0;
tau_disc = zeros(length(TimeMesh),1); tau(1) = 0.0;

TauArray = zeros(1,3);
i_I = 1; LastIEvent = 1;

% Initialising deterministic compartment compartment
R = zeros(length(TimeMesh),1); R(1) = R0;
S = zeros(length(TimeMesh),1); S(1) = S0;
% N = zeros(length(0:dt:t_final),1); N(1) = N0;
Ni = N0;
ii = 1; 

% Track Absolute time
AbsT = 0; 

di = zeros(length(TimeMesh),1);

for ContT=TimeMesh(2:end)
    ii=ii+1;
    [doDisc ,dI] = IsDiscrete(S(ii),I(i_I),R(ii),Params,dt,threshold);
    di(ii) = dI;
    if(doDisc == 1)
    
        % Update both S and R using Forward Euler
        dSdt = mBirth*Ni - mDeath*S(ii-1);
        S(ii) = S(ii-1) + dt*dSdt;
        dRdt =  - mDeath*R(ii-1);
        R(ii) = R(ii-1) + dt*dRdt;

        % Calculate N
        Ni = S(ii) + R(ii) + I(i_I);

        Si=S(ii-1); Ri=R(ii-1); %Ni=N(ii-1);

        Dtau = dt;
        stayWhile = true;
        TimePassed = 0;
        while stayWhile

            % Caclculate putative times since last event using Trapazoid Rule
            % Trapazoid Method for ILoss
            LossTrap = Dtau*(mGamma*I(i_I));
            sumILoss = sumILoss + LossTrap;

            % Trapazoid Method for IGain 
            GainTrap = Dtau*0.5*( mBeta*(Si*I(i_I)/Ni) + mBeta*(S(ii)*I(i_I))/Ni );
            sumIGain = sumIGain + GainTrap;

            % Trapazoid Method for IDeath
            DeathTrap = Dtau*(mDeath*I(i_I));
            sumIDeath = sumIDeath + DeathTrap;

            % Check if an event occurs within current Forward Euler Step, 
            % and find time of occurance via linearising solution over current step
            DTauGain = 0.0; DTauLoss = 0.0; DTauDeath=0.0;
            if(uIGain < (1 - exp(-sumIGain)) )
                DTauGain = sqrt((Si/dSdt)^2+2.0*(log(1/(1-uIGain))-(sumIGain-GainTrap))/(mBeta*I(i_I)*dSdt/Ni))-(Si/dSdt);
            end
            if(uILoss < (1 - exp(-sumILoss)) )
                DTauLoss = (log(1/(1-uILoss))-(sumILoss-LossTrap))/(mGamma*I(i_I));            
            end
            if(uIDeath < (1 - exp(-sumIDeath)) )
                DTauDeath = (log(1/(1-uIDeath))-(sumIDeath-DeathTrap))/(mDeath*I(i_I));
            end
            TauArray(1) = DTauGain; TauArray(2) = DTauLoss; TauArray(3) = DTauDeath;

            % Check if any event occured within current step
            if(sum(TauArray) > 0)
                TauArray(TauArray==0.0) = Inf;

                % Identify first event occurance time and type of event
                [Dtau1,pos] = min(TauArray);

                TimePassed = TimePassed + Dtau1;

                AbsT = AbsT + Dtau1;
                tau(i_I+1) = AbsT;

                % Implement first event
                if(pos == 1)
    %                 I = [I (I(i_I)+1)];
                    I(i_I+1) = (I(i_I)+1);
                    i_I=i_I+1;

                    Si = S(ii);
                    S(ii) = S(ii)-1;

                    uIGain = rand;
                    sumIGain = 0;

                    % Bring other compartments up to time tau
                    sumILoss = sumILoss - LossTrap;
                    % Trapazoid Method for ILoss
                    LossTrap = Dtau1*(mGamma*I(i_I-1));
                    sumILoss = sumILoss + LossTrap;

                    sumIDeath = sumIDeath - DeathTrap;
                    % Trapazoid Method for IDeath
                    DeathTrap = Dtau1*(mDeath*I(i_I-1));
                    sumIDeath = sumIDeath + DeathTrap;

                    Dtau = Dtau-Dtau1;

                elseif(pos == 2)
                    I(i_I+1) = (I(i_I)-1);
                    i_I=i_I+1;

                    R(ii) = R(ii)+1;

                    uILoss = rand;
                    sumILoss = 0;

                    % Bring other compartments up to time tau
                    sumIGain = sumIGain - GainTrap;
                    % Trapazoid Method for IGain 
                    GainTrap = Dtau1*0.5*( mBeta*(Si*I(i_I-1)/Ni) + mBeta*((Si+Dtau1*dSdt)*I(i_I-1))/Ni );
                    sumIGain = sumIGain + GainTrap;

                    sumIDeath = sumIDeath - DeathTrap;
                    % Trapazoid Method for IDeath
                    DeathTrap = Dtau1*(mDeath*I(i_I-1));
                    sumIDeath = sumIDeath + DeathTrap;

                    Dtau = Dtau-Dtau1;

                elseif(pos == 3)
                    I(i_I+1) = (I(i_I)-1);
                    i_I=i_I+1;

                    uIDeath = rand;
                    sumIDeath = 0;

                    % Bring other compartments up to time tau
                    sumIGain = sumIGain - GainTrap;
                    % Trapazoid Method for IGain 
                    GainTrap = Dtau1*0.5*( mBeta*(Si*I(i_I-1)/Ni) + mBeta*((Si+Dtau1*dSdt)*I(i_I-1))/Ni );
                    sumIGain = sumIGain + GainTrap;

                    sumILoss = sumILoss - LossTrap;
                    % Trapazoid Method for ILoss
                    LossTrap = Dtau1*(mGamma*I(i_I-1));
                    sumILoss = sumILoss + LossTrap;

                    Dtau = Dtau-Dtau1;

                end
                LastIEvent = i_I;
    %             TimePassed = TimePassed + Dtau;

            else
                stayWhile = false;
            end

            if((AbsT > ContT) || (TimePassed >= dt))
                stayWhile = false;
            end

        end
    elseif(doDisc == 0)
        i_I = i_I + 1;
        
        % Update both System using Forward Euler
        
        dSdt = mBirth*Ni - mDeath*S(ii-1) - mBeta*S(ii-1)*I(i_I-1)/Ni;
        S(ii) = S(ii-1) + dt*dSdt;
        dRdt =  - mDeath*R(ii-1) + mGamma*I(i_I-1);
        R(ii) = R(ii-1) + dt*dRdt;
        
        dIdt = mBeta*S(ii-1)*I(i_I-1)/Ni - mGamma*I(i_I-1) - mDeath*I(i_I-1);
        I(i_I) = I(i_I-1) + dt*dIdt;
        tau(i_I) = ContT;
        tau_disc(i_I) = dt;
        
        LastIEvent = i_I;
        % Calculate N
        Ni = S(ii) + R(ii) + I(i_I);
        
    end
    AbsT = ContT;
    
    
     
end
toc;

% Clear everything that is not needed
I(LastIEvent:end) = [];
tau(LastIEvent:end) = [];
tau_disc(LastIEvent:end) = [];
tau_mask = tau(tau_disc ~= dt);
%%
% figure;
% skip = 10^4;
% hold on;
% 
% plot(0:skip*dt:t_final,S(1:skip:end),'-','linewidth',1.5)
% plot(tau(1:skip:end),I(1:skip:end),'.-')
% plot(0:skip*dt:t_final,R(1:skip:end),'-','linewidth',1.5)
% 
% % set(gca, 'YScale', 'log')
% legend('S','I','R')
% title('Hybrid')
% hold off;

%%
% clf;
figure;
subplot(2,2,1)
hold on;

plot(0:dt:t_final,S,'-','linewidth',1.5)
plot(tau,I,'.-')
plot(0:dt:t_final,R,'-','linewidth',1.5)

% set(gca, 'YScale', 'log')
legend('S','I','R')
title('Hybrid')
hold off;

subplot(2,2,3)
h = histogram(tau, 0:100*dt:t_final);
axis([0 t_final 0 1.25*max(h.BinCounts)])
ylabel('# Stochastic Events per 100dt')
xlabel('Time')
subplot(2,2,4)
h = histogram(tau_mask, 0:100*dt:t_final);
axis([0 t_final 0 1.25*max(h.BinCounts)])
ylabel('# Stochastic Events per 100dt')
xlabel('Time')


SPlane = zeros(length(0:dt:t_final),1);
IPlane = zeros(length(0:dt:t_final),1);



ii=1; jj=1;
for ContT=0:dt:t_final
    SPlane(ii) = S(ii);
    if(tau(jj) < ContT)
        jj=jj+1;
    end
    if(jj>length(I))
        jj = length(I);
    end
    IPlane(ii) = I(jj);
    ii=ii+1;
end

subplot(2,2,2)
plot(SPlane,IPlane,'-','linewidth',1.5)
ylabel('I')
xlabel('S')
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')

axis([0.5*min(S) N0 1 1.5*max(I)])

%%
function [Flag, dI] = IsDiscrete(ss,iI,rr,p,dt,threshold)
    Flag = 1;
    
    dIdt = p.mBeta*ss*iI/(ss+iI+rr) + p.mGamma*iI + p.mDeath*iI;
    dI = abs(dIdt)*dt;
    
    if(dI > threshold)
        Flag = 0;
    end

end
