function [S,R, time , I,tau] = SIR_Dem_Hybrid(mBeta, mGamma, mDeath, N0, I0, dt, t_final, RND_SEED)

rng(RND_SEED);

mBirth = mDeath;

% N0 = 100000;
% I0 = 5;
R0 = 0;
S0 = N0-I0-R0;

% Initialising stochasitic info
sumILoss = 0; sumIGain = 0; sumIDeath = 0; 
uILoss = rand; uIGain = rand; uIDeath = rand;

% Initialising stochasitic compartment makes for shorter run time
I = zeros(length(0:dt:t_final),1); I(1) = I0;
tau = zeros(length(0:dt:t_final),1); tau(1) = 0.0;
TauArray = zeros(1,3);
i_I = 1; LastIEvent = 1;

% Initialising deterministic compartment compartment
R = zeros(length(0:dt:t_final),1); R(1) = R0;
S = zeros(length(0:dt:t_final),1); S(1) = S0;
N = zeros(length(0:dt:t_final),1); N(1) = N0;
ii = 1; 
time = [0:dt:t_final];
% Track Absolute time
AbsT = 0; 

for ContT=dt:dt:t_final
    ii=ii+1;
    
    % Update both S and R using Forward Euler
    dSdt = mBirth*N(ii-1) - mDeath*S(ii-1);
    S(ii) = S(ii-1) + dt*dSdt;
    dRdt =  - mDeath*R(ii-1);
    R(ii) = R(ii-1) + dt*dRdt;
    
    % Calculate N
    N(ii) = S(ii) + R(ii) + I(i_I);
    
    Si=S(ii-1); Ri=R(ii-1); Ni=N(ii-1);
    Dtau = dt;
    stayWhile = true;
    TimePassed = 0;
    while stayWhile
        
        % Caclculate putative times since last event using Trapazoid Rule
        % Trapazoid Method for ILoss
        LossTrap = Dtau*(mGamma*I(i_I));
        sumILoss = sumILoss + LossTrap;

        % Trapazoid Method for IGain 
        GainTrap = Dtau*0.5*( mBeta*(Si*I(i_I)/Ni) + mBeta*(S(ii)*I(i_I))/N(ii) );
        sumIGain = sumIGain + GainTrap;

        % Trapazoid Method for IDeath
        DeathTrap = Dtau*(mDeath*I(i_I));
        sumIDeath = sumIDeath + DeathTrap;

        % Check if an event occurs within current Forward Euler Step, 
        % and find time of occurance via linearising solution over current step
        DTauGain = 0.0; DTauLoss = 0.0; DTauDeath=0.0;
        if(uIGain < (1 - exp(-sumIGain)) )
            DTauGain = sqrt((Si/dSdt)^2+2.0*(log(1/(1-uIGain))-(sumIGain-GainTrap))/(mBeta*I(i_I)*dSdt/N(ii)))-(Si/dSdt);
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
    AbsT = ContT;
    
    
     
end
% toc;

% Clear everything that is not needed
I(LastIEvent:end) = [];
tau(LastIEvent:end) = [];

end

