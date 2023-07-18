% clear all;
addpath('Method')
dt = 10^(-4);
t_final = 1000;

mBeta = 1.4/7; % Infect "___" people a week
mGamma = 0.7/7; % infecion for "___" weeks
mDeath = 1/(1.5*365); %lifespan

R_0 = mBeta/(mGamma+mDeath)

N0 = 10^5;
I0 = 2;

RndSeed = RndSeed+1; %61,62, 68

tic;
[S,R, ~ , I,tau] = SIR_Dem_Hybrid(mBeta, mGamma, mDeath, N0, I0, dt, t_final, RndSeed);
% [S,R, ~ , I,tau] = SIR_Dem_GillespieDirect(mBeta, mGamma, mDeath, N0, I0, t_final, RndSeed);
% [S,R, ~ , I,tau] = SIR_Dem_ModifiedNextReaction(mBeta, mGamma, mDeath, N0, I0, t_final, RndSeed);
% [S,R, ~ , I,tau] = SIR_Dem_TauLeaping(mBeta, mGamma, mDeath, N0, I0, dt, t_final, RndSeed);
ClockTime = toc
% filename = ['Results/Hybrid/RndSeed_',num2str(RndSeed),'_N0_',num2str(N0),'.csv'];
% tic;
% save(filename);
% toc;
% filename = ['Results/Hybrid/RndSeed_',num2str(RndSeed),'_N0_',num2str(N0),'.mat'];
% tic;
% save(filename);
% toc;


%%


% clf;
figure;
subplot(3,1,1)
hold on;

try 
    plot(tau,S,'-','linewidth',1.5)
    plot(tau,I,'.-')
    plot(tau,R,'-','linewidth',1.5)
catch
    time = 0:dt:t_final;
    plot(time,S,'-','linewidth',1.5)
    plot(tau,I,'.-')
    plot(time,R,'-','linewidth',1.5)
end

% set(gca, 'YScale', 'log')
legend('S','I','R')
title('Hybrid')
hold off;

subplot(3,1,2)
h = histogram(tau, 0:100*dt:t_final);
axis([0 t_final 0 1.25*max(h.BinCounts)])
ylabel('# Stochastic Events')
xlabel('Time')

try
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
catch
    SPlane = S;
    IPlane = I;
end
subplot(3,1,3)
plot(SPlane,IPlane,'-','linewidth',1.5)
ylabel('I')
xlabel('S')
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')

axis([0.5*min(S) N0 1 1.5*max(I)])