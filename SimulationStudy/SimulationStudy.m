clear all;

dt = 10^(-3);
t_final = 1000;

mBeta = 1.5/7; % Infect "___" people a week
mGamma = 0.5/7; % infecion for "___" weeks
mDeath = 1/(2*365); %lifespan

N0 = 1000;
I0 = 2;

RndSeed = 3;

tic;
[S,R, time , I,tau] = SIR_Dem_Hybrid(mBeta, mGamma, mDeath, N0, I0, dt, t_final, RndSeed);
% [S,R, time , I,tau] = SIR_Dem_GillespieDirect(mBeta, mGamma, mDeath, N0, I0, t_final, RndSeed);
% [S,R, time , I,tau] = SIR_Dem_ModifiedNextReaction(mBeta, mGamma, mDeath, N0, I0, t_final, RndSeed);
% [S,R, time , I,tau] = SIR_Dem_TauLeaping(mBeta, mGamma, mDeath, N0, I0, dt, t_final, RndSeed);
ClockTime = toc;
filename = ['Results/SIR_Dem_Hybrid_',num2str(RndSeed),'.mat'];
save(filename);
%%

% clf;
% figure;
subplot(2,2,1)
hold on;

plot(time,S,'-','linewidth',1.5)
plot(tau,I,'.-')
plot(time,R,'-','linewidth',1.5)

% set(gca, 'YScale', 'log')
legend('S','I','R')
title('Hybrid')
hold off;

subplot(2,2,3)
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
subplot(2,2,2)
plot(SPlane,IPlane,'-','linewidth',1.5)
ylabel('I')
xlabel('S')
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')

axis([0.5*min(S) N0 1 1.5*max(I)])