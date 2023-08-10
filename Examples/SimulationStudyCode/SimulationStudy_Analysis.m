% clear all;
% close all;
addpath('Solver');

master_path = 'CleanedData/';

N0_Vector = [2:0.25:6.25];
HowManySimsToDo = 100;

dtObserve = 10^(-2); tFinal = 1500;
TimeObserve = 0:dtObserve:tFinal;

%%

IData_JFS = zeros(length(N0_Vector),HowManySimsToDo,length(TimeObserve));
IData_Gil = zeros(length(N0_Vector),HowManySimsToDo,length(TimeObserve));
        
jj = 1;
plotter = 0;
for N0ii = N0_Vector
    jj = jj + 1;
    plotter = plotter + 1;
    for ii=1:HowManySimsToDo
        
        DATA = readmatrix([master_path,'JumpSwitchFlow/JSF_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv']);
%         IData_JFS(jj,ii,:) = DATA(:,3);
        
        if( mod(N0ii,1) == 0)
            subplot(length(N0_Vector),2,plotter)
            hold on;
            plot(DATA(:,1),DATA(:,3),'.','linewidth',2)
            hold off;
        end
        
        DATA = readmatrix([master_path,'Gillespie/GIL_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv']);
%         IData_Gil(jj,ii,:) = DATA(:,3);

        if( mod(N0ii,1) == 0)
            subplot(length(N0_Vector),2,plotter+1)
            hold on;
            plot(DATA(:,1),DATA(:,3),'.','linewidth',2)
            hold off;
        end
    end
    plotter = plotter + 1;

end

%%

jj = 1;
for N0ii = N0_Vector
    jj = jj + 1;
    for ii=1:HowManySimsToDo
        
        DATA = readmatrix([master_path,'JumpSwitchFlow/JFS_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv']);
        IData_JFS(jj,ii,:) = DATA(:,3);
        
        DATA = readmatrix([master_path,'Gillespie/GIL_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv']);
        IData_Gil(jj,ii,:) = DATA(:,3);
    end

end



%%
figure;

TimerData = readmatrix([master_path,'META.csv']);

hold on;
scatter(10.^(TimerData(:,1)),TimerData(:,3),'filled')
scatter(10.^(TimerData(:,1)),TimerData(:,4),'filled')
hold off;
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
xlabel('N_0 - log scale ','fontsize',14)
ylabel('CPU run time (seconds) - log scale','fontsize',14)
legend('Hybrid','Gilllespie','fontsize',14,'location','northwest')
% exportgraphics(gca,'RunTime.png')
%%
figure;

hold on;
boxplot(TimerData(:,3),(TimerData(:,1)))
boxplot(TimerData(:,4),(TimerData(:,1)))
ax = gca;

xlabel('log(N_0)','fontsize',14)
ylabel('CPU run time (seconds) - log scale','fontsize',14)
legend('Hybrid','Gilllespie','fontsize',14,'location','northwest')
hold off;
ax.YAxis.Scale ="log";
% ax.XAxis.Scale ="log";

% Get the current x-axis tick values
xtickValues = get(gca, 'XTick');

% Calculate the indices for the ticks you want to keep (every third tick)
desiredIndices = 1:4:length(xtickValues);

% Extract the x-axis tick labels for the desired indices
desiredLabels = xtickLabels(desiredIndices);

% Extract the x-axis tick values for the desired indices
desiredTicks = xtickValues(desiredIndices);

% Set the new x-axis tick values
xticks(desiredTicks);

% Create corresponding tick labels for the desired tick positions
tickLabels = cell(size(desiredTicks));
for i = 1:length(desiredTicks)
    tickLabels{i} = desiredLabels{i};
end

% Set the tick labels
xticklabels(tickLabels);

legend('Hybrid','Gilllespie','fontsize',14,'location','northwest')

%%
N0_Vector = [2:0.25:6.5];
HowManySimsToDo = 100;

JSF_Average = zeros(length(N0_Vector),1);
JSF_Error = zeros(length(N0_Vector),1);
Gil_Average = zeros(length(N0_Vector),1);
Gil_Error = zeros(length(N0_Vector),1);

confidenceLevel = 0.95;
zValue = norminv(1 - (1 - confidenceLevel) / 2);

ii=0;
for N0ii=N0_Vector
    ii=ii+1;
    strt=(ii-1)*HowManySimsToDo+1;
    fin = (ii)*HowManySimsToDo;
    
    JSF_Average(ii) = mean(TimerData(strt:fin,3));
    JSF_Error(ii) = zValue * std(TimerData(strt:fin,3)) / sqrt(HowManySimsToDo);

    Gil_Average(ii) = mean(TimerData(strt:fin,4));
    Gil_Error(ii) = zValue * std(TimerData(strt:fin,4)) / sqrt(HowManySimsToDo);
    
end

%%

hold on;
plot(10.^(N0_Vector), JSF_Average, 'o-','color', 1/255*[136 170 219],'linewidth',2);
plot(10.^(N0_Vector), Gil_Average, 'o-','color', 1/255*[255 154 162],'linewidth',2);
errorbar(10.^(N0_Vector), JSF_Average, JSF_Error,'color', 1/255*[136 170 219], 'LineStyle', 'none', 'CapSize', 10,'linewidth',2);
errorbar(10.^(N0_Vector), Gil_Average, Gil_Error,'color', 1/255*[255 154 162], 'LineStyle', 'none', 'CapSize', 10,'linewidth',2);
xlabel('N_0 - log scale','fontsize',14)
ylabel('CPU run time (seconds) - log scale','fontsize',14)
legend('Jump-switch-flow','Gilllespie','fontsize',14,'location','northwest')
hold off;
ax = gca;
ax.YAxis.Scale ="log";
ax.XAxis.Scale ="log";
axis([10^2 10^6.5 10^-1.5 10^3 ])
% exportgraphics(gca,'RunTime.png')