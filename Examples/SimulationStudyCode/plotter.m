master_path = 'CleanedData/';
dtObserve = 10^(-2); tFinal = 1500;
TimeObserve = 0:dtObserve:tFinal;
HowManySimsToDo = 100;

IData_JFS = zeros(HowManySimsToDo,length(TimeObserve));
IData_Gil = zeros(HowManySimsToDo,length(TimeObserve));
        
jj = 1;
plotter = 0;
for N0ii = 6
    jj = jj + 1;
    plotter = plotter + 1;
    for ii=1:HowManySimsToDo
        ii
        DATA = readmatrix([master_path,'JumpSwitchFlow/JSF_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv']);
        IData_JFS(ii,:) = DATA(:,3);
        
        DATA = readmatrix([master_path,'Gillespie/GIL_N0Index_', num2str(N0ii) , '_iiSeed_', num2str(ii) ,'.csv']);
        IData_Gil(ii,:) = DATA(:,3);
    end
    plotter = plotter + 1;

end
IData_JFS(:,end)=[];
%%
1+1
confidenceLevel = 0.95;


JSF_averageData = mean(IData_JFS);
tValue = tinv((1 + confidenceLevel) / 2, size(IData_JFS, 1) - 1);
stdDev_JFS = std(IData_JFS);
JFS_Error = tValue * stdDev_JFS / sqrt(size(IData_JFS, 1));
JSF_envelopeUpper = JSF_averageData + JFS_Error;
JSF_envelopeLower = JSF_averageData - JFS_Error;

Gil_averageData = mean(IData_Gil);
tValue = tinv((1 + confidenceLevel) / 2, size(IData_Gil, 1) - 1);
stdDev_Gil = std(IData_Gil);
Gil_Error = tValue * stdDev_Gil / sqrt(size(IData_Gil, 1));
Gil_envelopeUpper = Gil_averageData + Gil_Error;
Gil_envelopeLower = Gil_averageData - Gil_Error;

figure;
hold on;

% Plot the average data
plot(TimeObserve(1:end-1), JSF_averageData, 'color', 1/255*[136 170 219], 'LineWidth', 2);
plot(TimeObserve, Gil_averageData, 'color', 1/255*[255 154 162] , 'LineWidth', 2);


fill([TimeObserve(1:end-1), fliplr(TimeObserve(1:end-1))], [JSF_envelopeUpper, fliplr(JSF_envelopeLower)], 1/255*[136 170 219] , 'FaceAlpha', 0.5);
fill([TimeObserve, fliplr(TimeObserve)], [Gil_envelopeUpper, fliplr(Gil_envelopeLower)], 1/255*[255 154 162], 'FaceAlpha', 0.5);

plot(TimeObserve(1:end-1), JSF_averageData, 'color', 1/255*[136 170 219], 'LineWidth', 2);
plot(TimeObserve, Gil_averageData, 'color', 1/255*[255 154 162] , 'LineWidth', 2);

hold off;
% ax = gca;
% ax.YAxis.Scale ="log";

ylabel('Infected population','fontsize',14)
xlabel('Time (days)','fontsize',14)

legend({'Jump-switch-flow','Gilllespie','95% CI','95% CI'},'fontsize',14,'location','southeast','NumColumns',2)
title('Average Trajectory for N_0 = 10^3','fontsize',16)

% axis([0 100 0 310])
