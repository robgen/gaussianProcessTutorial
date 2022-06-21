%% Input

Optimiser               = 'quasinewton';
Basis                   = 'constant';
KernelType              = 'ardsquaredexponential'; % ARD kernels cannot be optimised.
InitialNoiseVariance    = 0.7;
constantSIG             = true;
optimise                = 'auto'; %'none'; % 

grey = linspace(0, 0.5, 4);
font = 18;

%% Prepare input set

a = 12;
b = 150;
c = 10;
d = -30;
e = 100;

training(:,1) = linspace(0,10,100);

for v = 4 : -1 : 1
    training(:,v+1) = (a+randi(10,1))*sin(training(:,1)+b+rand) + ...
        (d-rand)*log(training(:,1)+c-rand) + e*rand(size(training,1),1);
end

tab = array2table(training, ...
    'VariableNames', {'x', 'y1', 'y2', 'y3', 'y4' });

%% Fit GP

fittedGp = cell(4,1);
for v = 1 : 4
    fittedGp{v} = fitrgp(tab{:,1}, tab{:,v+1}, ...
        'Optimizer', Optimiser, 'BasisFunction',Basis, ...
        'KernelFunction', KernelType, 'Sigma', InitialNoiseVariance, ...
        'ConstantSigma', constantSIG, 'OptimizeHyperparameters', optimise );
    close all
end

%% Plot

close all

Xplot = linspace( min(training(:,1)), max(training(:,1)), 1000 )';

posX = [ 272 833 272 833 ];
posY = [ 535 535 41  41];

driftDS = [ 0.2 0.7 1.7 2.5 ];

for v = 1 : 4
    
    [predictedSA, ~, predictedSAconfidence] = predict(fittedGp{v}, Xplot );
    
    figure('Position', [posX(v) posY(v) 560 420]); hold on
    plot(Xplot, predictedSA, 'Color', grey(v)*[1 1 1], 'LineWidth', 2)
    area = fill([Xplot; flipud(Xplot)], [predictedSAconfidence(:,1); flipud(predictedSAconfidence(:,2))], [1 1 1]*grey(v), 'FaceAlpha', 0.1); set(area,'Edgecolor','w')
    
    scatter(tab{:,1}, tab{:,v+1}, 40, grey(v)*[1 1 1], 'filled')
    
    legend('Mean prediction', '95% confidence bounds', 'Training set', 'Location', 'NorthEast')
    
    title( sprintf('V%d', v) )
    xlabel('X'); ylabel('Y')
    set(gca, 'FontSize', font)
    
end
