%% Create joint prior of the training and test points

% assign training points (tSTARi, fSTARi)
NpointsPerDim = 3;
MINt = 0; MAXt = 20;

[x1,x2] = meshgrid( linspace(MINt, MAXt, NpointsPerDim)' , linspace(MINt, MAXt, NpointsPerDim)' );

X1 = reshape(x1, numel(x1), 1);
X2 = reshape(x2, numel(x2), 1);

f  = sin(rand * X1) + cos(rand * X2) + rand(NpointsPerDim*NpointsPerDim,1);

tab = table(X1, X2);

%% Fit the GP using the matlab built in function

%
KernelType = 'SquaredExponential';
GPmodel = fitrgp(tab, f,'KernelFunction',KernelType);

% predict for some values of tstar
[x1star, x2star] = meshgrid( MINt : 0.1 : MAXt, MINt : 0.1 : MAXt);

TABstar = table( reshape(x1star,numel(x1star),1) , reshape(x2star,numel(x2star),1), 'VariableNames', {'X1', 'X2'});

[fstarAVG, ~, fstar95] = predict(GPmodel, TABstar); % media e intervalli di confidenza

FstarAvg = reshape(fstarAVG, size(x1star,1), size(x1star,2));
Fstar95lower = reshape(fstar95(:,1), size(x1star,1), size(x1star,2));
Fstar95upper = reshape(fstar95(:,2), size(x1star,1), size(x1star,2));

%% Plot

figure
hold on

surf(x1star, x2star, FstarAvg, 'FaceAlpha',0.9)
surf(x1star, x2star, Fstar95lower, 'FaceAlpha',0.2)
surf(x1star, x2star, Fstar95upper, 'FaceAlpha',0.2)

plot3(X1, X2, f, 'ok', 'MarkerSize', 15, 'MarkerFaceColor','k')

%title(sprintf('%s = %2.2f ; %s = %2.2f', GPmodel.KernelInformation.KernelParameterNames{1}, GPmodel.KernelInformation.KernelParameters(1), GPmodel.KernelInformation.KernelParameterNames{2}, GPmodel.KernelInformation.KernelParameters(2)))
xlabel('x1'); ylabel('x2'); zlabel('prediction')
set(gca, 'Fontsize', 16)
