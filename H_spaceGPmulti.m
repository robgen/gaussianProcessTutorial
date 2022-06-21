%% Create joint prior of the training and test points

% assign training points (tSTARi, fSTARi)
NpointsPerDim = 3;
MINt = 0; MAXt = 20;

[x1,x2] = meshgrid( linspace(MINt, MAXt, NpointsPerDim)' , linspace(MINt, MAXt, NpointsPerDim)' );

% temporary form of the input/output training data
Xtemp = { [reshape(x1, numel(x1), 1) reshape(x2, numel(x2), 1)] };
Ytemp = { sin(rand * Xtemp{1}(:,1)) + cos(rand * Xtemp{1}(:,2)) + rand(NpointsPerDim*NpointsPerDim,1) cos(rand * Xtemp{1}(:,2)) + sin(rand * Xtemp{1}(:,2)) + rand(NpointsPerDim*NpointsPerDim,1) };

%tab = table(X1, X2);

%% Fit multi output GP using neil lawrence's scripts

% set the options of the multi oputput Gaussian Process
options = multigpOptions('ftc');
options.optimiser = 'optimiMinimize'; 
options.kernType = 'gg';
options.nlf = 1;                        % number of latent functions

% set dimensions
inputDIM    = 2;
outputDIM   = size(Ytemp, 2) + options.nlf;

%% define the final form of the input/output training data

% When we want to include the structure of the latent force kernel within
% the whole kernel structure, and we don't have access to any data from the
% latent force, we just put ones in the vector X and empty in the vector y.

for j=1:options.nlf
   Y{j} = [];
   X{j} = zeros(1, inputDIM);  
end
for i = 1:size(Ytemp, 2)
  Y{i+options.nlf} = Ytemp{i};
  X{i+options.nlf} = Xtemp{i};
end

%% Create and train the multiGP

% Creates the model
model = multigpCreate(inputDIM, outputDIM, X, Y, options);

%% Plot

% figure
% hold on
% 
% surf(x1star, x2star, FstarAvg, 'FaceAlpha',0.9)
% surf(x1star, x2star, Fstar95lower, 'FaceAlpha',0.2)
% surf(x1star, x2star, Fstar95upper, 'FaceAlpha',0.2)
% 
% plot3(X1, X2, f, 'ok', 'MarkerSize', 15, 'MarkerFaceColor','k')
% 
% %title(sprintf('%s = %2.2f ; %s = %2.2f', GPmodel.KernelInformation.KernelParameterNames{1}, GPmodel.KernelInformation.KernelParameters(1), GPmodel.KernelInformation.KernelParameterNames{2}, GPmodel.KernelInformation.KernelParameters(2)))
% xlabel('x1'); ylabel('x2'); zlabel('prediction')
% set(gca, 'Fontsize', 16)
