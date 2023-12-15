
%% Create Environment
obsInfo = rlNumericSpec([7,1]);
obsInfo.Name = "Bicycle Model States";
obsInfo.Description = 'e_y, de_y, e_phi, de_phi, u_prev, v_x, R';

maxSteer = pi / 6;

actInfo = rlNumericSpec(1);
actInfo.Name = "Steering Action";
actInfo.Description = 'delta';
actInfo.LowerLimit = -maxSteer;
actInfo.UpperLimit = maxSteer;

addpath("../UtilFunctions")
env = rlFunctionEnv(obsInfo, actInfo, "environmentStepFunction", "environmentResetFunction");


%% Create Actor DDPG

% Define path for the state input
statePath = [
    featureInputLayer(prod(obsInfo.Dimension),Name="NetObsInLayer")
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(200,Name="sPathOut")];

% Define path for the action input
actionPath = [
    featureInputLayer(prod(actInfo.Dimension),Name="NetActInLayer")
    fullyConnectedLayer(200,Name="aPathOut",BiasLearnRateFactor=0)];

% Define path for the critic output (value)
criticCommonPath = [
    concatenationLayer(1,2,Name="concat")
    reluLayer
    fullyConnectedLayer(1,Name="CriticOutput")];

% Create layerGraph object and add layers
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,criticCommonPath);

% Connect paths and convert to dlnetwork object
criticNetwork = connectLayers(criticNetwork,"sPathOut","concat/in1");
criticNetwork = connectLayers(criticNetwork,"aPathOut","concat/in2");

criticNetwork1 = dlnetwork(criticNetwork);
criticNetwork2 = dlnetwork(criticNetwork);

critic1 = rlQValueFunction(criticNetwork1, ...
    obsInfo,actInfo,...
    ObservationInputNames="NetObsInLayer", ...
    ActionInputNames="NetActInLayer");
critic2 = rlQValueFunction(criticNetwork2, ...
    obsInfo,actInfo,...
    ObservationInputNames="NetObsInLayer", ...
    ActionInputNames="NetActInLayer");

actorCommonPath = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(128)
    reluLayer(Name="CommonRelu")
    ];
meanPath = [
    fullyConnectedLayer(200, Name="meanIn")
    reluLayer
    fullyConnectedLayer(prod(actInfo.Dimension), Name="meanOut")
    ];
stdPath = [
    fullyConnectedLayer(200, Name="stdIn")
    reluLayer
    fullyConnectedLayer(prod(actInfo.Dimension))
    softplusLayer(Name="stdOut")
    ];

actorNetwork = layerGraph(actorCommonPath);
actorNetwork = addLayers(actorNetwork, meanPath);
actorNetwork = addLayers(actorNetwork, stdPath);
actorNetwork = connectLayers(actorNetwork, "CommonRelu", "meanIn");
actorNetwork = connectLayers(actorNetwork, "CommonRelu", "stdIn");

actorNetwork = dlnetwork(actorNetwork);

actor = rlContinuousGaussianActor(actorNetwork,obsInfo,actInfo, ...
    ActionMeanOutputNames="meanOut", ...
    ActionStandardDeviationOutputNames="stdOut");

criticOptions = rlOptimizerOptions(LearnRate=1e-03,GradientThreshold=1);
actorOptions = rlOptimizerOptions(LearnRate=5e-04,GradientThreshold=1);

agentOptions = rlSACAgentOptions(...
    SampleTime=1e-2,...
    ActorOptimizerOptions=actorOptions,...
    CriticOptimizerOptions=criticOptions,...
    ExperienceBufferLength=1e6,...
    MiniBatchSize=128);

agent = rlSACAgent(actor,[critic1 critic2],agentOptions);

getAction(agent, rand(obsInfo(1).Dimension))

opt = rlTrainingOptions(...
    MaxEpisodes=20000,...
    MaxStepsPerEpisode=100,...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=100*3, ...
    SaveAgentCriteria="EpisodeReward",...
    SaveAgentValue=100*3);
trainResults = train(agent,env,opt);

simOptions = rlSimulationOptions(MaxSteps=10000);
experience = sim(env,agent,simOptions);