
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
commonPath = [
    additionLayer(2,Name="add")
    reluLayer
    fullyConnectedLayer(1,Name="CriticOutput")];

% Create layerGraph object and add layers
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);

% Connect paths and convert to dlnetwork object
criticNetwork = connectLayers(criticNetwork,"sPathOut","add/in1");
criticNetwork = connectLayers(criticNetwork,"aPathOut","add/in2");
criticNetwork = dlnetwork(criticNetwork);

critic = rlQValueFunction(criticNetwork, ...
    obsInfo,actInfo,...
    ObservationInputNames="NetObsInLayer", ...
    ActionInputNames="NetActInLayer");

actorNetwork = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(200)
    reluLayer
    fullyConnectedLayer(prod(actInfo.Dimension))
    tanhLayer
    scalingLayer(Scale=max(actInfo.UpperLimit))
    ];

actorNetwork = dlnetwork(actorNetwork);

actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);

% figure(1)
% plot(layerGraph(actorNetwork))
% title("Actor Network")
% xlabel("27k parameters")
% summary(actorNetwork)
% analyzeNetwork(actorNetwork)
% figure(2)
% plot(layerGraph(criticNetwork))
% title("Critic Network")
% xlabel("27.4k parameters")
% summary(criticNetwork)
% analyzeNetwork(criticNetwork)

criticOptions = rlOptimizerOptions(LearnRate=1e-03,GradientThreshold=1);
actorOptions = rlOptimizerOptions(LearnRate=5e-04,GradientThreshold=1);

agentOptions = rlDDPGAgentOptions(...
    SampleTime=1e-2,...
    ActorOptimizerOptions=actorOptions,...
    CriticOptimizerOptions=criticOptions,...
    ExperienceBufferLength=1e6,...
    MiniBatchSize=128);

agentOptions.NoiseOptions.Variance = 0.4;
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

agent = rlDDPGAgent(actor,critic,agentOptions);

%%

opt = rlTrainingOptions(...
    MaxEpisodes=10000,...
    MaxStepsPerEpisode=300,...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=300*3, ...
    SaveAgentCriteria="EpisodeReward",...
    SaveAgentValue=300*3);
trainResults = train(agent,env,opt);

simOptions = rlSimulationOptions(MaxSteps=10000);
experience = sim(env,agent,simOptions);