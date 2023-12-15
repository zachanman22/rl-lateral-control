
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

criticOptions = rlOptimizerOptions(LearnRate=1e-03,GradientThreshold=1);
actorOptions = rlOptimizerOptions(LearnRate=5e-04,GradientThreshold=1);

agentOptions = rlDDPGAgentOptions(...
    SampleTime=1e-2,...
    ActorOptimizerOptions=actorOptions,...
    CriticOptimizerOptions=criticOptions,...
    ExperienceBufferLength=1e6,...
    MiniBatchSize=128);
agentOptions.NoiseOptions = rl.option.GaussianActionNoise;
agentOptions.NoiseOptions.Variance = 0.4;
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

agent = rlDDPGAgent(actor,critic,agentOptions);

%% Transition Equation Approximation Network
% Observation and action paths
obsPath = featureInputLayer(obsInfo.Dimension(1),Name="obsIn");
actionPath = featureInputLayer(actInfo.Dimension(1),Name="actIn");

% Common path: concatenate along dimension 1
commonPath = [concatenationLayer(1,2,Name="concat")
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(obsInfo.Dimension(1),Name="nextObsOut")];

% Add layers to layerGraph object
transNet = layerGraph(obsPath);
transNet = addLayers(transNet,actionPath);
transNet = addLayers(transNet,commonPath);

% Connect layers
transNet = connectLayers(transNet,"obsIn","concat/in1");
transNet = connectLayers(transNet,"actIn","concat/in2");

% Convert to dlnetwork object
transNet = dlnetwork(transNet);

% figure(1)
% plot(layerGraph(transNet))
% title("Transition Model Network")
% xlabel("18.5k parameters")
% summary(transNet)
% analyzeNetwork(transNet)

% Display number of weights
% summary(transNet)

transitionFcnAppx = rlContinuousDeterministicTransitionFunction( ...
    transNet,obsInfo,actInfo,...
    ObservationInputNames="obsIn",...
    ActionInputNames="actIn",...
    NextObservationOutputNames="nextObsOut");

rewardFunction = @rewardFcn;
isDoneFunction = @isDone;

generativeEnv = rlNeuralNetworkEnvironment(obsInfo, actInfo, transitionFcnAppx, rewardFunction, isDoneFunction);


transitionOptimizerOptions = rlOptimizerOptions(...
    LearnRate=1e-4,...
    GradientThreshold=1.0);

agentOptions = rlMBPOAgentOptions(...
    MiniBatchSize=128, ...
    ModelExperienceBufferLength=1e6,...
    RealSampleRatio=0.2, ...
    TransitionOptimizerOptions=transitionOptimizerOptions);

agentOptions.ModelRolloutOptions.NoiseOptions = rl.option.GaussianActionNoise;

agent = rlMBPOAgent(agent, generativeEnv, agentOptions);
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