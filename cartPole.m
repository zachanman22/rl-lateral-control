env = rlPredefinedEnv("CartPole-Continuous");

obsInfo = getObservationInfo(env);
numObservations = obsInfo.Dimension(1);
actInfo = getActionInfo(env);

agent = rlSACAgent(obsInfo, actInfo);

opt = rlTrainingOptions(...
    MaxEpisodes=10000,...
    MaxStepsPerEpisode=1000,...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=2800, ...
    SaveAgentCriteria="EpisodeReward",...
    SaveAgentValue=2900);
trainResults = train(agent,env,opt);