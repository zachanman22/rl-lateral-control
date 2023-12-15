
To run DDPG Training:

1. Determine whether you will train with the Base reward or Modified reward.
2. Navigate to the UtilFunctions folder and edit environmentStepFunction.m so that the variable modifiedReward = 0 if using base reward or modifiedReward = 1 if using the modified reward.
3. Navigate to the DDPG_Training folder and run DDPGTraining.m.

To run MBPO Training:

1. Determine whether you will train with the Base reward or Modified reward.
2. Navigate to the UtilFunctions folder and edit environmentStepFunction.m and rewardFcn.m so that the variable modifiedReward = 0 if using base reward or modifiedReward = 1 if using the modified reward (in both files).
3. Navigate to the MBPO_Training folder and run MBPOTraining.m.

To run Imitation Learning:

1. Navigate to the imitation-learning folder and follow the instructions in ImitateMPCControllerForLateralControl.mlx.

To simulate the steering dynamics:

Note: There are example agents in the folders exampleDDPGAgentsBase, exampleDDPGAgentsModified, and exampleMBPOAgentsBase
1. Load the agent .mat file that you want to simulate by double clicking it in the Matlab file explorer. This should load saved_agent to your workspace.
2. Determine whether the agent used the Base reward or Modified reward.
3. Navigate to the UtilFunctions folder and edit environmentStepFunction.m so that the variable modifiedReward = 0 if using base reward or modifiedReward = 1 if using the modified reward.
4. Navigate to the Simulate_Steering folder and run simulateSteeringDynamics.m. To change the name of the plot title, change the variable named plotTitle. To add the difference between consecutive steering angles throughout the trajectory, change the variable plotDSteer to 1.

To generate MPC trajectories:

1. Navigate to the MPC folder and run mpcExpertTest.m
2. After the file is finished running, in the Matlab workspace, there will be a variable called trajectories. Save the variable by right clicking it and then clicking Save As. Name the .mat file to something like exampleTrajectories.mat. The trajectories are 1000 x 1000 x 9. The first column is the number of trajectories generated. The number of trajectories to generate can be changed by changing the variable numTrajectories in mpcExpertTest.m. The second column is the number of time steps in each trajectory. The number of time steps per trajectory can be changed by changing the variable numSteps in mpcExpertTest.m. The third column is all of the states which are [e_y, de_y, e_angle, de_angle, u_prev, u, v_x/R, v_x, R].
