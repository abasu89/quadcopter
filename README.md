# Teach a Quadcopter How to Fly
This project aims to teach a quadcopter how to hover at a (x,y,z) position specified by the user. Please note that the project is still ongoing.

The project consists of four files:

1) physics_sim.py: This file contains the simulator for the quadcopter. 

2) task.py: The task/environment is specified in this file.

3) agent.py: A Deep Deterministic Policy Gradient (DDPG) learning agent is implemented in this file.

4) Quadcopter_Project.ipynb: This is the main notebook used to train the agent, evaluate its learning behavior via the reward function, and provide reflections on the project.

## References
The DDPG agent was implemented using the following conference paper: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
