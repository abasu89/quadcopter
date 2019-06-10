import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent.""" 
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., hover_pos=None): # takes in the different aspects of starting pose to create instance
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            hover_pos: target/goal (x,y,z) hover position for the agent
            hover_angle: target/goal hover Euler angles in x,y,z axes for the agent
        """
        # Simulation
        # specify different attributes of Task class
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) # allocate simulation parameters to self
        self.action_repeat = 3 # ???

        self.state_size = self.action_repeat * 6 # ???
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        # Goal
        self.hover_pos = hover_pos if hover_pos is not None else np.array([0., 0., 10.]) #set default hover position to [0, 0, 10]
        self.hover_pos_v = np.array([0., 0., 0.]) #set hover velocity in all directions to 0
        #self.hover_angle = np.array([0., 0., 0.]) if hover_angle is None else np.copy(hover_angle) #set default hover orientation to horizontal (0,0,0)
        self.hover_angular_v = np.array([0., 0., 0.]) #set hover angle velocity in all directions to 0
        

    def get_reward(self):
        """Uses current pose, velocity and orientation of sim to return reward."""
        #reward = 10000.0 -3*(abs(self.sim.pose[:3] - self.hover_pos)).sum()
        reward = 0
        penalty = abs(abs(self.sim.pose[:3] - self.hover_pos).sum() - abs(self.sim.v).sum())
        reward -= penalty
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all) # ??? state_size = action_repeat * 6
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state