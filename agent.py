from keras import layers, models, optimizers, initializers
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import keras
import numpy as np
import copy
from task import Task
from collections import namedtuple, deque
import random

class Actor: 
    
    def __init__(self, state_size, action_size, action_low, action_high):
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        
        self.build_model()
        
    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states') # create input layer of shape state_size with name "states"
        
        net = layers.Dense(units=32, activation='relu')(states) # create hidden layer with 32 units and activation relu and add to 'states' layer
        
        net = layers.Dense(units=64, activation=None)(net) # add another hidden layer with 64 units and activation relu and add to net
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=32, activation='relu')(net) # add another hidden layer with 32 units and activation relu and add to net
        initializer = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions', kernel_initializer=initializer)(net) # add output layer with action_size units, activation sigmoid generating action outputs in the range [0,1], name of layer is raw_actions
        
        
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions) # add another layer that scales previous outputs to desired range for each action dimension
        
        self.model = models.Model(inputs=states, outputs=actions) # create Keras model (includes all layers needed in computation of outputs from inputs)
        
        action_gradients = layers.Input(shape=(self.action_size,)) # ??
        loss = K.mean(-action_gradients * actions) # define loss function using action value gradients
        
        optimizer = optimizers.Adam(lr=0.0001) # initialize Adam optimizer
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss) # update trainable weights using optimizer
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], outputs=[], updates=updates_op) # ?? 
         

class Critic:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states') # create input layer of size state_size for states
        actions = layers.Input(shape=(self.action_size,), name='actions') # create input layer of size action_size for actions

        # create network for states
        net_states = layers.Dense(units=32, activation='relu')(states) 
        net_states = layers.Dense(units=64, activation=None)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        # create network for actions
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation=None)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)

        # merge states and actions layers
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net) # apply relu activation
        initializer = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003) # define initial weights of final layer
        Q_values = layers.Dense(units=1, name='q_values', kernel_initializer=initializer)(net) # add layer with one unit that outputs Q-value for the given state-action pair

        self.model = models.Model(inputs=[states, actions], outputs=Q_values) # create model

        # set optimizer and compile model using Adam optimizer and MSE loss function
        optimizer = optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=optimizer, loss='mse')

        action_gradients = K.gradients(Q_values, actions) # compute action gradients

        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients) # ?? 
        

class ReplayBuffer:
    
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size) # initiate memory by using double-ended queue with buffer = buffer_size
        self.batch_size = batch_size # set batch size for sampling
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"]) # set template of each experience
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done) # add each experience to memory
        self.memory.append(e)
        
    def sample(self, batch_size=64):
        return random.sample(self.memory, k=self.batch_size) # sample a batch from memory
    
    def __len__(self):
        return len(self.memory) # return memory size
        
        
class QuadAgent():
    
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high 
    
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high) # initiate Local Actor
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high) # initiate Target Actor
        
        self.critic_local = Critic(self.state_size, self.action_size) # initiate Local Critic
        self.critic_target = Critic(self.state_size, self.action_size) #initiate Target Critic
        
        self.critic_target.model.set_weights(self.critic_local.model.get_weights()) # use Local Critic weights to set Target Critic weights
        self.actor_target.model.set_weights(self.actor_local.model.get_weights()) # use Local Actor weights to set Target Actor weights
        
        # set various parameters
        self.exploration_mu = 0.2
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma) # initiate OUNoise 
        self.count = 0 # count of steps taken in episode
        
        # initiate memory parameters and instance
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
    
        self.gamma = 0.99  # discount factor
        self.tau = 0.05  # for soft update of target parameters
        
    
    def reset_episode(self): # reset noise, task, state
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.count = 0 
        self.episode_reward = 0
        self.avg_reward = 0
        return state
    
    def step(self, action, reward, next_state, done): 
         # Save experience / reward
        self.count += 1
        self.episode_reward += reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state
        
    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size]) # reshape state array to 1-dim
        action = self.actor_local.model.predict(state)[0] # select action from local actor model 
        return list(action + self.noise.sample())  # add some noise for exploration
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
    
    def score(self):
        self.avg_reward = self.episode_reward / float(self.count) if self.count else 0.0
        return self.episode_reward, self.avg_reward

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state