import random
import numpy as np
# import gymnasium as gym
# import time
# from gymnasium import spaces
# import os
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import pickle


class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

class RBFFeatureEncoder:
    def __init__(self, env): # modify
        self.env = env
        # TODO init rbf encoder
        #sample some observations to fit the rbf
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

        # now a light preprocessing, just to have a good scaler. will use scikit pipeline
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(observation_examples)    #learning mean and std from datas

        #now I can init the sampler
        rbf_sampler = RBFSampler(gamma=1.0, n_components=100, random_state=1)  #gamma 1 is a common default, 100 features seems a reasonable trade off between performance and speed
        rbf_sampler.fit(scaler.transform(observation_examples))  #fit the rbf to scaled data   

        #store stuff, also size cuz is requested
        self.scaler = scaler
        self.rbf_sampler = rbf_sampler
        self.rbf_size = rbf_sampler.n_components

    def encode(self, state): # modify
        # TODO use the rbf encoder to return the features
        #first I scale the state using same transform as above
        # I gotta reshape too cuz a single state from the environment is  1D array [pos, vel] but the scaler wants 2D array (samples, features)
        scaled_state = self.scaler.transform(state.reshape(1, -1))    #(1,-1)-> 1 sample, n features. with -1 numpy infers automatically the n
        return self.rbf_sampler.transform(scaled_state).flatten()       #I want just the features so I flatten, for the Q function later

    @property
    def size(self): # modify
        # TODO return the number of features
        return self.rbf_size

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder, alpha=0.01, alpha_decay=1, 
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.995, final_epsilon=0.2, lambda_=0.9): # modify if you want (e.g. for forward view)
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lambda_ = lambda_
        
    def Q(self, feats):
        feats = feats.reshape(-1,1)
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)
        # TODO update the weights
        # copute TD error
        if done:
            td_target = reward      #if terminal state
        else:
            td_target = reward + self.gamma * np.max(self.Q(s_prime_feats))         #Q learning update error (so it uses maxQ)

        #TD error computation delta = y - Q(s,a)
        delta = td_target - self.Q(s_feats)[action]       #actual error computation, y - current Q

        # eligibility trace update
        self.traces = self.gamma * self.lambda_ * self.traces   #decay all traces
        self.traces[action] = self.traces[action] + s_feats     #add current features to trace

        #now I can update wheights usind TD(lambda) formula -> alpha * delta (computed above) * E(s,a)
        self.weights[action] += self.alpha * delta * self.traces[action]
        
    def update_alpha_epsilon(self): # do not touch
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)
        self.alpha = self.alpha*self.alpha_decay
        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None): # do not touch
        if epsilon is None: epsilon = self.epsilon
        if random.random()<epsilon:
            return self.env.action_space.sample()
        return self.policy(state)
       
        
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        print(f'ep | eval | epsilon | alpha')
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)
                
                s = s_prime
                
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
