# define neural net \pi_\phi(s) as a class
import torch
import numpy as np
import random
import logging

torch.set_default_dtype(torch.float64)

class Policy(object):
    def __init__(self):
        self.is_clone = False
        
    def get_weights(self):
        return [layer.weight.data if hasattr(layer, 'weight') else None for layer in self.model ]

    def update_weights(self, new_weights):
        with torch.no_grad():
            for i,layer in enumerate(self.model):
                if hasattr(layer, 'weight'):
                    layer.weight.data = new_weights[i]
        # for v,v_ in zip(new_weights, self.model.parameters()):
        #     v_.data.copy_(v.data)
        return None

    def clone(self):
        pass
        
    def compute_prob(self, states, actions):
        """
        Compute probability for each action
        """
        self.mnode
        pass

    def rollout(self, env):
        obs = env.reset()
        done = False
        rews =[]
        i = 0
        actions = []
        while not done:
            prob = self.compute_prob(obs)
            prob = prob.detach().numpy()
            #prob /= np.sum(prob)
            action = np.asscalar(np.random.choice(prob.shape[0], p=prob.flatten(), size=1)) #choose according distribution prob
            #action = np.random.randint(len(obs[-1].flatten()))
            actions.append(action)
            try:
                newobs, reward, done, info = env.step(action)
            except:
                print()
            rews.append(reward)
            obs = newobs
            if self.is_clone is False:
                print(f'Rolling out step: {i}, max: {np.max(prob)}, min: {np.min(prob)}, average: {np.mean(prob)}')
            i += 1
        if self.is_clone is False:
            print(f'Clone: {self.is_clone}, Env Index: {env.env_index}, last rolling out step: {i}, max: {np.max(prob)}, min: {np.min(prob)}, average: {np.mean(prob)}')
        return rews, actions, env.env_index


class SimplePolicy(Policy):
    def __init__(self, varsize):
        Policy.__init__(self)
        self.varsize = varsize
        self.model = torch.nn.Sequential(
                            torch.nn.Linear(varsize+1,16),
                            torch.nn.ReLU(),
                            #torch.nn.BatchNorm1d(16),
                            torch.nn.Linear(16, 16),
                            torch.nn.ReLU(),
                            #torch.nn.BatchNorm1d(16),
                            torch.nn.Linear(16, 8)
                        )

    def compute_prob(self, obs):
        """
        Compute probability for each action
        """
        A, b, c0, cuts_a, cuts_b = obs
        if cuts_b.shape[0] == 1:
            return 1
        state = torch.Tensor(np.concatenate((A,b.reshape(-1,1)), axis = 1)).type(torch.float64)
        action = torch.Tensor(np.concatenate((cuts_a,cuts_b.reshape(-1,1)), axis = 1)).type(torch.float64)
        with torch.no_grad():    
            g = self.model(state)
            h = self.model(action)
        score = torch.mean(g @ h.T, axis = 0)
        # print(f'max score: {torch.max(score)}, min score: {torch.min(score)}, mean score: {torch.mean(score)}')
        #score = score/torch.sum(score)
        score = torch.softmax(score, dim = 0)
        if np.isnan(score[0]):
            print(score)
        return score
    
    def clone(self):
        actor_clone = SimplePolicy(self.varsize)
        for v,v_ in zip(self.model.parameters(), actor_clone.model.parameters()):
            v_.data.copy_(v.data)
        #actor_clone.model = actor_clone.model.float()
        actor_clone.is_clone = True
        return actor_clone

class SimplePolicy2(Policy):
    def __init__(self, varsize):
        Policy.__init__(self)
        self.varsize = varsize
        self.model = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(varsize+1),
                            torch.nn.Linear(varsize+1,64),
                            torch.nn.ReLU(),
                            torch.nn.Linear(64,32),
                            torch.nn.ReLU(),
                            torch.nn.Linear(32, 16),
                            torch.nn.ReLU(),
                            torch.nn.Linear(16, 8)
                        )

    def compute_prob(self, obs):
        """
        Compute probability for each action
        """
        A, b, c0, cuts_a, cuts_b = obs
        if cuts_b.shape[0] == 1:
            return 1
        state = torch.Tensor(np.concatenate((A,b.reshape(-1,1)), axis = 1)).type(torch.float64)
        action = torch.Tensor(np.concatenate((cuts_a,cuts_b.reshape(-1,1)), axis = 1)).type(torch.float64)
        with torch.no_grad():    
            g = self.model(state)
            h = self.model(action)
        score = torch.mean(g @ h.T, axis = 0)
        # print(f'max score: {torch.max(score)}, min score: {torch.min(score)}, mean score: {torch.mean(score)}')
        #score = score/torch.sum(score)
        score = torch.softmax(score, dim = 0)
        if np.isnan(score[0]):
            print(score)
        return score
    
    def clone(self):
        actor_clone = SimplePolicy2(self.varsize)
        for v,v_ in zip(self.model.parameters(), actor_clone.model.parameters()):
            v_.data.copy_(v.data)
        #actor_clone.model = actor_clone.model.float()
        actor_clone.is_clone = True
        return actor_clone
