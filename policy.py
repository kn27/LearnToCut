# define neural net \pi_\phi(s) as a class
import torch
import numpy as np
import random

class Policy(object):
    def __init__(self):
        pass
        self.model = torch.nn.Sequential(
                    torch.nn.Linear(10,16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 8),
                    torch.nn.ReLU(),
                    torch.nn.Linear(8, 1)
                )

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
        actor_clone = Policy()
        for v,v_ in zip(self.model.parameters(), actor_clone.model.parameters()):
            v_.data.copy_(v.data)
        return actor_clone

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
            if i%10 == 0:
                print(f'Rolling out step: {i}')
            prob = self.compute_prob(np.expand_dims(obs,0))
            prob /= np.sum(prob)
            action = np.asscalar(np.random.choice(actsize, p=prob.flatten(), size=1)) #choose according distribution prob
            #action = np.random.randint(len(obs[-1].flatten()))
            actions.append(action)
            newobs, reward, done, info = env.step(action)
            rews.append(reward)
            obs = newobs
            i += 1
        print(f'Last rolling out step: {i}')
        return rews, actions
