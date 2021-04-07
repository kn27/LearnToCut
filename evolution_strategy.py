from gymenv_v2 import make_multiple_env
import numpy as np
from value import ValueFunction
from policy import Policy   

import wandb
wandb.login()
run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-easy"])
#run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-hard"])
#run=wandb.init(project="finalproject", entity="ieor-4575", tags=["test"])

### TRAINING

# Setup: You may generate your own instances on which you train the cutting agent.
custom_config = {
    "load_dir"        : 'instances/randomip_n60_m60',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(20)),                # take the first 20 instances from the directory
    "timelimit"       : 50,                             # the maximum horizon length is 50
    "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
}

# Easy Setup: Use the following environment settings. We will evaluate your agent with the same easy config below:
easy_config = {
    "load_dir"        : 'instances/train_10_n60_m60',
    "idx_list"        : list(range(10)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}

# Hard Setup: Use the following environment settings. We will evaluate your agent with the same hard config below:
hard_config = {
    "load_dir"        : 'instances/train_100_n60_m60',
    "idx_list"        : list(range(99)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}

if __name__ == "__main__":

    # hyperparameters
    numtrajs = 1  # num of trajecories from the current policy to collect in each iteration
    iterations = 1000  # total num of iterations
    gamma = .99  # discount
    sigma = 0.01

    # create env
    env = make_multiple_env(**easy_config) 

    # initialize networks
    actor = Policy()

    #To record training reward for logging and plotting purposes
    rrecord = []
    
    # main iteration
    for ite in range(iterations): 
        J = np.zeros(N,numtrajs)
        delta = np.zeros(actor.shape)

        for n in range(N):
            # Make a copy of the current network
            actor_clone = actor.clone()
            weights = actor_clone.get_weights()
            epsilons = random.random(shape = weights.shape)
            actor_clone.update_weights(weights + sigma * epsilons)
            J = 0
            for num in range(numtrajs):
                rews = actor_clone.rollout(env)
                J+= np.sum([reward * gamma **i for i,reward in enumerate(rews)])
            delta += J/numtrajs * epsilons / sigma
        
        delta = delta/N
        actor.update_weights(actor.weights + delta)
        rews = actor.rollout()
        rrecord.appenD(np.sum(rews))

        fixedWindow=100
        movingAverage=0
        if len(rrecord) >= fixedWindow:
            movingAverage=np.mean(rrecord[len(rrecord)-fixedWindow:len(rrecord)-1])
        if ite % 50 == 0:
            print(f'Ite {ite}: movingAverage = {movingAverage}')
        #wandb logging
        wandb.log({ "training reward" : rrecord[-1], "training reward moving average" : movingAverage})
        




    