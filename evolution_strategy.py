from gymenv_v2 import make_multiple_env
import numpy as np
from value import ValueFunction
from policy import Policy   
import wandb
import torch
from multiprocessing import Pool
import copy
            
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
    iterations = 100  # total num of iterations
    gamma = .99  # discount
    sigma = 0.01
    N = 3

    # create env
    env = make_multiple_env(**easy_config) 

    # initialize networks
    actor = Policy()

    #To record training reward for logging and plotting purposes
    rrecord = []
    
    # main iteration
    for ite in range(iterations): 
        weights = actor.get_weights() 

        for n in range(N):
            # Make a copy of the current network
            epsilons = [np.random.random(layer.shape) if layer is not None else None for layer in weights]
            actor_clone = actor.clone()
            #assert actor_clone.model[0].weight.data.dtype == torch.float64
            #assert all((list(actor_clone.model.parameters())[0] == list(actor.model.parameters())[0]).flatten()) # this is due to dtype when copying??

            #TODO: This is really weird. Need to look into it.
            #actor.model[0].weight.data is list(actor.model.parameters())[0].data
            #actor.model[0].weight is list(actor.model.parameters())[0]

            #TODO: Anothe weird thing, why does copy the network downgrade the dtype from  float64 down to float32?

            new_weights = [layer + epsilons[i] * sigma if layer is not None else None for i,layer in enumerate(weights)]
            actor_clone.update_weights(new_weights)
            #assert all((list(actor_clone.model.parameters())[0] != list(actor.model.parameters())[0]).flatten())

            #TODO: Multiprocessing output the exact same result ..
            
            # Use multiple processes to roll out faster
            def rollout(combo):
                actor, env, seed = combo
                np.random.seed(seed)
                return actor.rollout(env)

            with Pool(5) as p:
                results = (p.map(rollout,zip([copy.deepcopy(actor_clone) for _ in range(numtrajs)], [copy.deepcopy(env) for _ in range(numtrajs)], list(range(numtrajs)))))
            
            results, actions = list(zip(*results))
            J = [round(np.sum([reward * gamma **i for i,reward in enumerate(result)]),3) for result in results]
            print(f'iter = {ite}, n = {n}, J = {J}')
            J = np.sum(J)

            # J = 0
            # actionss = []
            # for num in range(numtrajs):
            #     print(f'iter = {ite}, n = {n}, traj = {num}')
            #     rews,actions = actor_clone.rollout(env)
            #     actionss.append(actions)
            #     J+= np.sum([reward * gamma **i for i,reward in enumerate(rews)])

            # Update the weights variable which will later be used to update actor
            weights = [layer + J/numtrajs * epsilons[i] / sigma/N if layer is not None else None for i,layer in enumerate(weights)] 
        
        test = list(actor.model.parameters())[0]
        actor.update_weights(weights)
        assert all((list(actor_clone.model.parameters())[0] != test).flatten())

        rews, _ = actor.rollout(env)
        rrecord.append(np.sum(rews))

        fixedWindow=100
        movingAverage=0
        
        if len(rrecord) >= fixedWindow:
            movingAverage=np.mean(rrecord[len(rrecord)-fixedWindow:len(rrecord)-1])
        if ite % 50 == 0:
            print(f'Ite {ite}: movingAverage = {movingAverage}')
        
        wandb.log({ "training reward" : rrecord[-1], "training reward moving average" : movingAverage})
        




    