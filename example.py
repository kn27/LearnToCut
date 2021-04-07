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
    alpha = 1e-2  # learning rate for PG
    beta = 1e-2  # learning rate for baseline
    numtrajs = 1  # num of trajecories from the current policy to collect in each iteration
    iterations = 1000  # total num of iterations
    envname = "CartPole-v0"  # environment name
    gamma = .99  # discount
    #epsilon = 0.2

    # create env
    env = make_multiple_env(**easy_config) 

    # initialize networks
    actor = Policy()
    baseline = ValueFunction()


    #To record training reward for logging and plotting purposes
    rrecord = []
    baseline_losses = []
    actor_losses = []

    # main iteration
    for ite in range(iterations): 
        OBS = []  # observations
        ACTS = []  # actions
        ADS = []  # advantages (to compute policy gradient)
        VAL = []  # Monte carlo value predictions (to compute baseline, and policy gradient)

        for num in range(numtrajs):
            obss = []  # states
            acts = []   # actions
            rews = []  # instant rewards

            obs = env.reset()
            done = False
    
            while not done:
                prob = actor.compute_prob(np.expand_dims(obs,0))
                prob /= np.sum(prob) #normalizing again to account for numerical errors
                if np.random.random() > max(0.20,min(0.9,ite/iterations)):
                    # what is the actsize here? 
                    # actsize
                    action = np.random.randint(actsize)
                else:
                    action = np.asscalar(np.random.choice(actsize, p=prob.flatten(), size=1)) #choose according distribution prob
                
                newobs, reward, done, info = env.step(list(action))
                obss.append(obs)
                acts.append(action)
                rews.append(reward)
                obs = newobs
            
            rrecord.append(np.sum(rews))
            V = [np.sum([gamma **j * rew for j,rew in enumerate(rews[i:])]) for i in range(len(rews))]
            V_ = discounted_rewards(rews, gamma)
            assert np.allclose(V, V_, atol=1e-06)
            
            VAL += V
            OBS += obss
            ACTS += acts
            
        assert len(OBS) == len(VAL)
        baseline_loss = baseline.train(np.array(OBS), np.array(VAL).reshape(-1,1))
        baseline_losses.append(np.asscalar(baseline_loss))
        
        baselines = baseline.compute_values(np.array(OBS))
        ADS = np.array(VAL).reshape(-1,1) - baselines
        
        actor_loss = actor.train(np.array(OBS), np.array(ACTS), ADS)
        actor_losses.append(np.asscalar(actor_loss))
        
        fixedWindow=100
        movingAverage=0
        if len(rrecord) >= fixedWindow:
            movingAverage=np.mean(rrecord[len(rrecord)-fixedWindow:len(rrecord)-1])
        if ite % 50 == 0:
            print(f'Ite {ite}: movingAverage = {movingAverage}')
        #wandb logging
        wandb.log({ "training reward" : rrecord[-1], "training reward moving average" : movingAverage})
        wandb.log({"Training reward" : repisode})





    