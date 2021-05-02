import parser
import time
import os
import numpy as np
import logz
import ray
import utils
import socket

from gymenv_v2 import make_multiple_env
import wandb
from es.shared_noise import *
from es.policies import MLPRowFeatureAttenttionPolicy, MLPRowFeatureLSTMEmbeddingPolicy
from es.optimizers import Adam, SGD
from es.alg_utils import rollout, rollout_envs, rollout_evaluate, rollout_envs_random
from es.utils import compute_stats


custom_config = {
    "load_dir"        : 'instances/randomip_n15_m15',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(10)),                # take the first 20 instances from the directory
    "timelimit"       : 12,                             # the maximum horizon length is 50
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

def get_policy(policy_params):
    policy_type = policy_params['policy_type']
    if policy_type == 'attention':
        policy_params['hsize'] = 64
        policy_params['numlayers'] = 2
        policy_params['embed'] = 10
        policy_params['rowembed'] = 10
        policy = MLPRowFeatureAttenttionPolicy(policy_params)
    elif policy_type == 'lstmembed':
        policy_params['hsize'] = 64
        policy_params['numlayers'] = 2
        policy_params['embed'] = 10
        policy_params['rowembed'] = 10
        policy = MLPRowFeatureLSTMEmbeddingPolicy(policy_params)
    else:
        raise NotImplementedError
    
    if policy_params['reload']:
        print(f'Reload params from {policy_params["reload_dir_path"]}')
        old_weights = np.load(policy_params['reload_dir_path'])
        old_weights.allow_pickle = True
        policy.update_weights(old_weights['arr_0'][0]) # NOTE: saved weights is actually weights, stats, stats
    return policy

@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 config='',
                 policy_params = None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02):

        # initialize OpenAI environment for each worker
        self.env = make_multiple_env(**config, seed=env_seed)
        
        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        
        self.policy_params = policy_params
        self.policy = get_policy(policy_params)
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length
        
    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        return self.policy.get_weights_plus_stats()
   
    def rollout(self, shift = 0., rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break
            
        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """
        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps = self.rollout(shift = 0., rollout_length = self.env.timelimit)
                rollout_rewards.append(reward)
                
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps  = self.rollout(shift = shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps = self.rollout(shift = shift) 
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps" : steps}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

class ARSLearner(object):
    """ 
    Object class implementing the ARS algorithm.
    """
    def __init__(self, config,
                 policy_type = None,
                 ob_filter = None,
                 num_workers=2, 
                 num_deltas=320, 
                 deltas_used=320,
                 delta_std=0.02, 
                 logdir=None, 
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123,
                 reload=False,
                 reload_dir_path='',
                 tag = ''):
        
        logdir += os.path.join(logdir, ' | '.join([f'{str(param)}={str(params[param])}' for param in params if param != 'reload_dir_path']))
        logz.configure_output_dir(logdir)
        logz.save_params(params)
        
        assert config in ['custom_config', 'easy_config', 'hard_config'] 
        env = make_multiple_env(**eval(config), seed = 0) 
        numvars = env.reset()[0].shape[0] # This is to set the policy param
        max_gap = {i: single_env.env.max_gap()[0] for i, single_env in enumerate(env.envs)}
        print(f'Max gap : {max_gap}')

        self.timesteps = 0
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')
       
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
        print('Created deltas table.')

        # initialize policy 
        policy_params = {'numvars':numvars,
                        'ob_filter':ob_filter,
                        'policy_type': policy_type,
                        'reload': reload,
                        'reload_dir_path':reload_dir_path}
        
        self.policy = get_policy(policy_params)   
        self.w_policy = self.policy.get_weights()
            
        # initialize optimization algorithm
        self.optimizer = SGD(self.w_policy, self.step_size)        
        print("Initialization of ARS complete.")

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      config=eval(config),
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts
            
        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)

        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results 
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], [] 

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]
        
        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat
        

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        
        g_hat = self.aggregate_rollouts()                    
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        logz.save_policy(self.w_policy)
        return

    def train(self, num_iter):
        
        wandb.login()
        run=wandb.init(project="project-local", entity="ieor-4575", tags=[f"training-easy"])

        rewards_record = []

        start = time.time()
        for i in range(num_iter):
            t1 = time.time()
            self.train_step()
            t2 = time.time()
            print('total training time: ', t2 - t1)           
            print('iter ', i,' done')

            # record statistics every 10 iterations
            if ((i + 1) % 1 == 0):
                t3 = time.time()
                rewards = self.aggregate_rollouts(num_rollouts = 5, evaluate = True)
                t4 = time.time()
                print('total evaluation time: ', t4- t3)
                if ((i + 1) % 10 == 0):
                    w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                    np.savez(self.logdir + f"/lin_policy_plus_{i}", w)
                
                print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("timesteps", self.timesteps)
                logz.dump_tabular()

                rewards_record.append(np.mean(rewards))
                fixedWindow=10
                movingAverage=0
                if len(rewards_record) >= fixedWindow:
                    movingAverage=np.mean(rewards_record[len(rewards_record)-fixedWindow:len(rewards_record)-1])
                wandb.log({ "Training reward" : rewards_record[-1], 
                            "movingAverage" : movingAverage, 
                            "AverageReward": np.mean(rewards), 
                            'StdRewards': np.std(rewards),  
                            'MaxRewardRollout': np.max(rewards) , 
                            'MinRewardRollout': np.min(rewards)})
                
            t1 = time.time()
            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)
         
            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)            
            t2 = time.time()
            print('Time to sync statistics:', t2 - t1)
                        
        return 

def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ARS = ARSLearner(config = params['config'],
                     policy_type = params['policy_type'],
                     ob_filter = params['filter'],
                     num_workers=params['n_workers'], 
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'], 
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed = params['seed'],
                     reload = params['reload'],
                     reload_dir_path = params['reload_dir_path'],
                     tag = params['tag'])
    
    ARS.train(params['n_iter'])

    return 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='easy_config')
    parser.add_argument('--policy_type', '-pt', type=str, default='linear') # 'linear', 'attention', 'mlp'
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-nw', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--dir_path', type=str, default='data')
    parser.add_argument('--reload', default=False, action='store_true')
    parser.add_argument('--reload_dir_path', type=str, default='data')
    parser.add_argument('--filter', type=str, default='MeanStdFilter') #MeanStdFilter' for v2, 'NoFilter' for v1
    parser.add_argument('--tag', type=str, default='Main') 

    local_ip = socket.gethostbyname(socket.gethostname())

    ray.init(local_mode=False, address='auto', _redis_password='5241590000000000')
    args = parser.parse_args()
    params = vars(args)
    run_ars(params)

