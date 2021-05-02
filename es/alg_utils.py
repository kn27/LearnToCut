import numpy as np

def rollout_envs_random(env, policy, num_rollouts, rollout_length, gamma):
    rewards = []
    times = []
    actionss = []
    env_index = []
    for i in range(num_rollouts):
        actions = []
        ob = env.reset()
        env_index.append(env.env_index)
        factor = 1.0
        #ob = env.reset()
        done = False
        t = 0
        rsum = 0
        while not done and t <= rollout_length:
            action = policy.act(ob)
            #print(action)
            ob, r, done, _ = env.step(action)
            rsum += r * factor
            factor *= gamma
            t += 1
            actions.append(action)
        rewards.append(rsum)
        times.append(t)
        actionss.append(actions)
    #print({zip(env_index, rewards)})
    return rewards, times, actionss

def rollout_envs(envs, policy, num_rollouts, rollout_length, gamma):
    rewards = []
    times = []
    actionsss = []
    for env in envs:
        r, t, actionss = rollout(env, policy, num_rollouts, rollout_length, gamma)
        rewards += r
        times += t
        actionsss.append(actionss)
    return rewards, times, actionsss

def rollout_evaluate(envs, policy, num_rollouts, rollout_length, gamma):
    rewards = []
    times = []
    for env in envs:
        for i in range(num_rollouts):
            r, t = rollout(env, policy, 1, rollout_length, gamma)
            rewards += r
            times += t
    return rewards, times

def rollout(env, policy, num_rollouts, rollout_length, gamma):
    rewards = []
    times = []
    actionss = []
    for i in range(num_rollouts):
        actions = []
        ob = env.reset()
        factor = 1.0
        #ob = env.reset()
        done = False
        t = 0
        rsum = 0
        while not done and t <= rollout_length:
            action = policy.act(ob)
            #print(action)
            ob, r, done, _ = env.step(action)
            rsum += r * factor
            factor *= gamma
            t += 1
            actions.append(action)
        rewards.append(rsum)
        times.append(t)
        actionss.append(actions)
    return rewards, times, actions

def randomrollout(env, num_rollouts, rollout_length):
    rewards = []
    times = []
    for i in range(num_rollouts):
        ob = env.reset()
        done = False
        t = 0
        rsum = 0
        while not done and t <= rollout_length:
            action = np.zeros_like(env.action_space.sample())
            ob, r, done = env.step(action)
            rsum += r
            t += 1
        rewards.append(rsum)
        times.append(t)
    return rewards, times

# baseline rollout
def rollout_baseline(env, policy, num_rollouts, rollout_length):
    rewards = []
    times = []
    for i in range(num_rollouts):
        ob, _ = env.reset()
        #ob = env.reset()
        done = False
        t = 0
        rsum = 0
        while not done and t <= rollout_length:
            action = policy.act(ob, env)
            #print(action)
            ob, r, done = env.step(action)
            #ob, r, done, _ = env.step(action)
            rsum += r
            t += 1
        rewards.append(rsum)
        times.append(t)
    return rewards, times

def rollout_baseline_envs(envs, policy, num_rollouts, rollout_length):
    rewards = []
    times = []
    for env in envs:
        r, t = rollout_baseline(env, policy, num_rollouts, rollout_length)
        rewards += r
        times += t
    return rewards, times
