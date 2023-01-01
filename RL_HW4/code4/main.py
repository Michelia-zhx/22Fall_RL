from arguments import get_args
from algo import *
import os
import numpy as np
import time
from itertools import product
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from env import Make_Env
from gym_minigrid.wrappers import *
import scipy.misc

args = get_args()
num_updates = int(args.num_frames // args.num_steps)
params = {
        'n': [100, 300, 500],
        'start_planning': [5, 10, 30],
        'h': [10, 30, 50],
        'm': [100, 300, 500]
    }

# environment initial
envs = Make_Env(env_mode=2)
action_shape = envs.action_shape
observation_shape = envs.state_shape
print(action_shape, observation_shape)

def plot(record, info, n, start_planning, h, m, t):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record['steps'], record['mean'],
            color='blue', label='reward')
    ax.fill_between(record['steps'], record['min'], record['max'],
                    color='blue', alpha=0.2)
    ax.set_xlabel('number of steps')
    ax.set_ylabel('Average score per episode')
    ax.set_title("Dyna-Q with n = {}, h = {}, m = {}, start = {}".format(n, h, m, start_planning))
    
    # os.makedirs(t + '-{}'.format(info), exist_ok=True)
    fig.savefig('{}-{}-{}-{}-performance.png'.format(n, h, m, start_planning))
    plt.close()


def execute(n, start_planning, h, m):
    t = str(time.time())
    start = time.time()
    param_info = "{}-{}-{}-{}".format(start_planning, n, h, m)
    record = {'steps': [0],
              'max': [0],
              'mean': [0],
              'min': [0]}
    agent = QAgent(envs)
    # dynamics_model = DynaModel()
    dynamics_model = NetworkModel(8, 8, policy=agent, args=args)

    # start to train your agent
    for i in range(num_updates):
        # an example of interacting with the environment
        obs = envs.reset()
        obs = obs.astype(int)
        for step in range(args.num_steps):
            # Sample actions with epsilon greedy policy
            epsilon = max(0.1, 1 - i / (num_updates))
            if np.random.rand() < epsilon:
                action = envs.action_sample()
            else:
                action = agent.select_action(obs)

            # interact with the environment
            obs_next, reward, done, info = envs.step(action)
            obs_next = obs_next.astype(int)
            agent.update(obs, action, reward, obs_next, done)
            # add your Q-learning algorithm
            dynamics_model.store_transition(obs, action, reward, obs_next)
            obs = obs_next

            if done:
                obs = envs.reset()

        for _ in range(m):
            dynamics_model.train_transition(128)

        if i > start_planning:
            if isinstance(dynamics_model, DynaModel):
                # print("running dyna")
                for _ in range(n):
                    result = dynamics_model.sample_pair()
                    assert result is not None
                    s, a = result
                    s_ = dynamics_model.predict(s, a)
                    r = envs.R(s, a, s_)
                    done = envs.D(s, a, s_)
                    agent.update(s, a, r, s_, done)
            else:
                for _ in range(n):
                    s = dynamics_model.sample_state()
                    for _ in range(h):
                        if np.random.rand() < epsilon:
                            a = envs.action_sample()
                        else:
                            a = agent.select_action(s)
                        s_ = dynamics_model.predict(s, a)
                        r = envs.R(s, a, s_)
                        done = envs.D(s, a, s_)
                        agent.update(s, a, r, s_, done)
                        s = s_
                        if done:
                            break

        if (i + 1) % (args.log_interval) == 0:
            total_num_steps = (i + 1) * args.num_steps
            obs = envs.reset()
            obs = obs.astype(int)
            reward_episode_set = []
            reward_episode = 0.
            for step in range(args.test_steps):
                action = agent.select_action(obs)
                obs_next, reward, done, info = envs.step(action)
                reward_episode += reward
                obs = obs_next
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0.
                    obs = envs.reset()

            end = time.time()
            print("TIME {} Updates {}, num timesteps {}, FPS {} \n avrage/min/max reward {:.1f}/{:.1f}/{:.1f}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                    i, total_num_steps, int(total_num_steps / (end - start)),
                    np.mean(reward_episode_set),
                    np.min(reward_episode_set),
                    np.max(reward_episode_set)))
            record['steps'].append(total_num_steps)
            record['mean'].append(np.mean(reward_episode_set))
            record['max'].append(np.max(reward_episode_set))
            record['min'].append(np.min(reward_episode_set))
            plot(record, args.info, n, start_planning, h, m, t)

def main():
    for vals in product(*params.values()):
        n = vals[0]
        h = vals[2]
        m = vals[3]
        start_planning = vals[1]
        figure_name = '{}-{}-{}-{}-performance.png'.format(n, h, m, start_planning)
        if os.path.exists(figure_name):
            print(figure_name + "has existed.")
            continue
        execute(**dict(zip(params, vals)))

if __name__ == "__main__":
    main()
