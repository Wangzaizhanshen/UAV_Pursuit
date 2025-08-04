"""
用DDPG训练无人机聚集路径规划任务
"""
import random

from UAV_Pursuit2 import UAVPursuit
from GL_MFDDPG import GL_MFDDPG
from MF_DDPG import DDPG
import numpy as np


def main():
    env = UAVPursuit()
    agent = GL_MFDDPG(state_dim=16, action_dim=2, num=env.uav_num, use_mf=True, act_mf=0, gamma=0.98, tau=0.005,
                      max_size=100000, bact_size=128, learning_rate=0.0005, epsilon=0.3)
    t_agent = DDPG(state_dim=16, action_dim=2, num=env.target_num, use_mf=True, act_mf=0, gamma=0.98, tau=0.005,
                   max_size=100000, bact_size=128, learning_rate=0.002, epsilon=0.1)
    # 加载模型参数
    agent.load_models(900, 4)
    t_agent.load_tar_models(900, 4)
    action, action_ = [], []
    t_action, t_action_ = [], []
    for i in range(env.uav_num):
        action.append([])
        action_.append([])
    for i in range(env.target_num):
        t_action.append([])
        t_action_.append([])
    for episode in range(env.episode_max):
        uav_reward = np.zeros([env.uav_num, ])
        tar_reward = np.zeros([env.target_num])

        # 开始一局游戏
        env.reset()
        # 需要对输入状态进行处理,保存初始状态
        state0 = env.state
        state1 = env.state
        state0_ = env.state
        state1_ = env.state
        for step in range(env.round_max):
            # 结束跳出循环
            if env.t_done.sum() == env.target_num or env.done.sum() == env.uav_num:
                break

            # 获取动作
            '''
            输入env.state维度要改
            '''
            for i in range(env.target_num):
                t_action[i], t_action_[i] = t_agent.choose_action(env.t_state[i], train=False)

            # 提前获得一下状态下的所有无人机动作
            delta_x = (env.state[:, :] - state0[:, :]) / 8
            for i in range(env.uav_num):
                # 对state的维度进行更改，适配GRU网络的输入，选取时间步长为3
                state_LSTM = np.stack((state0[i],
                                       (state0 + delta_x)[i],
                                       (state0 + delta_x * 2)[i],
                                       (state0 + delta_x * 3)[i],
                                       (state0 + delta_x * 4)[i],
                                       (state0 + delta_x * 5)[i],
                                       (state0 + delta_x * 6)[i],
                                       (state0 + delta_x * 7)[i],
                                       env.state[i]), axis=0)
                action[i], action_[i] = agent.choose_action(state_LSTM, train=False)
            # print(len(action[0]))
            # 进行一步仿真
            state, reward, done, state_, t_state, t_reward, t_done, t_state_ = env.step(action, t_action)
            env.render()

            uav_reward += env.reward
            tar_reward += env.t_reward
        # env.draw(episode)
        env.draw_gif(episode)
        # 记录数据
        print('第', episode + 1, '轮追捕无人机奖励值为:', uav_reward, '逃逸无人机奖励值为:', tar_reward)
        print('critic_loss:', agent.critic_loss)
        print('actor_loss:', agent.actor_loss)



if __name__ == '__main__':
    main()
