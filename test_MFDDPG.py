"""
用DDPG训练无人机聚集路径规划任务
"""
import random

from UAV_Pursuit import UAVPursuit
from MF_DDPG import DDPG
import numpy as np


def main():
    env = UAVPursuit()
    agent = DDPG(state_dim=16, action_dim=2, num=env.uav_num, use_mf=True, act_mf=0, gamma=0.98, tau=0.005,
                 max_size=100000, bact_size=128, learning_rate=0.0005, epsilon=0.3)
    t_agent = DDPG(state_dim=16, action_dim=2, num=env.target_num, use_mf=True, act_mf=0, gamma=0.98, tau=0.005,
                   max_size=100000, bact_size=128, learning_rate=0.002, epsilon=0.1)
    # 加载模型参数
    agent.load_models(100, 1)
    t_agent.load_models(100, 1)
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
            for i in range(env.uav_num):
                action[i], action_[i] = agent.choose_action(env.state[i], train=False)
            # print(len(action[0]))
            # 进行一步仿真
            state, reward, done, state_, t_state, t_reward, t_done, t_state_ = env.step(action, t_action)
            env.render()

            uav_reward += env.reward
            tar_reward += env.t_reward
            # 记录数据
        print('第', episode + 1, '轮追捕无人机奖励值为:', uav_reward, '逃逸无人机奖励值为:', tar_reward)
        print('critic_loss:', agent.critic_loss)
        print('actor_loss:', agent.actor_loss)
        # env.draw(1)


if __name__ == '__main__':
    main()
