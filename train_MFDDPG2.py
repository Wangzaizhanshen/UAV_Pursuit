"""
用DDPG训练无人机聚集路径规划任务
"""
import random

from UAV_Pursuit2 import UAVPursuit
from MF_DDPG import DDPG
from tensorboardX import SummaryWriter
import numpy as np


def main():
    writer = SummaryWriter(log_dir='./DDPG_logs/3')
    writer_step = 0

    env = UAVPursuit()
    agent = DDPG(state_dim=16, action_dim=2, num=env.uav_num, use_mf=True, act_mf=0, gamma=0.98, tau=0.005,
                 max_size=100000, bact_size=128, learning_rate=0.005, epsilon=0.3)
    t_agent = DDPG(state_dim=16, action_dim=2, num=env.target_num, use_mf=True, act_mf=0, gamma=0.98, tau=0.005,
                   max_size=100000, bact_size=128, learning_rate=0.005, epsilon=0.3)
    action, action_ = [], []
    act, act_ = [], []
    t_act, t_act_ = [], []
    t_action, t_action_ = [], []
    for i in range(env.uav_num):
        action.append([])
        action_.append([])
        act.append([])
        act_.append([])
    for i in range(env.target_num):
        t_action.append([])
        t_action_.append([])
        t_act.append([])
        t_act_.append([])
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
            # epsilon参数取值
            if episode < 0.2 * env.episode_max:
                agent.epsilon = 0.8
                t_agent.epsilon = 0.8
            elif episode < 0.4 * env.episode_max:
                agent.epsilon = 0.6
                t_agent.epsilon = 0.6
            elif episode < 0.6 * env.episode_max:
                agent.epsilon = 0.4
                t_agent.epsilon = 0.4
            elif episode < 0.7 * env.episode_max:
                agent.epsilon = 0.2
                t_agent.epsilon = 0.2
            elif episode < 0.8 * env.episode_max:
                agent.epsilon = 0.1
                t_agent.epsilon = 0.1
            else:
                agent.epsilon = 0.
                t_agent.epsilon = 0.
            '''
            输入env.state维度要改
            '''
            for i in range(env.target_num):
                t_action[i], t_action_[i] = t_agent.choose_action(env.t_state[i], train=True)
                t_act[i], t_act_[i] = agent.choose_action(env.t_state_[i], train=False)
            for i in range(env.uav_num):
                action[i], action_[i] = agent.choose_action(env.state[i], train=True)
                act[i], act_[i] = agent.choose_action(env.state_[i], train=False)
            # 进行一步仿真
            state, reward, done, state_, t_state, t_reward, t_done, t_state_ = env.step(action, t_action)
            # env.render()
            # 存储无人机交互信息
            '''
            remember中交互信息维度要改
            '''
            act_mf, act_mf_ = [], []
            for i in range(env.uav_num):
                temp_n = 0  # 在己方无人机数量
                act_mf.append([0])
                act_mf_.append([0])
                # 无人机成功或失败时，done=Ture，但多无人机时，若其他无人机未结束，此无人机会一直停留在此处，此时该无人机的交互信息不再采纳
                if env.state_[i][0] == env.state[i][0] and env.state_[i][1] == env.state[i][1]:
                    continue
                # 计算平均动作
                for j in range(env.uav_num):
                    if i == j:
                        continue
                    # 超出范围的无人机不计算
                    if j not in env.prob_indices[i]:
                        act_mf[i] += np.zeros([2])
                        act_mf_[i] += np.zeros([2])
                        continue
                    act_mf[i] += action[j]
                    act_mf_[i] += act_[j]
                    temp_n += 1
                act_mf[i] = act_mf[i] / (temp_n or 1)
                act_mf_[i] = act_mf_[i] / (temp_n or 1)
                agent.remember(env.state[i], env.action[i], env.reward[i], env.done[i], env.state_[i], act_mf[i],
                               act_mf_[i])
            t_act_mf, t_act_mf_ = [], []
            # 存储逃逸无人机交互信息
            for i in range(env.target_num):
                temp_n = 0  # 在无人机探测范围内的己方无人机数量
                t_act_mf.append([0])
                t_act_mf_.append([0])
                # 无人机成功或失败时，done=Ture，但多无人机时，若其他无人机未结束，此无人机会一直停留在此处，此时该无人机的交互信息不再采纳
                if env.t_state_[i][0] == env.t_state[i][0] and env.t_state_[i][1] == env.t_state[i][1]:
                    continue
                # 计算平均动作
                for j in range(env.target_num):
                    # 不计算本架无人机
                    if i == j:
                        continue
                    # 超出探测范围的无人机不计算
                    if j not in env.prob_tar_indices[i]:
                        t_act_mf[i] += np.zeros([2])
                        t_act_mf_[i] += np.zeros([2])
                        continue
                    t_act_mf[i] += t_action_[j]
                    t_act_mf_[i] += t_act_[j]
                    temp_n += 1
                t_act_mf[i] = t_act_mf[i] / (temp_n or 1)  # (env.uav_num - 1)
                t_act_mf_[i] = t_act_mf_[i] / (temp_n or 1)  # (env.uav_num - 1)
                t_agent.remember(env.t_state[i], env.t_action[i], env.t_reward[i], env.t_done[i], env.t_state_[i],
                                 t_act_mf[i], t_act_mf_[i])
            act_mf.clear()
            act_mf_.clear()
            t_act_mf.clear()
            t_act_mf_.clear()
            # 进行训练
            agent.learn()
            t_agent.learn()

            uav_reward += env.reward
            tar_reward += env.t_reward
            # 记录数据
        print('第', episode + 1, '轮追捕无人机奖励值为:', uav_reward, '逃逸无人机奖励值为:', tar_reward)
        print('critic_loss:', agent.critic_loss)
        print('actor_loss:', agent.actor_loss)
        for i in range(env.uav_num):
            writer.add_scalar('reward[%d]' % i, uav_reward[i], writer_step)
        writer.add_scalar('critic_loss', agent.critic_loss.detach().cpu().numpy().tolist(), writer_step)
        writer.add_scalar('actor_loss', agent.actor_loss.detach().cpu().numpy().tolist(), writer_step)
        writer_step += 1

        # 保存训练模型
        if (episode + 1) % 100 == 0:
            agent.save_model(episode + 1, 3)
            t_agent.save_tar_model(episode + 1, 3)

    # 训练结束
    writer.close()


if __name__ == '__main__':
    main()
