"""
用DDPG训练无人机聚集路径规划任务
"""
# 导入随机模块
import random

# 导入无人机追击环境类
from UAV_Pursuit2 import UAVPursuit
# 导入自定义的GL_MFDDPG智能体类
from GL_MFDDPG import GL_MFDDPG
# 导入自定义的DDPG智能体类
from MF_DDPG import DDPG
# 导入tensorboardX用于可视化训练过程
from tensorboardX import SummaryWriter
# 导入numpy用于数值计算
import numpy as np


def main():
    # 创建SummaryWriter对象，用于记录训练日志，日志保存路径为'./DDPG_logs/4'
    writer = SummaryWriter(log_dir='./DDPG_logs/4')
    # 初始化日志记录步数
    writer_step = 0

    # 创建无人机追击环境实例
    env = UAVPursuit()
    # 创建GL_MFDDPG智能体实例（用于追捕无人机）
    # 参数说明：状态维度16，动作维度2，数量为环境中无人机数量，使用均值场，act_mf为0，
    # 折扣因子gamma=0.98，软更新系数tau=0.005，经验回放池最大容量100000，
    # 批处理大小128，学习率0.0005，探索率epsilon=0.1
    agent = GL_MFDDPG(state_dim=16, action_dim=2, num=env.uav_num, use_mf=True, act_mf=0, gamma=0.98, tau=0.005,
                      max_size=100000, bact_size=128, learning_rate=0.0005, epsilon=0.1)
    # 创建DDPG智能体实例（用于目标/逃逸无人机）
    # 参数说明类似上面的agent，学习率为0.005
    t_agent = DDPG(state_dim=16, action_dim=2, num=env.target_num, use_mf=True, act_mf=0, gamma=0.98, tau=0.005,
                   max_size=100000, bact_size=128, learning_rate=0.005, epsilon=0.1)

    # 初始化存储动作的列表（用于追捕无人机）
    action, action_ = [], []
    act, act_ = [], []
    # 初始化存储动作的列表（用于目标/逃逸无人机）
    t_act, t_act_ = [], []
    t_action, t_action_ = [], []

    # 为每个追捕无人机初始化动作存储列表
    for i in range(env.uav_num):
        action.append([])
        action_.append([])
        act.append([])
        act_.append([])
    # 为每个目标/逃逸无人机初始化动作存储列表
    for i in range(env.target_num):
        t_action.append([])
        t_action_.append([])
        t_act.append([])
        t_act_.append([])

    # 开始训练循环，迭代次数为环境设定的最大回合数
    for episode in range(env.episode_max):
        # 初始化本回合中每个追捕无人机和目标无人机的总奖励
        uav_reward = np.zeros([env.uav_num, ])
        tar_reward = np.zeros([env.target_num])

        # 重置环境，开始新的一局
        env.reset()
        # 保存初始状态（用于后续构建LSTM输入）
        state0 = env.state
        state1 = env.state
        state0_ = env.state
        state1_ = env.state

        # 开始每一步的交互循环，步数上限为环境设定的最大回合步数
        for step in range(env.round_max):
            # 如果所有目标都被捕获或所有追捕无人机都完成任务，则跳出循环
            if env.t_done.sum() == env.target_num or env.done.sum() == env.uav_num:
                break

            # 根据当前回合数调整探索率epsilon（线性衰减策略）
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
            输入env.state维度要改（注释：可能是预留的维度调整提示）
            '''
            # 为每个目标/逃逸无人机选择动作
            for i in range(env.target_num):
                t_action[i], t_action_[i] = t_agent.choose_action(env.t_state[i], train=True)
            # 提前计算目标/逃逸无人机下一步的动作（用于均值场计算）
            for i in range(env.target_num):
                t_act[i], t_act_[i] = t_agent.choose_action(env.t_state_[i], train=False)

            # 初始化存储所有追捕无人机的动作和状态的数组
            uav_act = np.zeros([env.uav_num, 2])  # 存放所有无人机的动作
            uav_act_ = np.zeros([env.uav_num, 2])  # 存放所有无人机的下一步动作
            uav_s = np.zeros([env.uav_num, 16])  # 存放所有无人机的状态
            uav_s_ = np.zeros([env.uav_num, 16])  # 存放所有无人机的下一步状态

            # 计算状态变化率（用于构建历史状态序列）
            delta_x = (env.state[:, :] - state0[:, :]) / 8
            delta_x_ = (env.state_[:, :] - state0_[:, :]) / 8

            # 为每个追捕无人机选择动作
            for i in range(env.uav_num):
                # 构建LSTM网络的输入（时间步长为9的状态序列）
                state_LSTM = np.stack((state0[i],
                                       (state0 + delta_x)[i],
                                       (state0 + delta_x * 2)[i],
                                       (state0 + delta_x * 3)[i],
                                       (state0 + delta_x * 4)[i],
                                       (state0 + delta_x * 5)[i],
                                       (state0 + delta_x * 6)[i],
                                       (state0 + delta_x * 7)[i],
                                       env.state[i]), axis=0)
                # 选择当前状态下的动作
                action[i], action_[i] = agent.choose_action(state_LSTM, train=True)
                uav_act[i] = action[i]
                uav_s[i] = env.state[i]

                # 构建下一步状态的LSTM输入
                state_LSTM_ = np.stack((state0_[i],
                                        (state0_ + delta_x_ * 1)[i],
                                        (state0_ + delta_x_ * 2)[i],
                                        (state0_ + delta_x_ * 3)[i],
                                        (state0_ + delta_x_ * 4)[i],
                                        (state0_ + delta_x_ * 5)[i],
                                        (state0_ + delta_x_ * 6)[i],
                                        (state0_ + delta_x_ * 7)[i],
                                        env.state_[i]), axis=0)
                # 选择下一步状态下的动作（用于均值场计算）
                act[i], act_[i] = agent.choose_action(state_LSTM_, train=False)
                uav_act_[i] = act_[i]
                uav_s_[i] = env.state_[i]

            # 执行一步环境交互，获取新状态、奖励等信息
            state, reward, done, state_, t_state, t_reward, t_done, t_state_ = env.step(action, t_action)
            # env.render()  # 可选：渲染环境

            # 初始化用于存储均值场动作的列表
            act_mf, act_mf_ = [], []
            # 初始化无人机邻接矩阵（1表示是邻居，0表示不是）
            uav_adj = np.zeros([env.uav_num, env.uav_num])

            # 计算邻接矩阵（判断无人机之间是否为邻居）
            for i in range(env.uav_num):
                # 如果无人机位置未发生变化（已完成任务或失败），则不参与交互计算
                if env.state_[i][0] == env.state[i][0] and env.state_[i][1] == env.state[i][1]:
                    uav_act[i] = np.zeros([2])
                    uav_act_[i] = np.zeros([2])
                    uav_adj[i, :] = np.zeros([env.uav_num])
                    continue
                for j in range(env.uav_num):
                    if i == j:
                        continue
                    # 超出范围的无人机不计算
                    if j not in env.prob_indices[i]:
                        continue
                    uav_adj[i][j] = 1

            '''
            remember中交互信息维度要改（注释：可能是预留的维度调整提示）
            '''
            # 重新初始化均值场动作列表
            act_mf, act_mf_ = [], []
            # 计算另一组状态变化率（用于构建另一组历史状态序列）
            delta_y = (env.state[:, :] - state1[:, :]) / 8
            delta_y_ = (env.state_[:, :] - state1_[:, :]) / 8

            # 计算追捕无人机的均值场动作并存储经验
            for i in range(env.uav_num):
                temp_n = 0  # 邻居无人机数量
                act_mf.append([0])
                act_mf_.append([0])
                # 如果无人机位置未变化，跳过
                if env.state_[i][0] == env.state[i][0] and env.state_[i][1] == env.state[i][1]:
                    continue
                # 计算邻居无人机的平均动作（均值场）
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
                # 计算平均动作（避免除零）
                act_mf[i] = act_mf[i] / (temp_n or 1)
                act_mf_[i] = act_mf_[i] / (temp_n or 1)

                # 对state的维度进行更改，适配GRU网络的输入，选取时间步长为3
                # 构建用于存储经验的LSTM状态序列
                state_LSTM = np.stack((state1[i],
                                       (state1 + delta_y)[i],
                                       (state1 + delta_y * 2)[i],
                                       (state1 + delta_y * 3)[i],
                                       (state1 + delta_y * 4)[i],
                                       (state1 + delta_y * 5)[i],
                                       (state1 + delta_y * 6)[i],
                                       (state1 + delta_y * 7)[i],
                                       env.state[i]), axis=0)
                state_LSTM_ = np.stack((state1_[i],
                                        (state1_ + delta_y_ * 1)[i],
                                        (state1_ + delta_y_ * 2)[i],
                                        (state1_ + delta_y_ * 3)[i],
                                        (state1_ + delta_y_ * 4)[i],
                                        (state1_ + delta_y_ * 5)[i],
                                        (state1_ + delta_y_ * 6)[i],
                                        (state1_ + delta_y_ * 7)[i],
                                        env.state_[i]), axis=0)
                # 将经验存储到回放池
                agent.remember(state_LSTM, env.action[i], env.reward[i], env.done[i], state_LSTM_,
                               uav_act, uav_act_, uav_adj, uav_s, uav_s_)

            # 初始化目标/逃逸无人机的均值场动作列表
            t_act_mf, t_act_mf_ = [], []
            # 计算目标/逃逸无人机的均值场动作并存储经验
            for i in range(env.target_num):
                temp_n = 0  # 邻居目标无人机数量
                t_act_mf.append([0])
                t_act_mf_.append([0])
                # 如果目标无人机位置未变化，跳过
                if env.t_state_[i][0] == env.t_state[i][0] and env.t_state_[i][1] == env.t_state[i][1]:
                    continue
                # 计算邻居目标无人机的平均动作（均值场）
                for j in range(env.target_num):
                    if i == j:
                        continue
                    # 超出探测范围的不计算
                    if j not in env.prob_tar_indices[i]:
                        t_act_mf[i] += np.zeros([2])
                        t_act_mf_[i] += np.zeros([2])
                        continue
                    t_act_mf[i] += t_action_[j]
                    t_act_mf_[i] += t_act_[j]
                    temp_n += 1
                # 计算平均动作（避免除零）
                t_act_mf[i] = t_act_mf[i] / (temp_n or 1)
                t_act_mf_[i] = t_act_mf_[i] / (temp_n or 1)
                # 将经验存储到回放池
                t_agent.remember(env.t_state[i], env.t_action[i], env.t_reward[i], env.t_done[i], env.t_state_[i],
                                 t_act_mf[i], t_act_mf_[i])

            # 清空均值场动作列表（释放内存）
            act_mf.clear()
            act_mf_.clear()
            t_act_mf.clear()
            t_act_mf_.clear()

            # 智能体进行学习（从回放池中采样并更新网络）
            agent.learn()
            t_agent.learn()

            # 累加本回合的奖励
            uav_reward += env.reward
            tar_reward += env.t_reward

        # 打印本回合的训练信息
        print('第', episode + 1, '轮追捕无人机奖励值为:', uav_reward, '逃逸无人机奖励值为:', tar_reward)
        print('critic_loss:', agent.critic_loss)
        print('actor_loss:', agent.actor_loss)

        # 将奖励和损失记录到tensorboard
        for i in range(env.uav_num):
            writer.add_scalar('reward[%d]' % i, uav_reward[i], writer_step)
        writer.add_scalar('critic_loss', agent.critic_loss.detach().cpu().numpy().tolist(), writer_step)
        writer.add_scalar('actor_loss', agent.actor_loss.detach().cpu().numpy().tolist(), writer_step)
        writer_step += 1

        # 每100回合保存一次模型
        if (episode + 1) % 100 == 0:
            agent.save_model(episode + 1, 4)
            t_agent.save_tar_model(episode + 1, 4)

    # 训练结束，关闭日志记录
    writer.close()


# 当脚本直接运行时，执行main函数
if __name__ == '__main__':
    main()
