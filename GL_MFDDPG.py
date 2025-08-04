"""
DDPG框架
"""
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from SAGE import GraphSAGE
from torch_geometric.data import Data
from torch_geometric.data import Batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OUActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.1):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)

        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, batch_size, num):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0
        self.num = num

        self.state_memory = np.zeros((self.mem_size, 9, state_dim))  # 采取9个时间步
        self.action_memory = np.zeros((self.mem_size, action_dim))
        self.reward_memory = np.zeros((self.mem_size,))
        self.next_state_memory = np.zeros((self.mem_size, 9, state_dim))
        self.terminal_memory = np.zeros((self.mem_size,), dtype=np.bool_)

        self.act_mf_memory = np.zeros((self.mem_size, num, action_dim))
        self.next_act_mf_memory = np.zeros((self.mem_size, num, action_dim))
        self.uav_adj = np.zeros((self.mem_size, num, num))
        self.uav_s = np.zeros((self.mem_size, num, state_dim))
        self.uav_s_ = np.zeros((self.mem_size, num, state_dim))

    def store_transition(self, state, action, reward, state_, done, act_mf, act_mf_, uav_adj, uav_s, uav_s_):
        mem_idx = self.mem_cnt % self.mem_size

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done
        self.act_mf_memory[mem_idx] = act_mf
        self.next_act_mf_memory[mem_idx] = act_mf_
        self.uav_adj[mem_idx] = uav_adj
        self.uav_s[mem_idx] = uav_s
        self.uav_s_[mem_idx] = uav_s_

        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)
        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]
        act_mf = self.act_mf_memory[batch]
        act_mf_ = self.next_act_mf_memory[batch]
        uav_adj = self.uav_adj[batch]
        uav_s = self.uav_s[batch]
        uav_s_ = self.uav_s_[batch]

        return states, actions, rewards, terminals, states_, act_mf, act_mf_, uav_adj, uav_s, uav_s_

    def ready(self):
        return self.mem_cnt >= self.batch_size


# 变量初始化，可用于初始化权重参数
def fanin_init(size, fanin=None):
    # fain = fanin or size[0]
    v = 0.06  # 这是一个超参
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim  # 状态空间维度
        self.action_dim = action_dim  # 动作空间维度
        # 隐藏层维度
        self.hidden_size_LSTM = 64
        self.hidden_size1 = 64
        self.hidden_size2 = 128
        self.hidden_size3 = 32
        self.num_layers = 1

        # LSTM层
        self.lstm = nn.LSTM(input_size=self.state_dim, hidden_size=self.hidden_size_LSTM, num_layers=self.num_layers,
                            batch_first=True)
        # 全连接层
        self.fc1 = nn.Linear(self.hidden_size_LSTM, self.hidden_size1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.ln1 = nn.LayerNorm(self.hidden_size1)
        self.ln1.weight.data = fanin_init(self.ln1.weight.data.size())

        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.ln2 = nn.LayerNorm(self.hidden_size2)
        self.ln2.weight.data = fanin_init(self.ln2.weight.data.size())

        self.fc3 = nn.Linear(self.hidden_size2, self.hidden_size3)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.ln3 = nn.LayerNorm(self.hidden_size3)
        self.ln3.weight.data = fanin_init(self.ln3.weight.data.size())

        self.fc4 = nn.Linear(self.hidden_size3, self.action_dim)
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())
        # self.ln4 = nn.LayerNorm(self.hidden_size4)
        # self.ln4.weight.data = fanin_init(self.ln4.weight.data.size())

        # self.fc5 = nn.Linear(self.hidden_size4, self.action_dim)
        # self.fc5.weight.data = fanin_init(self.fc5.weight.data.size())

    def forward(self, state):
        x, hn = self.lstm(state)
        # print(len(x))
        x = F.relu(self.ln1(self.fc1(x[:, -1, :])))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        action = torch.tanh(self.fc4(x))
        return action

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num, use_mf):
        super(Critic, self).__init__()
        self.state_dim = state_dim  # 状态空间维度
        self.action_dim = action_dim  # 动作空间维度
        self.num = num  # 无人机数量
        # 使用平均场
        self.use_mf = use_mf
        # 隐藏层维度
        self.hidden_size_s1 = 64
        self.hidden_size_mf1 = 64
        self.hidden_size_a1 = 64
        if self.use_mf:
            self.hidden_size1 = self.hidden_size_s1 + self.hidden_size_a1 + self.hidden_size_mf1
        else:
            self.hidden_size1 = self.hidden_size_s1 + self.hidden_size_a1
        self.hidden_size2 = 64

        # 全连接层
        self.fcs1 = nn.Linear(self.state_dim, self.hidden_size_s1)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.ln1 = nn.LayerNorm(self.hidden_size_s1)
        self.ln1.weight.data = fanin_init(self.ln1.weight.data.size())

        # 平均场层
        if self.use_mf:
            # GAT层
            self.SAGE = GraphSAGE(self.state_dim, 16, self.action_dim)
            self.fmf1 = nn.Linear(self.action_dim, self.hidden_size_mf1)
            self.fmf1.weight.data = fanin_init(self.fmf1.weight.data.size())
            self.ln2 = nn.LayerNorm(self.hidden_size_mf1)
            self.ln2.weight.data = fanin_init(self.ln2.weight.data.size())

        self.fca1 = nn.Linear(self.action_dim, self.hidden_size_a1)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        self.ln3 = nn.LayerNorm(self.hidden_size_a1)
        self.ln3.weight.data = fanin_init(self.ln3.weight.data.size())

        self.fc1 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.ln4 = nn.LayerNorm(self.hidden_size2)
        self.ln4.weight.data = fanin_init(self.ln4.weight.data.size())

        self.fc3 = nn.Linear(self.hidden_size2, 1)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

    def forward(self, state, action, act_mf, uav_adj, uav_s):
        # 之前是直接把平均动作act_mf传进来，现在传进来一组无人机动作矩阵，通过GAT计算权重后，再通过动作矩阵结合权重求出平均场动作
        s1 = F.relu(self.ln1(self.fcs1(state)))
        a1 = F.relu(self.ln3(self.fca1(action)))
        x = torch.cat((s1, a1), dim=1)
        if self.use_mf:
            # 进行GAT预处理，act_mf[b_s, 80, act_dim]，uav_adj[b_s, 80, 80]，uav_s[b_s, 80, s_dim]
            batch = []
            for i in range(int(uav_adj.shape[0])):  # b_s
                edges = np.argwhere(uav_adj[i] > 0)  # 提取所有连接对坐标
                edge_index = torch.tensor([list(edges[:, 0]),  # 源节点索引
                                           list(edges[:, 1])],  # 目标节点索引
                                          dtype=torch.long).to(device)  # [2, m]
                temp_uav_s = uav_s[i][:self.num]  # 提取出对应无人机个数的特征向量
                data = Data(x=temp_uav_s, edge_index=edge_index)
                batch.append(data)
            batch = Batch.from_data_list(batch)
            # 经过GAT层计算权重
            w = self.SAGE.forward(batch.x, batch.edge_index)  # return_attention_weights=True
            w = F.leaky_relu(w)
            w = torch.softmax(w, dim=1)  # [b_s*num, act_dim]
            # 通过权重w对计算平均场动作a_mf
            w_ = w.reshape(int(w.shape[0] / self.num), self.num, self.action_dim)  # [b_s, num, act_dim]
            a_mf = act_mf[:, :self.num, :] * w_  # [b_s, num, act_dim]
            a_mf = a_mf.sum(dim=1) / self.num  # 对第二个维度进行求和取均值 [b_s, act_dim]

            mf1 = F.relu(self.ln2(self.fmf1(a_mf)))
            x = torch.cat((x, mf1), dim=1)
        x = F.relu(self.ln4(self.fc1(x)))
        q = self.fc3(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


def soft_update(target, source, tau):
    """
    软更新，在主网络参数基础上，做较小的改变，更新到目标网络
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    硬更新，在主网络参数基础上，做较小的改变，更新到目标网络
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class GL_MFDDPG:
    def __init__(self, state_dim, action_dim, num, use_mf, act_mf, gamma=0.99, tau=0.005, max_size=100000, bact_size=64,
                 learning_rate=0.005, epsilon=0.3, checkpoint_dir='./GL_MFDDPG_model'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新参数
        self.mem_size = max_size  # buffer容量大小
        self.batch_size = bact_size  # 单次训练抽样数量
        self.memory = ReplayBuffer(self.mem_size, self.state_dim, self.action_dim, self.batch_size, num)  # 回放缓存区
        self.noise = OUActionNoise(self.action_dim)
        self.checkpoint_dir = checkpoint_dir
        self.epsilon = epsilon
        self.use_mf = use_mf
        self.uav_num = num
        self.act_mf = act_mf

        self.critic_loss = torch.tensor(0)
        self.actor_loss = torch.tensor(0)

        # 策略网络（在线和目标）及其优化器
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.target_actor = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), learning_rate)

        # 价值网络（在线和目标）及其优化器
        self.critic = Critic(self.state_dim, self.action_dim, self.uav_num, use_mf=self.use_mf).to(device)
        self.target_critic = Critic(self.state_dim, self.action_dim, self.uav_num, use_mf=self.use_mf).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), learning_rate)

        # 用在线网络更新目标网络
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    # 更新网络参数
    def update_network_parameters(self):
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    # 将无人机信息存放到buffer中
    def remember(self, state, action, reward, done, state_, act_mf, act_mf_, uav_adj, uav_s, uav_s_):
        self.memory.store_transition(state, action, reward, state_, done, act_mf, act_mf_, uav_adj, uav_s, uav_s_)

    # 选择动作
    def choose_action(self, state, train=True):
        self.actor.eval()  # 网络评估模式，关闭梯度计算，单纯选择动作
        state = torch.tensor([state], dtype=torch.float32).to(device)
        action = self.actor.forward(state).squeeze()
        action_ = action
        # 使用平均场
        if self.use_mf:
            pass

        # 训练时，加入OU噪声
        if train:
            # 随机选择或根据网络选择
            if random.uniform(0, 1) < self.epsilon:
                action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                action = torch.tensor(action, dtype=torch.float32).to(device)

        self.actor.train()  # 从评估模式转为训练模式
        return action.detach().cpu().numpy(), action_.detach().cpu().numpy()

    # 训练
    def learn(self):
        # 当buffer中数量不足batch_size时，不训练
        if not self.memory.ready():
            return

        state, action, reward, done, state_, act_mf, act_mf_, uav_adj, uav_s, uav_s_ = self.memory.sample_buffer()
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action_tensor = torch.tensor(action, dtype=torch.float32).to(device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
        done_tensor = torch.tensor(done).to(device)
        state_tensor_ = torch.tensor(state_, dtype=torch.float32).to(device)
        act_mf_tensor = torch.tensor(act_mf, dtype=torch.float32).to(device)
        act_mf_tensor_ = torch.tensor(act_mf_, dtype=torch.float32).to(device)
        # uav_adj_tensor = torch.tensor(uav_adj, dtype=torch.float32).to(device)
        uav_s_tensor = torch.tensor(uav_s, dtype=torch.float32).to(device)
        uav_s_tensor_ = torch.tensor(uav_s_, dtype=torch.float32).to(device)

        with torch.no_grad():  # 此时不计算梯度，节省时间
            action_tensor_ = self.target_actor.forward(state_tensor_)
            q_ = self.target_critic.forward(state_tensor_[:, -1, :], action_tensor_,
                                            act_mf_tensor_, uav_adj, uav_s_tensor_).view(-1)
            q_[done_tensor] = 0.0
            target = reward_tensor + self.gamma * q_
        q = self.critic.forward(state_tensor[:, -1, :], action_tensor,
                                act_mf_tensor, uav_adj, uav_s_tensor).view(-1)

        # 计算损失值
        critic_loss = F.mse_loss(q, target.detach())
        self.critic_loss = critic_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action_tensor = self.actor.forward(state_tensor)
        actor_loss = - torch.mean(self.critic(state_tensor[:, -1, :], new_action_tensor,
                                              act_mf_tensor, uav_adj, uav_s_tensor))
        self.actor_loss = actor_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新
        self.update_network_parameters()

    # 保存模型
    def save_model(self, episode, num):
        # 判断路径是否存在，不存在创建
        if not os.path.exists(self.checkpoint_dir + '/Actor{}'.format(num)):
            os.makedirs(self.checkpoint_dir + '/Actor{}'.format(num))
        if not os.path.exists(self.checkpoint_dir + '/Target_actor{}'.format(num)):
            os.makedirs(self.checkpoint_dir + '/Target_actor{}'.format(num))
        if not os.path.exists(self.checkpoint_dir + '/Critic{}'.format(num)):
            os.makedirs(self.checkpoint_dir + '/Critic{}'.format(num))
        if not os.path.exists(self.checkpoint_dir + '/Target_critic{}'.format(num)):
            os.makedirs(self.checkpoint_dir + '/Target_critic{}'.format(num))
        self.actor.save_checkpoint(self.checkpoint_dir + '/Actor{}'.format(num) + '/DDPG_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          '/Target_actor{}'.format(num) + '/DDPG_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic.save_checkpoint(self.checkpoint_dir + '/Critic{}'.format(num) + '/DDPG_critic_{}'.format(episode))
        print('Saving critic network successfully!')
        self.target_critic.save_checkpoint(self.checkpoint_dir +
                                           '/Target_critic{}'.format(num) + '/DDPG_target_critic_{}'.format(episode))
        print('Saving target critic network successfully!')

    # 加载模型
    def load_models(self, episode, num):
        self.actor.load_checkpoint(self.checkpoint_dir + '/Actor{}'.format(num) + '/DDPG_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir + '/Target_actor{}'.format(num) +
                                          '/DDPG_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic.load_checkpoint(self.checkpoint_dir + '/Critic{}'.format(num) + '/DDPG_critic_{}'.format(episode))
        print('Loading critic network successfully!')
        self.target_critic.load_checkpoint(self.checkpoint_dir + '/Target_critic{}'.format(num) +
                                           '/DDPG_target_critic_{}'.format(episode))
        print('Loading target critic network successfully!')
