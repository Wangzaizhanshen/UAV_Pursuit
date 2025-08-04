"""
我方无人机目标是敌方基地，对途经的各种禁飞区和敌方无人机进行躲避，到达敌方基地位置即为任务成功
敌方目标的目标是我方无人机，对我方无人机进行拦截，同时避开自家作战单元，使用CBAA算法进行目标分配，击败我方无人机即为成功
构建多巡飞弹突防环境，添加攻击模型,存在对抗
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
import random
from CBAA import CBAA_agent
from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
from scipy.optimize import linear_sum_assignment


class UAVPursuit(gym.Env):
    """
    巡飞弹追击环境搭建
    战场面积500*500
    我/敌方巡飞弹速度大小1~2.5，速度方向0~2*pi，生成的位置范围[20~480, 20~480]
    巡飞弹探测范围为机头朝向[-90°,90°]，分为10个区间，每个区间大小18°，在每个区间内的探测范围为20

    state:维度为(agent_num,16)的数组
    [x, y, v, alpha,    该架巡飞弹的位置，速度，速度方向（与x轴之间的夹角）
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, 该巡飞弹探测范围为机头朝向[-90°,90°]，分为10个区间，每个区间大小18°，每个元素代表探测到的障碍物距离巡飞弹的距离
    d, beta]       敌方巡飞弹相对于该我方巡飞弹的距离d，巡飞弹速度方向与集结点位置的夹角beta

    action:维度为(agent_num,2)的数组
    [a_v, a_alpha]  a_v:改变速度大小,-1~1     a_alpha:改变速度方向,-pi/6~pi/6
    """

    def __init__(self):
        # 战场设置
        self.height = 500  # 战场长度
        self.width = 500  # 战场宽度
        self.t = 1  # 仿真步长1s
        # 我方巡飞弹设置
        self.uav_num = 10  # 巡飞弹数量
        self.uav_r = 1  # 巡飞弹近似为一个圆，半径为1
        self.uav_v_max = 2.5  # 巡飞弹最大速度
        self.uav_v_min = 1.5  # 巡飞弹最小速度
        self.uav_com_dis = 50  # 巡飞弹通信距离
        self.uav_probe = 50  # 巡飞弹雷达探测距离
        self.uav_probe_theta = 75  # 巡飞弹雷达探测半角°
        self.prob_max_num = 5  # 巡飞弹单次最多探测目标数量
        self.uav_ap_min = -1  # 巡飞弹线加速度最小值
        self.uav_ap_max = 1  # 巡飞弹线加速度最大值
        self.uav_av_min = - math.pi / 15  # 巡飞弹角加速度最小值
        self.uav_av_max = math.pi / 15  # 巡飞弹角加速度最大值
        self.attack_dis = 15  # 巡飞弹开始攻击距离
        self.kill_dis = 5  # 巡飞弹击落距离
        self.attack_angle = 0.3 * math.pi  # 巡飞弹攻击角度
        # 探测数据
        self.prob_indices = -np.ones([self.uav_num, self.uav_num - 1])  # 雷达范围内的己方巡飞弹序号，按照距离升序排列
        self.probe_d = np.zeros([self.prob_max_num, ])  # 雷达范围内距离最近的5个目标距离
        self.probe_a = np.zeros([self.prob_max_num, ])  # 雷达范围内距离最近的5个目标角度
        # 敌方巡飞弹设置
        self.target_num = 10  # 敌方巡飞弹数量
        self.target_r = 1
        self.tar_v_max = self.uav_v_max * 0.9
        self.tar_v_min = self.uav_v_min * 0.9
        self.t_com_dis = 50  # 巡飞弹通信距离
        self.t_probe = 50  # 巡飞弹雷达探测距离
        self.t_probe_theta = 75  # 巡飞弹雷达探测半角°
        self.t_prob_max_num = 5  # 巡飞弹单次最多探测目标数量
        self.prob_tar_indices = -np.ones([self.target_num, self.target_num - 1])  # 雷达范围内的己方巡飞弹序号，按照距离升序排列
        self.t_attack_dis = 15  # 敌方巡飞弹开始攻击距离
        self.t_kill_dis = 5  # 敌方巡飞弹击落距离
        self.t_attack_angle = 0.3 * math.pi  # 敌方巡飞弹攻击角度
        # 敌方巡飞弹状态简化
        self.target_x = np.zeros([self.target_num])
        self.target_y = np.zeros([self.target_num])
        self.target_v = np.zeros([self.target_num])
        self.target_alpha = np.zeros([self.target_num])
        self.target_x_ = np.zeros([self.target_num])
        self.target_y_ = np.zeros([self.target_num])
        self.target_v_ = np.zeros([self.target_num])
        self.target_alpha_ = np.zeros([self.target_num])
        # 地形障碍物设置
        self.obstacle_or_not = True  # 是否存在障碍物
        self.obstacle_num = 10  # 障碍物数量
        self.obstacle_r_min = 20  # 障碍物最小半径
        self.obstacle_r_max = 30  # 障碍物最大半径
        self.obstacle = np.array([])
        # 目标集结区设置
        self.gather_x = self.width - 20  # 集结区位置坐标
        self.gather_y = self.height - 20  # 集结区位置坐标
        self.gather_min = 50  # 集结区坐标最小值
        self.gather_max = self.width - 50  # 集结区坐标最大值
        self.gather_r_min = 10  # 集结区最小半径
        self.gather_r_max = 25  # 集结区最大半径
        self.gather_r = 10  # 预留
        # 初始态势设置
        self.area = random.randint(1, 4)
        self.step_num = 0
        # 训练设置
        self.round_max = 1000  # 单局游戏最大步数
        self.episode_max = 2000  # 最大训练轮数
        # 数据归一化
        self.data_min = np.array([self.uav_r, self.uav_r, self.uav_v_min, 0, self.uav_r, 0, 0, -math.pi, 0, -math.pi,
                                  0, -math.pi, 0, -math.pi, 0, -math.pi])
        self.data_max = np.array(
            [self.width, self.height, self.uav_v_max, 2 * math.pi, math.sqrt(self.width * self.height * 2),
             2 * math.pi, self.uav_probe, math.pi, self.uav_probe, math.pi, self.uav_probe, math.pi,
             self.uav_probe, math.pi, self.uav_probe, math.pi])
        self.data_min_new = -np.ones(16)
        self.data_max_new = np.ones(16)
        self.action_min = np.array([self.uav_ap_min, self.uav_av_min])
        self.action_max = np.array([self.uav_ap_max, self.uav_av_max])
        self.action_min_new = -np.ones(2)
        self.action_max_new = np.ones(2)
        # 指标
        self.success = 0  # 追击成功次数
        self.uav_live = 0  # 我方巡飞弹存活数量
        self.tar_live = 0  # 敌方巡飞弹存活数量
        # 任务分配智能体
        self.uav_ind = []
        self.tar_ind = []
        self.task = None
        self.task_position = None
        self.agent = None
        self.assignment_result = None
        self.G = None
        self.attack_idx = None
        self.flag = [False for _ in range(self.uav_num)]
        self.hit_pro = np.zeros([self.target_num])

        # 暴露率
        self.exposure_num = 0  # 无人机暴露数量
        self.exposure_uav = np.zeros(self.uav_num)
        self.exposure_rate = 0

        # 状态空间
        state_list_low, state_list_high = [], []
        for i in range(self.uav_num):
            state_low = [20, 20, self.uav_v_min, 0, 0, 0, 50, 0, 50, 0, 50, 0, 50, 0, 50, 0]
            state_list_low.append(state_low)
            state_high = [self.width - 20, self.height - 20, self.uav_v_max, math.pi * 2, 500, math.pi * 2,
                          50, 0, 50, 0, 50, 0, 50, 0, 50, 0]
            state_list_high.append(state_high)
        self.low_state = np.array(state_list_low, dtype=np.float32)
        self.high_state = np.array(state_list_high, dtype=np.float32)
        # shape=(2, 16)
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        # 动作空间
        action_list_low, action_list_high = [], []
        for i in range(self.uav_num):
            action_low = [-1, -math.pi / 15]
            action_list_low.append(action_low)
            action_high = [1, math.pi / 15]
            action_list_high.append(action_high)
        self.low_action = np.array(action_list_low, dtype=np.float32)
        self.high_action = np.array(action_list_high, dtype=np.float32)
        # shape = (2, 2)
        self.action_space = spaces.Box(
            self.low_action, self.high_action, dtype=np.float32
        )
        # 我方巡飞弹的交互信息
        self.state = np.array([])
        self.action = np.array([])
        self.reward = np.array([])
        self.state_ = np.array([])
        self.done = np.array([])
        self.temp_done = np.array([])
        # 敌方巡飞弹的交互信息
        self.t_state = np.array([])
        self.t_action = np.array([])
        self.t_reward = np.array([])
        self.t_state_ = np.array([])
        self.t_done = np.array([])
        self.temp_done_tar = np.array([])
        self.t_attack_idx = np.ones([self.uav_num], dtype=int) * (-1)
        self.t_hit_pro = np.ones([self.target_num])  # 敌方巡飞弹被击毁的概率
        self.attack_flag = [False for _ in range(self.target_num)]

        # 画图准备，存储我方巡飞弹和敌方巡飞弹的位置数据
        self.uav_x, self.uav_y, self.t_x, self.t_y = [], [], [], []
        for i in range(self.uav_num):
            self.uav_x.append([])
            self.uav_y.append([])
        for target in range(self.target_num):
            self.t_x.append([])
            self.t_y.append([])
        # pygame画图
        self.screen = None
        self.clock = None
        self.isopen = True
        self.fps = 60
        # 我机与敌机本时刻的距离和角度
        self.s_d = np.zeros([self.uav_num])
        self.s_a = np.zeros([self.uav_num])
        # 指标
        self.success = 0  # 突防任务成功次数
        self.uav_live = self.uav_num  # 我方巡飞弹突防数量
        self.tar_live = self.target_num  # 敌方巡飞弹存活数量
        self.kill_flag = np.zeros(self.target_num)

    # 状态归一化 [-1, 1]
    def state_normalization(self, state, flag):
        if flag:  # 归一化
            state = (state - self.data_min) * (self.data_max_new - self.data_min_new) / (self.data_max - self.data_min) \
                    + self.data_min_new
        else:  # 去归一化
            state = (state - self.data_min_new) * (self.data_max - self.data_min) / \
                    (self.data_max_new - self.data_min_new) + self.data_min
        return state

    # 动作归一化 [-1, 1]
    def action_normalization(self, action, flag):
        if flag:  # 动作归一化
            action = (action - self.action_min) * (self.action_max_new - self.action_min_new) / \
                     (self.action_max - self.action_min) + self.action_min_new
        else:  # 动作去归一化
            action = (action - self.action_min_new) * (self.action_max - self.action_min) / \
                     (self.action_max_new - self.action_min_new) + self.action_min

        return action

    # 计算二维距离
    def calc_dis(self, x0, y0, x1, y1):
        d = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        return d

    # 计算巡飞弹速度方向与集结点位置的夹角
    def calc_angle(self, x0, y0, v0, alpha0, x1, y1):
        vector_v = np.array([v0 * math.cos(alpha0), v0 * math.sin(alpha0)])
        vector_d = np.array([x1 - x0, y1 - y0])
        cross_product = np.cross(vector_v, vector_d)
        vector_v_mo = math.sqrt(vector_v[0] ** 2 + vector_v[1] ** 2)
        vector_d_mo = math.sqrt(vector_d[0] ** 2 + vector_d[1] ** 2)
        beta_cos = (vector_v[0] * vector_d[0] + vector_v[1] * vector_d[1]) / (vector_d_mo * vector_v_mo)
        beta_cos = np.clip(beta_cos, -1, 1)
        beta = math.acos(beta_cos)  # 此时beta值为[0, 180°]
        if cross_product < 0:
            beta = -beta
        return beta

    # 生成指定数量的不相交的圆，生成障碍物
    def generate_circle(self, r_min, r_max, num, gather_x, gather_y, gather_r):
        circles = []
        circles.append((gather_x, gather_y, gather_r))
        result = []
        if len(self.obstacle):
            for i in range(self.obstacle_num):
                circles.append(self.obstacle[i])
        while True:
            radius = int(random.uniform(r_min, r_max))
            x, y = 0, 0
            if self.area == 1:  # 障碍物在右边
                x = random.uniform(0.2 * self.width + radius, 0.8 * self.width - radius)
                y = random.uniform(radius, self.height - radius)
            elif self.area == 2:  # 障碍物在上边
                x = random.uniform(radius, self.width - radius)
                y = random.uniform(0.2 * self.height + radius, 0.8 * self.height - radius)
            elif self.area == 3:  # 障碍物在左边
                x = random.uniform(0.2 * self.width + radius, 0.8 * self.width - radius)
                y = random.uniform(radius, self.height - radius)
            elif self.area == 4:  # 障碍物在下边
                x = random.uniform(radius, self.width - radius)
                y = random.uniform(0.2 * self.height + radius, 0.8 * self.height - radius)

            circle = (x, y, radius)

            if not any(self.intersect(circle, c) for c in circles):
                result.append(circle)
                circles.append(circle)
                if len(result) == num:
                    circles.clear()
                    return result

    # 判断2圆是否相交
    def intersect(self, circle1, circle2):
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        r1 += 8
        r2 += 8
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return distance < (r1 + r2)

    # 对一组数据进行升序排序，并返回其序号
    def sort_and_get_indices(self, arr, axis):
        sorted_indices = np.argsort(arr, axis=axis)  # 对数组进行升序排序，并返回排序后的索引
        sorted_arr = np.sort(arr, axis=axis)  # 对数组进行升序排序
        return sorted_arr, sorted_indices

    # 根据索引对原数组进行重新排序
    def sort_array_by_sequence(self, arr, sequence):
        sorted_arr = arr[sequence]
        return sorted_arr

    # 在一组升序的数据中插入一个新数据，并剔除最大的数据，保留原数组的维度
    def insert_number(self, arr, target):
        left = 0
        right = len(arr) - 1
        insert_pos = len(arr)

        while left <= right:
            mid = (left + right) // 2

            if target <= arr[mid]:
                insert_pos = mid
                right = mid - 1
            else:
                left = mid + 1

        arr.insert(insert_pos, target)
        arr.pop()  # 移除最后一个元素
        return arr, insert_pos

    # 计算我机雷达探测的信息
    def calc_probe_information(self, state):
        """
        计算我方巡飞弹雷达探测的信息
        :param state: 我方巡飞弹状态空间 np.array[(uav_num, 16)]
        :return: state: 计算后的我方巡飞弹状态空间 np.array[(uav_num, 16)]
        """
        prob_indices = -np.ones([self.uav_num, self.uav_num - 1])
        # 巡飞弹探测其他巡飞弹
        # 单巡飞弹，探测不到信息
        if self.uav_num == 1:
            state[0][6] = self.uav_probe  # 距离为探测距离上限
            state[0][7] = 0  # 角度为0
            state[0][8] = self.uav_probe  # 距离为探测距离上限
            state[0][9] = 0  # 角度为0
            state[0][10] = self.uav_probe  # 距离为探测距离上限
            state[0][11] = 0  # 角度为0
            state[0][12] = self.uav_probe  # 距离为探测距离上限
            state[0][13] = 0  # 角度为0
            state[0][14] = self.uav_probe  # 距离为探测距离上限
            state[0][15] = 0  # 角度为0
        # 多巡飞弹
        else:
            for i in range(self.uav_num):
                temp_d = -np.ones([self.uav_num, ])
                temp_a = np.zeros([self.uav_num, ])
                temp_d_ = -np.ones([self.uav_num, ])  # 通信距离判断
                # 判断本巡飞弹是否停止
                if self.done[i]:
                    for j in range(self.uav_num):
                        temp_d[j] = self.uav_probe
                        temp_a[j] = 0
                        temp_d_[j] = self.uav_com_dis
                else:
                    # 探测我方其他巡飞弹
                    for j in range(self.uav_num):
                        if i == j:
                            continue
                        # 判断是否有其他巡飞弹停止，若停止则不计算该巡飞弹的距离和角度
                        if self.done[j]:
                            temp_d[j] = self.uav_probe
                            temp_a[j] = 0
                            temp_d_[j] = self.uav_probe
                            continue
                        # 计算距离和角度
                        temp_d[j] = self.calc_dis(state[i][0], state[i][1], state[j][0], state[j][1])
                        temp_a[j] = self.calc_angle(state[i][0], state[i][1], state[i][2],
                                                    state[i][3], state[j][0], state[j][1])
                        temp_d_[j] = self.calc_dis(state[i][0], state[i][1], state[j][0], state[j][1])
                        if temp_d[j] > self.uav_probe:  # 探测距离限制，超过的距离最大值，角度为0
                            temp_d[j] = self.uav_probe + 1
                            temp_a[j] = 0
                        elif abs(temp_a[j]) > np.degrees(self.uav_probe_theta):  # 在探测距离范围内，但不在探测角度内，同上
                            temp_d[j] = self.uav_probe + 1
                            temp_a[j] = 0
                        # 通信距离限制
                        if temp_d_[j] > self.uav_com_dis:
                            temp_d_[j] = self.uav_com_dis + 1
                    # 对巡飞弹i的通信范围内巡飞弹进行排序
                    temp_d_, comm_indices1 = self.sort_and_get_indices(temp_d_, -1)
                    temp_d, temp_indices = self.sort_and_get_indices(temp_d, -1)
                    temp_a = self.sort_array_by_sequence(temp_a, temp_indices)
                    # 判断对于巡飞弹i，有哪些己方巡飞弹在探测范围内
                    for k in range(len(temp_d)):
                        # 把之前加的1减去
                        if self.uav_probe + 1 or -1 in temp_d:
                            temp_d[k] = self.uav_probe
                    # 去掉自身
                    temp_d_ = temp_d_[1:]
                    temp_d = temp_d[1:]
                    temp_a = temp_a[1:]
                    comm_indices1 = comm_indices1[1:]
                    for k in range(len(temp_d_)):
                        if temp_d_[k] <= self.uav_com_dis:
                            prob_indices[i][k] = comm_indices1[k]
                self.prob_indices = prob_indices  # 存储做通信矩阵
                # 探测到巡飞弹不足5架
                if self.uav_num <= self.prob_max_num:
                    for k in range(temp_d.size):
                        state[i][6 + 2 * k] = temp_d[k]
                        state[i][7 + 2 * k] = temp_a[k]
                else:
                    for k in range(self.prob_max_num):
                        state[i][6 + 2 * k] = temp_d[k]
                        state[i][7 + 2 * k] = temp_a[k]
        # 探测禁飞区
        if self.obstacle_or_not:
            for i in range(self.uav_num):
                # 计算每个巡飞弹到每个障碍物的距离和角度
                obs_d = -np.ones([self.obstacle_num, ])
                obs_a = np.zeros([self.obstacle_num, ])
                for j in range(self.obstacle_num):
                    obs_d[j] = self.calc_dis(state[i][0], state[i][1], self.obstacle[j][0], self.obstacle[j][1]) - \
                               self.obstacle[j][2]
                    obs_a[j] = self.calc_angle(state[i][0], state[i][1], state[i][2], state[i][3],
                                               self.obstacle[j][0], self.obstacle[j][1])
                    # 判断该障碍物不在巡飞弹探测范围内
                    if obs_d[j] > self.uav_probe or abs(obs_a[j]) > np.degrees(self.uav_probe_theta):
                        continue
                    # 判断该障碍物到巡飞弹的距离是否在最近的5个中
                    temp_list_d = [state[i][6], state[i][8], state[i][10], state[i][12], state[i][14]]
                    temp_list_d, insert_i = self.insert_number(temp_list_d, obs_d[j])

                    temp_list_a = [state[i][7], state[i][9], state[i][11], state[i][13], state[i][15]]
                    temp_list_a.insert(insert_i, obs_a[j])
                    temp_list_a.pop()
                    # 若在最近的5个中，则更新state
                    if insert_i < 5:
                        state[i][6] = temp_list_d[0]
                        state[i][8] = temp_list_d[1]
                        state[i][10] = temp_list_d[2]
                        state[i][12] = temp_list_d[3]
                        state[i][14] = temp_list_d[4]

                        state[i][7] = temp_list_a[0]
                        state[i][9] = temp_list_a[1]
                        state[i][11] = temp_list_a[2]
                        state[i][13] = temp_list_a[3]
                        state[i][15] = temp_list_a[4]

        return state

    # 计算敌机雷达探测的信息，同样存在三个禁飞区，躲避我方巡飞弹
    def calc_t_probe_information(self, t_state, state):
        """
        计算敌方巡飞弹雷达探测的信息
        :param t_state: 敌方巡飞弹状态空间 np.array[(tar_num, 16)]
                t_state
        :return: state: 敌方后的我方巡飞弹状态空间 np.array[(tar_num, 16)]
        """
        prob_indices = -np.ones([self.target_num, self.target_num - 1])
        # 巡飞弹探测其他巡飞弹
        # 单巡飞弹，探测不到信息
        if self.target_num == 1:
            t_state[0][6] = self.t_probe  # 距离为探测距离上限
            t_state[0][7] = 0  # 角度为0
            t_state[0][8] = self.t_probe  # 距离为探测距离上限
            t_state[0][9] = 0  # 角度为0
            t_state[0][10] = self.t_probe  # 距离为探测距离上限
            t_state[0][11] = 0  # 角度为0
            t_state[0][12] = self.t_probe  # 距离为探测距离上限
            t_state[0][13] = 0  # 角度为0
            t_state[0][14] = self.t_probe  # 距离为探测距离上限
            t_state[0][15] = 0  # 角度为0
        # 多巡飞弹
        else:
            for i in range(self.target_num):
                temp_d = -np.ones([self.target_num + self.uav_num, ])
                temp_a = np.zeros([self.target_num + self.uav_num, ])
                temp_d_ = -np.ones([self.target_num, ])  # 通信距离判断
                # 判断本巡飞弹是否停止
                if self.t_done[i]:
                    for j in range(self.target_num):
                        temp_d[j] = self.t_probe
                        temp_a[j] = 0
                        temp_d_[j] = self.t_com_dis
                else:
                    # 探测其他敌方巡飞弹
                    for j in range(self.target_num):
                        if i == j:
                            continue
                        # 判断是否有其他巡飞弹停止，若停止则不计算该巡飞弹的距离和角度
                        if self.t_done[j]:
                            temp_d[j] = self.t_probe
                            temp_a[j] = 0
                            temp_d_[j] = self.t_probe
                            continue
                        # 计算距离和角度
                        temp_d[j] = self.calc_dis(t_state[i][0], t_state[i][1], t_state[j][0], t_state[j][1])
                        temp_a[j] = self.calc_angle(t_state[i][0], t_state[i][1], t_state[i][2],
                                                    t_state[i][3], t_state[j][0], t_state[j][1])
                        temp_d_[j] = self.calc_dis(t_state[i][0], t_state[i][1], t_state[j][0], t_state[j][1])
                        if temp_d[j] > self.t_probe:  # 探测距离限制，超过的距离最大值，角度为0
                            temp_d[j] = self.t_probe + 1
                            temp_a[j] = 0
                        elif abs(temp_a[j]) > np.degrees(self.t_probe_theta):  # 在探测距离范围内，但不在探测角度内，同上
                            temp_d[j] = self.t_probe + 1
                            temp_a[j] = 0
                        # 通信距离限制
                        if temp_d_[j] > self.t_com_dis:
                            temp_d_[j] = self.t_com_dis + 1
                    # 探测敌方巡飞弹
                    for j in range(self.uav_num):
                        # 判断是否有其他巡飞弹停止，若停止则不计算该巡飞弹的距离和角度
                        if self.done[j]:
                            temp_d[self.target_num - 1 + j] = self.uav_probe
                            temp_a[self.target_num - 1 + j] = 0
                            continue
                        # 计算距离和角度
                        temp_d[self.target_num - 1 + j] = self.calc_dis(t_state[i][0], t_state[i][1],
                                                                        state[j][0], state[j][1])
                        temp_a[self.target_num - 1 + j] = self.calc_angle(t_state[i][0], t_state[i][1],
                                                                          t_state[i][2],
                                                                          t_state[i][3], state[j][0],
                                                                          state[j][1])
                        # 探测距离和角度限制，不在探测范围内，距离最大，角度置0
                        if temp_d[self.target_num - 1 + j] > self.uav_probe or \
                                abs(temp_a[self.target_num - 1 + j]) > np.degrees(self.uav_probe_theta):
                            temp_d[self.target_num - 1 + j] = self.uav_probe
                            temp_a[self.target_num - 1 + j] = 0
                    # 对巡飞弹i求出的所有探测到巡飞弹距离进行排序
                    temp_d, temp_indices = self.sort_and_get_indices(temp_d, -1)
                    temp_a = self.sort_array_by_sequence(temp_a, temp_indices)
                    # 对巡飞弹i的通信范围内巡飞弹进行排序
                    temp_d_, comm_indices1 = self.sort_and_get_indices(temp_d_, -1)
                    # 判断对于巡飞弹i，有哪些己方巡飞弹在探测范围内
                    for k in range(len(temp_d)):
                        # 把之前加的1减去
                        if self.t_probe + 1 or -1 in temp_d:
                            temp_d[k] = self.t_probe
                    # 去掉自身
                    temp_d_ = temp_d_[1:]
                    temp_d = temp_d[1:]
                    temp_a = temp_a[1:]
                    temp_indices = temp_indices[1:]
                    for k in range(len(temp_d_)):
                        if temp_d_[k] <= self.uav_com_dis:
                            prob_indices[i][k] = temp_indices[k]
                self.prob_tar_indices = prob_indices
                # 探测到巡飞弹不足5架
                if self.target_num <= self.t_prob_max_num:
                    for k in range(temp_d.size):
                        t_state[i][6 + 2 * k] = temp_d[k]
                        t_state[i][7 + 2 * k] = temp_a[k]
                else:
                    for k in range(self.prob_max_num):
                        t_state[i][6 + 2 * k] = temp_d[k]
                        t_state[i][7 + 2 * k] = temp_a[k]
        # 探测地形禁飞区
        if self.obstacle_or_not:
            for i in range(self.target_num):
                # 计算每个巡飞弹到每个障碍物的距离和角度
                obs_d = -np.ones([self.obstacle_num, ])
                obs_a = np.zeros([self.obstacle_num, ])
                for j in range(self.obstacle_num):
                    obs_d[j] = self.calc_dis(t_state[i][0], t_state[i][1], self.obstacle[j][0], self.obstacle[j][1]) - \
                               self.obstacle[j][2]
                    obs_a[j] = self.calc_angle(t_state[i][0], t_state[i][1], t_state[i][2], t_state[i][3],
                                               self.obstacle[j][0], self.obstacle[j][1])
                    # 判断该障碍物不在巡飞弹探测范围内
                    if obs_d[j] > self.t_probe or abs(obs_a[j]) > np.degrees(self.t_probe_theta):
                        continue
                    # 判断该障碍物到巡飞弹的距离是否在最近的5个中
                    temp_list_d = [t_state[i][6], t_state[i][8], t_state[i][10], t_state[i][12], t_state[i][14]]
                    temp_list_d, insert_i = self.insert_number(temp_list_d, obs_d[j])

                    temp_list_a = [t_state[i][7], t_state[i][9], t_state[i][11], t_state[i][13], t_state[i][15]]
                    temp_list_a.insert(insert_i, obs_a[j])
                    temp_list_a.pop()
                    # 若在最近的5个中，则更新state
                    if insert_i < 5:
                        t_state[i][6] = temp_list_d[0]
                        t_state[i][8] = temp_list_d[1]
                        t_state[i][10] = temp_list_d[2]
                        t_state[i][12] = temp_list_d[3]
                        t_state[i][14] = temp_list_d[4]

                        t_state[i][7] = temp_list_a[0]
                        t_state[i][9] = temp_list_a[1]
                        t_state[i][11] = temp_list_a[2]
                        t_state[i][13] = temp_list_a[3]
                        t_state[i][15] = temp_list_a[4]

        return t_state

    # 随机生成的巡飞弹不能在障碍物内，不能和敌方巡飞弹重合
    def generate_uav(self):
        while True:
            done = 0
            state = self.observation_space.sample()[0].tolist()
            if self.area == 1:  # 左
                state[0] = random.uniform(30, 0.4 * self.width)
                state[1] = random.uniform(30, 0.4 * self.height)
            elif self.area == 2:  # 中
                state[0] = random.uniform(0.2 * self.width, 0.8 * self.width)
                state[1] = random.uniform(0.3 * self.height, 0.7 * self.height)
            elif self.area == 3:  # 右
                state[0] = random.uniform(0.6 * self.width, 470)
                state[1] = random.uniform(0.6 * self.height, 470)
                state[3] = math.pi
            # 集结区
            temp_d = self.calc_dis(state[0], state[1], self.gather_x, self.gather_y)
            if temp_d <= self.uav_r + self.gather_r:
                continue
            # 禁飞区
            if self.obstacle_or_not:  # 存在禁飞区
                for i in range(self.obstacle_num):
                    temp_d = self.calc_dis(state[0], state[1], self.obstacle[i][0], self.obstacle[i][1])
                    if temp_d > self.uav_r + self.obstacle[i][2] + 20:
                        done += 1

            if done == self.obstacle_num:
                return state
            # if not self.obstacle_or_not and not self.artillery_or_not and not self.radar_or_not:  # 其他情况分情况讨论，不存在禁飞区
            #     return state

    def generate_t_uav(self):
        while True:
            done = 0
            t_state = self.observation_space.sample()[0].tolist()
            t_state[0] = random.uniform(30, 0.3 * self.width)
            t_state[1] = random.uniform(0.7 * self.height, 470)
            # 集结区
            temp_d = self.calc_dis(t_state[0], t_state[1], self.gather_x, self.gather_y)
            if temp_d <= self.uav_r + self.gather_r:
                continue
            # 禁飞区
            if self.obstacle_or_not:  # 存在禁飞区
                for i in range(self.obstacle_num):
                    temp_d = self.calc_dis(t_state[0], t_state[1], self.obstacle[i][0], self.obstacle[i][1])
                    if temp_d > self.uav_r + self.obstacle[i][2] + 20:
                        done += 1
            if done == self.obstacle_num:
                return t_state

    # 进行目标态势评估，计算威胁矩阵
    def situation_assessment(self, state, t_state):
        """
        态势评估函数
        :param state: 我方巡飞弹状态
        :param t_state: 敌方巡飞弹状态
        :return:
        """
        self.attack_flag = [False for _ in range(self.target_num)]  # 哪些我方敌机正被攻击
        # 权重因子
        w1, w2 = 0.25 * self.width, 0.4
        # 综合威胁矩阵
        s_threat = np.zeros([self.uav_num, self.target_num])
        self.uav_ind = []
        self.tar_ind = []
        # 计算综合威胁矩阵
        for i in range(self.uav_num):
            for j in range(self.target_num):
                # 若我机坠毁，或该巡飞弹已被分配，赋0
                if self.done[i]:
                    s_threat[i][j] = 0
                    continue
                # 若敌机坠毁，赋0
                if self.t_done[j]:
                    s_threat[i][j] = 0
                    continue
                # 敌机相对于我机
                d = self.calc_dis(state[i][0], state[i][1], t_state[j][0], t_state[j][1])
                if abs(state[i][5]) <= 0.5 * math.pi:
                    beta = self.calc_angle(state[i][0], state[i][1], state[i][2], state[i][3],
                                           t_state[j][0], t_state[j][1])
                    s_threat[i][j] += w1 / d + w2 * math.cos(beta)
                else:
                    s_threat[i][j] += w1 / d
        # 处理威胁矩阵
        # 威胁矩阵中0行和0列代表无人机已坠毁或被分配，将其剔除
        non_zero_rows = np.any(s_threat, axis=1)  # 找到非零元素所在的行
        non_zero_cols = np.any(s_threat, axis=0)  # 找到非零元素所在的列
        s_threat_ = s_threat[non_zero_rows][:, non_zero_cols]  # 得到新的威胁矩阵
        # 确定新威胁矩阵和旧威胁矩阵行、列的对应关系
        row_mapping = np.where(non_zero_rows)[0]  # 存活的我方无人机索引
        col_mapping = np.where(non_zero_cols)[0]  # 存活的敌方无人机索引
        # 存储初始威胁矩阵，最全
        s_threat_0 = s_threat_.copy()

        # 进行迭代分配
        # 当我方无人机数量大于等于目标数量，进行多对多对一分配：
        while True:
            if len(col_mapping) != 0:  # 仍存在目标
                if len(row_mapping) >= len(col_mapping):
                    # 初始化CBBA算法，进行目标分配
                    self.task = np.array(col_mapping, dtype=int)
                    self.agent = [CBAA_agent(i, self.task, s_threat_[i]) for i in range(len(row_mapping))]
                    self.G = np.ones((len(row_mapping), len(row_mapping)))
                    self.assignment_result = self.target_assignment(len(row_mapping))
                    # 处理分配信息
                    uav_ind = row_mapping.tolist()  # 存活敌机ID
                    tar_ind = [-1 for _ in range(len(row_mapping))]
                    for i in range(len(uav_ind)):
                        if sum(self.assignment_result[i]):
                            for j in range(len(col_mapping)):
                                if self.assignment_result[i][j] == 1:
                                    tar_ind[i] = self.task[j]
                else:
                    num = len(row_mapping)
                    t_threat = np.sum(s_threat_, axis=0)
                    sort_array, sort_ind = self.sort_and_get_indices(t_threat, 0)
                    ind = sort_ind[-num:]  # 存储索引
                    task = []
                    d = []
                    for i in range(len(col_mapping)):  # 找对应索引
                        if i in ind:
                            task.append(col_mapping[i])  # 添加对应任务
                        else:
                            d.append(i)  # 添加需要删除的列索引
                    s_threat_1 = np.delete(s_threat_, d, axis=1)
                    # 重新初始化CBBA信息
                    self.task = np.array(task, dtype=int)
                    self.agent = [CBAA_agent(i, self.task, s_threat_1[i]) for i in range(num)]
                    self.G = np.ones((num, num))
                    # 进行目标分配
                    self.assignment_result = self.target_assignment(num)
                    # 处理分配信息
                    uav_ind = row_mapping.tolist()
                    tar_ind = [-1 for _ in range(num)]
                    for i in range(num):
                        if sum(self.assignment_result[i]):
                            for j in range(num):
                                if self.assignment_result[i][j] == 1:
                                    tar_ind[i] = self.task[j]
            else:
                uav_ind = row_mapping.tolist()
                tar_ind = [-1 for _ in range(len(uav_ind))]

            if len(self.uav_ind) == 0:
                self.uav_ind = uav_ind
                self.tar_ind = tar_ind
            else:
                for i in range(len(uav_ind)):
                    ind = self.uav_ind.index(uav_ind[i])
                    self.tar_ind[ind] = tar_ind[i]

            for i in range(self.target_num):
                if i in tar_ind:  # 成为攻击目标
                    self.attack_flag[i] = True
            if -1 in self.tar_ind:  # 实现多对一分配
                # 未分配的无人机索引
                ind = [index for index, value in enumerate(self.tar_ind) if value == -1]  # self.tar_ind.index(-1)
                # print(ind)
                row_mapping_l = []
                s_threat_l = []
                # 更新威胁矩阵，去掉已分配的我方无人机
                for i in range(len(ind)):
                    row_mapping_l.append(self.uav_ind[ind[i]])
                    s_threat_l.append(s_threat_0[ind[i]])
                row_mapping = np.array(row_mapping_l)
                s_threat_ = np.array(s_threat_l)

            else:
                break

        # 输出攻击无人机ID,目标ID，相对应
        # print("uav:", self.uav_ind)
        # print("tar:", self.tar_ind)
        return self.uav_ind, self.tar_ind

    # 进行目标分配
    def target_assignment(self, num):
        t = 0  # 迭代次数
        assignment_result = []
        # print(len(self.agent))
        while True:
            converged_list = []
            for agent in self.agent:
                # 根据当前本地信息选择任务
                agent.select_task()

            # 冲突消除阶段
            if num > 1:
                message_pool = [agent.send_message() for agent in self.agent]  # 信息都放信息池里
                for id, agent in enumerate(self.agent):
                    # 接收邻居的投标信息
                    g = self.G[id]

                    connected, = np.where(g == 1)
                    connected = list(connected)
                    connected.remove(id)  # 移除自己

                    if len(connected) > 0:
                        Y = {neighbor_id: message_pool[neighbor_id] for neighbor_id in connected}  # 组成邻居投标信息
                    else:
                        Y = None

                    # Update local information and decision
                    if Y is not None:
                        converged = agent.update_task(Y)  # 更新任务列表吧
                        converged_list.append(converged)  # 存储更新状态
                    assignment_result.append(agent.x)

                t += 1

                if sum(converged_list) == num:  # 都不存在冲突就结束，不然就继续迭代
                    break
            else:
                assignment_result.append(self.agent[0].x)
                break
        return assignment_result

    # 计算击毁概率
    def cal_hit_probability(self, state, t_state):
        """
        敌方巡飞弹被击毁概率计算
        :param state: 我方巡飞弹的状态空间 np.array([self.uav_num, 16])
        :param attack_idx: 我方巡飞弹选择攻击敌方巡飞弹的序号 np.array([self.uav_num])
        :return:hit_pro， 敌方巡飞弹被击毁概率 np.array([self.target_num])
        """
        hit_pro = np.zeros([self.target_num])
        alive_pro = np.ones([self.target_num])  # 敌方巡飞弹存活概率
        for i, j in zip(range(len(self.uav_ind)), range(len(self.uav_ind))):
            uav = self.uav_ind[i]
            tar = self.tar_ind[j]
            if tar >= 0:  # 判断是否有目标
                self.s_d[uav] = self.calc_dis(state[uav][0], state[uav][1], t_state[tar][0], t_state[tar][1])
                self.s_a[uav] = self.calc_angle(state[uav][0], state[uav][1], state[uav][2], state[uav][3],
                                                t_state[tar][0], t_state[tar][1])
                # 使用本时刻敌机的位置和角度
                # 当我方巡飞弹uav和敌方巡飞弹tar距离小于self.attack_dis时，计算
                if self.s_d[uav] <= self.attack_dis - 1 and abs(self.s_a[uav]) <= self.attack_angle:  # self.attack_dis
                    temp_d = self.attack_dis - self.s_d[uav]
                    if self.s_d[uav] <= self.kill_dis:
                        beta = 1
                        alive_pro[tar] = 0
                        hit_pro[tar] = beta * (1 - alive_pro[tar])
                    # 计算敌方巡飞弹存活概率
                    # 计算衰减概率
                    else:
                        beta = (self.s_d[i] - self.kill_dis) / (self.attack_dis - self.kill_dis)
                        alive_pro[tar] *= 1 - (math.log10(temp_d) * math.cos(self.s_a[uav]))
                        hit_pro[tar] = beta * (1 - alive_pro[tar])

        return hit_pro

    # 初始化
    def reset(self):
        # 信息初始化
        state, reward, done, state_, temp_done = [], [], [], [], []
        t_state, t_reward, t_done, t_state_, temp_done_tar = [], [], [], [], []
        for i in range(self.uav_num):
            state.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            reward.append(0)
            done.append(False)
            temp_done.append(False)
            state_.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.state = np.array(state, dtype=np.float32)
        self.reward = np.array(reward, dtype=np.float32)
        self.done = np.array(done, dtype=np.bool_)
        self.state_ = np.array(state_, dtype=np.float32)
        self.temp_done = np.array(temp_done, dtype=np.bool_)
        for i in range(self.target_num):
            t_state.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            t_reward.append(0)
            t_done.append(False)
            temp_done_tar.append(False)
            t_state_.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.t_state = np.array(t_state, dtype=np.float32)
        self.t_reward = np.array(t_reward, dtype=np.float32)
        self.t_done = np.array(t_done, dtype=np.bool_)
        self.t_state_ = np.array(t_state_, dtype=np.float32)
        self.temp_done_tar = np.array(temp_done_tar, dtype=np.bool_)

        self.area = random.randint(1, 3)
        self.attack_idx = np.ones([self.uav_num], dtype=int) * (-1)
        self.flag = [False for _ in range(self.target_num)]
        self.kill_flag = np.zeros(self.target_num)
        self.step_num = 0
        self.hit_pro = np.zeros([self.target_num])

        # 初始化集结区中心坐标
        self.gather_x = 400
        self.gather_y = 100  # random.uniform(0.1 * self.height, 0.4 * self.height)
        self.gather_r = random.uniform(self.gather_r_min, self.gather_r_max)

        # 生成禁飞区圆
        if self.obstacle_or_not:
            self.obstacle = self.generate_circle(self.obstacle_r_min, self.obstacle_r_max, self.obstacle_num,
                                                 self.gather_x, self.gather_y, self.gather_r)
        # 初始化敌方巡飞弹
        t_state = []
        for target in range(self.target_num):
            t_state.append(self.generate_t_uav())
            self.target_x[target] = t_state[target][0]
            self.target_y[target] = t_state[target][1]
            self.target_v[target] = t_state[target][2]
            self.target_alpha[target] = t_state[target][3]
        # 初始化我方巡飞弹
        state = []
        temp_uav = np.zeros([self.uav_num, 2])
        for i in range(self.uav_num):
            state.append(self.generate_uav())
        # 进行目标评估分配
        self.situation_assessment(state, t_state)
        uav_id = self.uav_ind.copy()
        tar_id = self.tar_ind.copy()
        for i in range(self.target_num):
            tar = [self.gather_x, self.gather_y]
            temp_d = self.calc_dis(t_state[i][0], t_state[i][1], tar[0], tar[1])
            temp_beta = self.calc_angle(t_state[i][0], t_state[i][1], t_state[i][2], t_state[i][3],
                                        tar[0], tar[1])
            t_state[i][4] = temp_d
            t_state[i][5] = temp_beta
        for i in range(self.uav_num):
            if i in uav_id:  # 分配完成,有攻击目标
                ind = uav_id.index(i)
                tar_ind = tar_id[ind]
                self.attack_idx[i] = tar_ind
                if tar_ind >= 0:
                    tar = [self.t_state[tar_ind][0], self.t_state[tar_ind][1]]
                else:
                    tar = [self.gather_x, self.gather_y]  # 不存在敌机目标，返航
            else:
                tar = [self.gather_x, self.gather_y]
            # 计算巡飞弹到目标的距离和角度
            # print(tar)
            temp_d = self.calc_dis(state[i][0], state[i][1], tar[0], tar[1])
            temp_beta = self.calc_angle(state[i][0], state[i][1], state[i][2], state[i][3],
                                        tar[0], tar[1])
            state[i][4] = temp_d
            state[i][5] = temp_beta
            temp_uav[i] = [state[i][0], state[i][1]]  # 暂时没用
        self.state = np.array(state, dtype=np.float32)
        self.t_state = np.array(t_state, dtype=np.float32)
        # 计算雷达探测的信息
        self.state = self.calc_probe_information(self.state)
        self.t_state = self.calc_t_probe_information(self.t_state, self.state)
        # 清除上一轮的画图数据
        for i in range(self.uav_num):
            self.uav_x[i].clear()
            self.uav_y[i].clear()
        for target in range(self.target_num):
            self.t_x[target].clear()
            self.t_y[target].clear()
        # 存储画图数据
        for i in range(self.uav_num):
            # 我方巡飞弹位置
            self.uav_x[i].append(self.state[i][0])
            self.uav_y[i].append(self.state[i][1])
        for target in range(self.target_num):
            self.t_x[target].append(self.target_x[target])
            self.t_y[target].append(self.target_y[target])

        # 状态归一化处理
        self.state = self.state_normalization(self.state, True)
        self.t_state = self.state_normalization(self.t_state, True)
        # 我机与敌机本时刻的距离和角度
        self.s_d = np.ones([self.uav_num]) * 100
        self.s_a = np.ones([self.uav_num]) * 100  # 初始范围外值
        # 指标
        self.success = 0  # 突防任务成功次数
        self.uav_live = self.uav_num  # 我方巡飞弹突防成功数量
        self.tar_live = self.target_num  # 敌方巡飞弹存活数量

        return

    # 执行动作
    def step(self, action, t_action):  # 输入的action时归一化后的
        # 对action进行处理，去归一化
        self.hit_pro = np.zeros([self.target_num])
        self.action = self.action_normalization(action, False)
        self.t_action = self.action_normalization(t_action, False)
        # 对state进行处理，去归一化
        self.state = self.state_normalization(self.state, False)
        self.t_state = self.state_normalization(self.t_state, False)
        if self.state_.sum() != 0:
            self.state_ = self.state_normalization(self.state_, False)
            self.t_state_ = self.state_normalization(self.t_state_, False)
            # 将state_赋给state，开始新一步仿真，更新状态
            for i in range(self.uav_num):
                state_ = self.state_[i]
                self.state[i] = state_
            for tar in range(self.target_num):
                # print("update")
                t_state_ = self.t_state_[tar]
                self.t_state[tar] = t_state_

        # 从state中读取巡飞弹信息
        x = np.zeros([self.uav_num, ])
        y = np.zeros([self.uav_num, ])
        v = np.zeros([self.uav_num, ])
        alpha = np.zeros([self.uav_num, ])
        for i in range(self.uav_num):
            x[i] = (self.state[i][0])
            y[i] = (self.state[i][1])
            v[i] = (self.state[i][2])
            alpha[i] = (self.state[i][3])
        # 采取动作，计算巡飞弹新状态信息
        x_ = np.zeros([self.uav_num, ])
        y_ = np.zeros([self.uav_num, ])
        v_ = np.zeros([self.uav_num, ])
        alpha_ = np.zeros([self.uav_num, ])

        t_x = np.zeros([self.target_num, ])
        t_y = np.zeros([self.target_num, ])
        t_v = np.zeros([self.target_num, ])
        t_alpha = np.zeros([self.target_num, ])

        for i in range(self.target_num):
            t_x[i] = (self.t_state[i][0])
            t_y[i] = (self.t_state[i][1])
            t_v[i] = (self.t_state[i][2])
            t_alpha[i] = (self.t_state[i][3])
        # 采取动作，计算巡飞弹新状态信息
        t_x_ = np.zeros([self.target_num, ])
        t_y_ = np.zeros([self.target_num, ])
        t_v_ = np.zeros([self.target_num, ])
        t_alpha_ = np.zeros([self.target_num, ])

        # 我方巡飞弹运动计算
        for i in range(self.uav_num):
            # 若该巡飞弹到达集结点，停止运动
            if self.done[i]:
                continue
            # 巡飞弹新速度方向，与x轴之间的夹角，范围在0~2*pi
            alpha_[i] = (alpha[i] + self.action[i][1] * self.t)
            if alpha_[i] < 0:
                alpha_[i] = 2 * math.pi + alpha_[i]
            elif alpha_[i] >= 2 * math.pi:
                alpha_[i] = alpha_[i] - 2 * math.pi
            # 巡飞弹新速度大小，范围在-2.5~2.5
            v_[i] = (v[i] + self.action[i][0] * self.t)
            if v_[i] < self.uav_v_min:
                v_[i] = self.uav_v_min
            elif v_[i] > self.uav_v_max:
                v_[i] = self.uav_v_max
            # 巡飞弹新x，y坐标
            x_[i] = (x[i] + v_[i] * math.cos(alpha_[i]) * self.t)
            y_[i] = (y[i] + v_[i] * math.sin(alpha_[i]) * self.t)

            # 更新state_值
            # print(self.state[0])
            self.state_[i] = self.state[i]
            self.state_[i][0] = x_[i]
            self.state_[i][1] = y_[i]
            self.state_[i][2] = v_[i]
            self.state_[i][3] = alpha_[i]
            # 存储画图数据
            self.uav_x[i].append(self.state_[i][0])
            self.uav_y[i].append(self.state_[i][1])
        # 敌方巡飞弹运动计算
        for i in range(self.target_num):
            if self.t_done[i]:
                continue
            t_alpha_[i] = (t_alpha[i] + self.t_action[i][1] * self.t)
            if t_alpha_[i] < 0:
                t_alpha_[i] = 2 * math.pi + t_alpha_[i]
            elif t_alpha_[i] >= 2 * math.pi:
                t_alpha_[i] = t_alpha_[i] - 2 * math.pi
            # 巡飞弹新速度大小，范围在-2.5~2.5
            t_v_[i] = (t_v[i] + self.t_action[i][0] * self.t)
            if t_v_[i] < self.tar_v_min:
                t_v_[i] = self.tar_v_min
            elif t_v_[i] > self.tar_v_max:
                t_v_[i] = self.tar_v_max
            # 巡飞弹新x，y坐标
            t_x_[i] = (t_x[i] + t_v_[i] * math.cos(t_alpha_[i]) * self.t)
            t_y_[i] = (t_y[i] + t_v_[i] * math.sin(t_alpha_[i]) * self.t)

            # 更新state_值
            # print(self.t_state[0])
            self.t_state_[i] = self.t_state[i]
            self.t_state_[i][0] = t_x_[i]
            self.t_state_[i][1] = t_y_[i]
            self.t_state_[i][2] = t_v_[i]
            self.t_state_[i][3] = t_alpha_[i]
            # 存储画图数据
            self.t_x[i].append(self.t_state_[i][0])
            self.t_y[i].append(self.t_state_[i][1])

        # 其余状态维度更新
        # 进行目标评估、分配
        self.attack_idx = np.ones([self.uav_num], dtype=int) * (-1)
        uav_id, tar_id = self.situation_assessment(self.state_, self.t_state_)
        # print("uav_id:",uav_id)
        # print("tar_id:", tar_id)
        # 更新目标位置
        for i in range(self.target_num):
            tar = [self.gather_x, self.gather_y]
            temp_d = self.calc_dis(self.t_state_[i][0], self.t_state_[i][1], tar[0], tar[1])
            temp_beta = self.calc_angle(self.t_state_[i][0], self.t_state_[i][1], self.t_state_[i][2],
                                        self.t_state_[i][3],
                                        tar[0], tar[1])
            self.t_state_[i][4] = temp_d
            self.t_state_[i][5] = temp_beta
        for i in range(self.uav_num):
            if i in uav_id:  # 无人机坠毁
                ind = uav_id.index(i)
                tar_ind = tar_id[ind]
                # 更新攻击目标
                self.attack_idx[i] = tar_ind
                if tar_ind >= 0:
                    tar = [self.t_state_[tar_ind][0] + self.t_state_[tar_ind][2] * math.cos(
                        self.t_state_[tar_ind][3]) * self.t,
                           self.t_state_[tar_ind][1] + self.t_state_[tar_ind][2] * math.sin(
                               self.t_state_[tar_ind][3]) * self.t]
                    tar1 = [self.t_state_[tar_ind][0], self.t_state_[tar_ind][1]]
                    # 同时将我机与敌机本时刻的位置和夹角保存，用于命中概率和结束计算
                    self.s_d[i] = self.calc_dis(x_[i], y_[i], tar1[0], tar1[1])
                    self.s_a[i] = self.calc_angle(x_[i], y_[i], v_[i], alpha_[i], tar1[0], tar1[1])
                # else:
                #     print("mo target")
                #     tar = [self.gather_x, self.gather_y]
                #     # 无攻击目标
                #     self.s_d[i] = 100
                #     self.s_a[i] = 100
            else:
                # print("no target")
                tar = [self.gather_x, self.gather_y]
                # 无攻击目标
                self.s_d[i] = 100
                self.s_a[i] = 100
            # 计算巡飞弹到目标的距离和角度
            temp_d = self.calc_dis(self.state_[i][0], self.state_[i][1], tar[0], tar[1])
            temp_beta = self.calc_angle(self.state_[i][0], self.state_[i][1], self.state_[i][2],
                                        self.state_[i][3],
                                        tar[0], tar[1])
            self.state_[i][4] = temp_d
            self.state_[i][5] = temp_beta

        # 计算巡飞弹雷达探测的信息
        self.state_ = self.calc_probe_information(self.state_)
        self.t_state_ = self.calc_t_probe_information(self.t_state_, self.state_)
        # 计算敌方巡飞弹被击毁的概率，此时要用本时刻敌机的位置和角度
        self.hit_pro = self.cal_hit_probability(self.state_, self.t_state_)
        # 计算哪些敌方巡飞弹被击毁
        for i in range(self.target_num):
            temp_pro = random.uniform(0, 1)
            if temp_pro <= self.hit_pro[i]:
                self.temp_done_tar[i] = True
                # ind = self.tar_ind.index(i)
                # tar = self.uav_ind[ind]
                # self.done[tar] = True  # 同归于尽
                # print("uav", tar)
        # 计算奖励值
        self.get_reward(self.state, self.state_)  # 得到了self.reward和self.done
        self.get_t_reward(self.t_state, self.t_state_)
        # 状态归一化
        # print(self.state[0][4])
        self.state = self.state_normalization(self.state, True)
        self.state_ = self.state_normalization(self.state_, True)
        self.action = self.action_normalization(self.action, True)

        self.t_state = self.state_normalization(self.t_state, True)
        self.t_state_ = self.state_normalization(self.t_state_, True)
        self.t_action = self.action_normalization(self.t_action, True)

        # 计数
        self.step_num += 1

        return self.state, self.reward, self.done, self.state_, self.t_state, self.t_reward, self.t_done, self.t_state_

    # 计算敌方巡飞弹奖励值
    def get_t_reward(self, t_state, t_state_):
        # 我方巡飞弹奖励
        for i in range(self.target_num):
            # 定义单步奖励
            # 有巡飞弹停止，该巡飞弹不获得奖励值
            if self.t_done[i]:
                self.t_reward[i] = 0
                continue
            # 距离奖励
            self.t_reward[i] = (1 - t_state_[i][4] / t_state[i][4])
            # 角度奖励
            self.t_reward[i] += math.cos(t_state_[i][5])
            # 障碍物奖励
            for j in range(self.prob_max_num):
                # 当间距小于20时，开始给予惩罚
                if self.attack_flag[i]:
                    if t_state_[i][6 + 2 * j] < 2 * self.uav_r:
                        self.t_reward[i] += -20
                        self.t_done[i] = True
                        self.tar_live -= 1
                        # print('巡飞弹', i, '撞')
                    elif t_state_[i][6 + 2 * j] < 10:
                        self.t_reward[i] += (1 - t_state[i][6 + 2 * j] / t_state_[i][6 + 2 * j]) * 3
                    elif 10 < t_state_[i][6 + 2 * j] < 20:
                        self.t_reward[i] += (1 - t_state[i][6 + 2 * j] / t_state_[i][6 + 2 * j]) * 2.5
                else:
                    if t_state_[i][6 + 2 * j] < 2 * self.uav_r:
                        self.t_reward[i] += -20
                        self.t_done[i] = True
                        self.tar_live -= 1
                        # print('巡飞弹', i, '撞')
                    elif t_state_[i][6 + 2 * j] < 7.5:
                        self.t_reward[i] += (1 - t_state[i][6 + 2 * j] / t_state_[i][6 + 2 * j]) * 2.5
            # 长期惩罚
            self.t_reward[i] += -0.3
            # 巡飞弹出界
            if t_state_[i][0] <= self.uav_r or t_state_[i][0] >= self.width - self.uav_r or \
                    t_state_[i][1] <= self.uav_r or t_state_[i][1] >= self.height + self.uav_r:
                self.t_reward[i] += -20
                self.t_done[i] = True
                self.tar_live -= 1
            # 巡飞弹被击落
            if not self.t_done[i]:
                if self.temp_done_tar[i]:
                    self.t_reward[i] += -15
                    self.t_done[i] = True
                    self.tar_live -= 1
            # 集结奖励
            if self.t_state_[i][4] + self.uav_r <= self.gather_r:
                # print("tar:", i)
                self.t_reward[i] += 20
                self.t_done[i] = True

        return

    # 计算我方巡飞弹奖励值
    def get_reward(self, state, state_):
        # 我方巡飞弹奖励
        for i in range(self.uav_num):
            # 定义单步奖励
            # 有巡飞弹停止，该巡飞弹不获得奖励值
            if self.done[i]:
                self.reward[i] = 0
                continue
            # 距离奖励
            # 如果目标进入攻击范围，此时奖励与击毁概率有关
            # 若存在攻击目标
            # if i in self.uav_ind:
            #     # 距离奖励
            #     ind = self.uav_ind.index(i)
            #     tar = self.tar_ind[ind]
            #     if state_[i][4] < self.attack_dis and abs(state_[i][5]) <= self.attack_angle:
            #         self.reward[i] = (1 - state_[i][4] / state[i][4]) * self.hit_pro[tar]
            #     else:
            #         self.reward[i] = (1 - state_[i][4] / state[i][4])
            #     # 角度奖励
            #     if abs(state_[i][5]) <= math.pi / 6:
            #         if state_[i][4] < self.attack_dis and abs(state_[i][5]) <= self.attack_angle:
            #             self.reward[i] += math.cos(state_[i][5]) * self.hit_pro[tar]
            #         else:
            #             self.reward[i] += math.cos(state_[i][5])
            #     else:
            #         if state_[i][4] < self.attack_dis and abs(state_[i][5]) <= self.attack_angle:
            #             self.reward[i] += math.cos(state_[i][5]) * self.hit_pro[tar]
            #         else:
            #             self.reward[i] += math.cos(state_[i][5])  # * 0.3
            #     # 命中奖励
            #     # 我方巡飞弹捕获敌方巡飞弹，此时也是用本时刻的敌机
            #     if self.s_d[i] < self.attack_dis and self.t_done[tar]:
            #         # 避免重复击落该敌方巡飞弹
            #         if self.kill_flag[tar] == 0:
            #             self.reward[i] += 20
            #             self.kill_flag[tar] = 1
            # else:
            #     # 距离奖励
            #     self.reward[i] = (1 - state_[i][4] / state[i][4])
            #     # 角度奖励
            #     self.reward[i] += math.cos(state_[i][5])

            ind = self.uav_ind.index(i)
            tar = self.tar_ind[ind]
            # 距离奖励
            self.reward[i] = (1 - self.state_[i][4] / self.state[i][4])
            # 根据无人机与探测障碍物之间距离获得距离奖励
            for j in range(self.prob_max_num):
                # 当间距小于20时，开始给予惩罚
                if state_[i][6 + 2 * j] < 2 * self.uav_r:
                    self.reward[i] += -5
                    self.done[i] = True
                    self.uav_live -= 1
                    # print('无人机', i, '撞')
                elif state_[i][6 + 2 * j] < 10:
                    self.reward[i] += (1 - self.state[i][6 + 2 * j] / self.state_[i][6 + 2 * j]) * 2.5 / (j + 1)
            # 角度奖励
            # 计算我机相对攻击敌机的占位角度
            alpha = self.t_state_[tar][3] + math.pi
            if alpha > 2 * math.pi:
                alpha += -2 * math.pi
            theta = self.calc_angle(self.t_state_[tar][0],
                                    self.t_state_[tar][1],
                                    self.t_state_[tar][2], alpha,
                                    self.state_[i][0], self.state_[i][1])
            self.reward[i] += math.cos(state_[i][5]) * 0.3 * (1 - abs(theta) / (math.pi + 1))
            # 敌方无人机进入我方无人机攻击区
            if state_[i][4] <= self.attack_dis and abs(state_[i][5]) <= self.attack_angle:
                self.reward[i] += 0.1
            if self.state_[i][4] < self.attack_dis and self.temp_done_tar[tar]:
                # 避免重复击落该敌方无人机
                if self.kill_flag[tar] == 0:
                    self.reward[i] += 20
                    self.kill_flag[tar] = 1
                    self.done[i] = True
                    self.success += 1
                    print("uav", i)
            # 长期惩罚
            self.reward[i] += -0.3
            # 巡飞弹出界
            if state_[i][0] <= self.uav_r or state_[i][0] >= self.width - self.uav_r or \
                    state_[i][1] <= self.uav_r or state_[i][1] >= self.height + self.uav_r:
                self.reward[i] += -20
                self.done[i] = True
                self.uav_live -= 1

            # 全部击落敌方巡飞弹奖励
            if self.success == self.target_num and self.flag:
                for j in range(self.uav_num):
                    self.reward[j] += 100
                self.flag = False

        return

    # 一个list，存储点用于画出巡飞弹形状
    def cal_points(self, x, y, alpha):
        points = [(x, y),
                  (x - math.sqrt(6 * 6 * 2) * math.cos(math.pi / 4 + alpha),
                   y - math.sqrt(6 * 6 * 2) * math.sin(math.pi / 4 + alpha)),
                  (x - math.sqrt(6 * 6 * 2) * math.cos(math.pi / 4 + alpha) + 3.5 * math.cos(-alpha),
                   y - math.sqrt(6 * 6 * 2) * math.sin(math.pi / 4 + alpha) - 3.5 * math.sin(-alpha)),
                  (x + 3 * math.cos(-alpha), y - 5 * math.sin(-alpha)),
                  (x - math.sqrt(6 * 6 * 2) * math.sin(math.pi / 4 + alpha) + 3.5 * math.cos(-alpha),
                   y + math.sqrt(6 * 6 * 2) * math.cos(math.pi / 4 + alpha) - 3.5 * math.sin(-alpha)),
                  (x - math.sqrt(6 * 6 * 2) * math.sin(math.pi / 4 + alpha),
                   y + math.sqrt(6 * 6 * 2) * math.cos(math.pi / 4 + alpha))
                  ]
        return points

    def render(self, mode='human'):
        import pygame
        if self.screen is None:
            # 初始化
            pygame.init()
            pygame.display.init()
            # 创建主屏幕
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('巡飞弹集群突防任务')
        if self.clock is None:
            # 创建时钟对象（控制游戏的FPS）
            self.clock = pygame.time.Clock()
        # 创建显示
        self.surf = pygame.Surface((self.width, self.height))
        self.surf.fill('white')  # 背景白色
        # 画集结区
        pygame.draw.circle(self.surf, 'green', (self.gather_x, self.gather_y), self.gather_r)
        # 画出禁飞区
        if self.obstacle_or_not:
            for circle in self.obstacle:
                pygame.draw.circle(self.surf, 'red', (circle[0], circle[1]), circle[2])
        # 画出巡飞弹
        # 我方巡飞弹位置和角度
        x, y, alpha = [], [], []
        state = self.state_normalization(self.state, False)  # 去归一化
        for i in range(self.uav_num):
            x.append(state[i][0])
            y.append(state[i][1])
            alpha.append(state[i][3])
        # 我方巡飞弹运动
        for i in range(self.uav_num):
            pygame.draw.polygon(self.surf, 'blue', self.cal_points(x[i], y[i], alpha[i]))
            pygame.draw.arc(self.surf, 'blue',
                            (x[i] - self.attack_dis, y[i] - self.attack_dis,
                             2 * self.attack_dis, 2 * self.attack_dis),
                            -alpha[i] - self.attack_angle, -alpha[i] + self.attack_angle,
                            self.attack_dis)
        # 翻转屏幕，screen的（0,0）在左上角，正常画图(0,0)在左下角
        # 我方巡飞弹位置和角度
        t_x, t_y, t_alpha = [], [], []
        t_state = self.state_normalization(self.t_state, False)  # 去归一化
        for i in range(self.target_num):
            t_x.append(t_state[i][0])
            t_y.append(t_state[i][1])
            t_alpha.append(t_state[i][3])
        # 我方巡飞弹运动
        for i in range(self.target_num):
            pygame.draw.polygon(self.surf, 'red', self.cal_points(t_x[i], t_y[i], t_alpha[i]))
            pygame.draw.arc(self.surf, 'red',
                            (t_x[i] - self.attack_dis, t_y[i] - self.attack_dis,
                             2 * self.attack_dis, 2 * self.attack_dis),
                            -t_alpha[i] - self.attack_angle, -t_alpha[i] + self.attack_angle,
                            self.attack_dis)
        self.surf = pygame.transform.flip(self.surf, False, True)
        # 将绘制的图像添加到主屏幕上
        self.screen.blit(self.surf, (0, 0))
        if mode == 'human':
            pygame.event.pump()  # event事件处理器
            self.clock.tick(self.fps)
        pygame.display.flip()  # 更新屏幕内容
        return

    # 画出巡飞弹轨迹图
    def draw(self, episode):
        plt.rcParams["font.family"] = "FangSong"  # 支持中文显示
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim=[0, self.width], ylim=[0, self.height], title='巡飞弹集群突防任务',
               ylabel='Y方向', xlabel='X方向')
        # 逃逸巡飞弹
        self.t_state = self.state_normalization(self.t_state, False)
        for tar in range(self.target_num):
            if self.t_done[tar]:
                plt.plot(self.t_x[tar], self.t_y[tar], 'grey')
                agent_points = Polygon(self.cal_points(
                    self.t_x[tar][-1], self.t_y[tar][-1], self.t_state[tar][3]), color='grey')
            else:
                plt.plot(self.t_x[tar], self.t_y[tar], 'r')
                agent_points = Polygon(self.cal_points(
                    self.t_x[tar][-1], self.t_y[tar][-1], self.t_state[tar][3]), color='r')
            ax.add_patch(agent_points)
        self.t_state = self.state_normalization(self.t_state, True)
        gather = Circle((self.gather_x, self.gather_y), self.gather_r, color='green')
        ax.add_patch(gather)
        # 禁飞区
        if self.obstacle_or_not:
            for circle in self.obstacle:
                obstacle = Circle((circle[0], circle[1]), circle[2], color='red')
                ax.add_patch(obstacle)
        # 我方巡飞弹轨迹
        self.state = self.state_normalization(self.state, False)
        for i in range(self.uav_num):
            plt.plot(self.uav_x[i], self.uav_y[i], 'b')
            agent_points = Polygon(self.cal_points(
                self.uav_x[i][-1], self.uav_y[i][-1], self.state[i][3]), color='b')
            ax.add_patch(agent_points)
        self.state = self.state_normalization(self.state, True)
        plt.savefig('D:/code/UAV_Pursuit/UAV_Pursuit/GL_MFDDPG/图{}.png'.format(episode), dpi=300)
        plt.show()

    def draw_gif(self, episode):
        plt.rcParams["font.family"] = "FangSong"  # 支持中文显示
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim=[0, self.width], ylim=[0, self.height], title='巡飞弹集群突防任务',
               ylabel='Y方向', xlabel='X方向')
        # 集结区
        gather = Circle((self.gather_x, self.gather_y), self.gather_r, color='green')
        ax.add_patch(gather)
        # 禁飞区
        if self.obstacle_or_not:
            for circle in self.obstacle:
                obstacle = Circle((circle[0], circle[1]), circle[2], color='red')
                ax.add_patch(obstacle)
        # 敌我方巡飞弹轨迹
        data = [[] for _ in range(self.uav_num)]
        for i in range(self.uav_num):
            for j in range(len(self.uav_x[i])):
                data[i].append([self.uav_x[i][j], self.uav_y[i][j]])
        lines = [ax.plot([], [], "b")[0] for _ in range(self.uav_num)]
        lines1 = [ax.plot([], [], "r")[0] for _ in range(self.target_num)]
        points = [ax.plot([], [], "o", color="b")[0] for _ in range(self.uav_num)]
        points1 = [ax.plot([], [], "o", color="r")[0] for _ in range(self.target_num)]

        # print(len(self.uav_x[0]))

        def update_lines(num, walks, lines, lines1, points, points1):
            for i in range(len(walks)):
                # 处理无人机轨迹和点print("num:",num)
                for line, point, j in zip(lines, points, range(self.uav_num)):
                    if num <= len(self.uav_x[j]):
                        line.set_data(self.uav_x[j][:num], self.uav_y[j][:num])
                        point.set_data([self.uav_x[j][num - 1]], [self.uav_y[j][num - 1]])  # 已修正
                    else:
                        line.set_data(self.uav_x[j][:], self.uav_y[j][:])
                        # 修正：将单个数值转为列表（序列）
                        point.set_data([self.uav_x[j][-1]], [self.uav_y[j][-1]])

                # 处理目标轨迹和点
                for line1, point1, k in zip(lines1, points1, range(self.target_num)):
                    if num <= len(self.t_x[k]):
                        line1.set_data(self.t_x[k][:num], self.t_y[k][:num])
                        point1.set_data([self.t_x[k][num - 1]], [self.t_y[k][num - 1]])  # 已修正
                    else:
                        line1.set_data(self.t_x[k][:], self.t_y[k][:])
                        point1.set_data([self.t_x[k][-1]], [self.t_y[k][-1]])
            return lines, lines1, points, points1

        max1 = 0
        max2 = 0
        for i in range(self.uav_num):
            max1 = max(max1, len(self.uav_x[i]))
        for i in range(self.target_num):
            max2 = max(max2, len(self.t_x[i]))
        frame = max(max1, max2)

        ani = FuncAnimation(fig, update_lines, frames=frame, fargs=(data, lines, lines1, points, points1),
                            interval=20)
        ani.save('E:/pyPractice/UAV_Pursuit/GL_MFDDPG/动图shi{}.gif'.format(episode), writer='pillow',
                 fps=1000)
        plt.show()

# uav = UAVPursuit()
# uav.reset()
# R = 0
# for i in range(uav.round_max):
#     # print(uav.t_done)
#     # +逆-顺
#     a0 = []
#     a1 = []
#     for j in range(uav.uav_num):
#         a0.append([random.uniform(-1, 1), random.uniform(-1, 1)])
#         a1.append([random.uniform(-1, 1), random.uniform(-1, 1)])
#     s, r, d, s_, t_s, t_r, t_d, t_s_ = uav.step(a0, a1)
#     # print(t_d)
#     # uav.render()
#
#     # print(uav.state_)
#     # print(uav.t_state)
#     # print(uav.t_done)
#     # print('reward:', uav.reward, '\n')
#     t_s = uav.state_normalization(t_s, False)
#     s = uav.state_normalization(s, False)
#
#     # t_s = uav.state_normalization(t_s, True)
#     # print(s[10][4])
#     R += uav.reward.sum()
#
#     if uav.t_done.sum() == uav.target_num or uav.done.sum() == uav.uav_num:
#         # print(uav.t_done)
#         # print(uav.done)
#         # print(uav.target_num)
#         # print("end")
#         break
# print(uav.t_x, '\n', uav.t_y)
# uav.draw(1)
