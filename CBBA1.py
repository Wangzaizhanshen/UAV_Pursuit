import numpy as np
import copy


class CBBA_agent:
    def __init__(self, agent_state, target_state, threat_list, id=None, vel=None, task_num=None, agent_num=None,
                 L_t=None):

        self.task_num = task_num
        self.agent_num = agent_num

        # Agent information
        self.id = id  # 智能体ID 理解为索引
        self.vel = vel  # 智能体速度

        # Local Winning Agent List
        self.z = np.ones(self.task_num, dtype=np.int8) * self.id  # 获胜智能体列表
        # Local Winning Bid List
        self.y = np.array([0 for _ in range(self.task_num)], dtype=np.float64)  # 获胜出价列表
        # Bundle
        self.b = []  # 任务束
        # Path
        self.p = []  # 路径
        # Maximum Task Number #单个智能体最大执行任务数
        self.L_t = L_t
        # Local Clock 时钟
        self.time_step = 0
        # Time Stamp List 每个智能体的时间戳列表
        self.s = {a: self.time_step for a in range(self.agent_num)}

        # This part can be modified depend on the problem 智能体状态
        self.state = np.array(agent_state)  # Agent State (Position)
        self.tar_state = target_state
        self.c = np.zeros(self.task_num)  # Initial Score (Euclidean Distance) 初始分数

        # socre function parameters 得分函数参数
        self.Lambda = 0.95
        self.c_bar = threat_list  # 执行任务j的静态评分

    def tau(self, j):
        # Estimate time agent will take to arrive at task j's location 估计智能体到达任务j的位置所需的时间
        # This function can be used in later
        pass

    def set_state(self, state):
        """
    Set state of agent
    """
        self.state = state

    def send_message(self):  # 发送获胜出价列表、获胜智能体列表、时间戳列表
        """
    Return local winning bid list
    [output]
    y: winning bid list (list:task_num)
    z: winning agent list (list:task_num)
    s: Time Stamp List (Dict:{agent_id:update_time})
    """
        return self.y.tolist(), self.z.tolist(), self.s

    def receive_message(self, Y):  # 接收信息
        self.Y = Y

    def build_bundle(self):  # 构建任务束，构建b、p
        """
    Construct bundle and path list with local information
    """
        J = [j for j in range(self.task_num)]  # 任务索引

        while len(self.b) < self.L_t:  # 构建每个智能体的任务束
            # Calculate S_p for constructed path list 计算sp构建路径列表
            S_p = 0
            # 计算时间折扣奖励
            if len(self.p) > 0:
                distance_j = 0
                distance_j += np.linalg.norm(self.state.squeeze() - self.tar_state[self.p[0]])  # 计算第一步距离
                S_p += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[self.p[0]]  # 计算第一步
                for p_idx in range(len(self.p) - 1):
                    distance_j += np.linalg.norm(self.tar_state[self.p[p_idx]] - self.tar_state[self.p[p_idx + 1]])
                    S_p += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[self.p[p_idx + 1]]

            # Calculate c_ij for each task j 计算任务得分
            best_pos = {}
            for j in J:
                c_list = []
                if j in self.b:  # If already in bundle list 以分配，再分配其他任务
                    self.c[j] = 0  # Minimum Score
                else:
                    for n in range(len(self.p) + 1):
                        p_temp = copy.deepcopy(self.p)
                        p_temp.insert(n, j)  # 插入任务
                        c_temp = 0
                        distance_j = 0
                        distance_j += np.linalg.norm(self.state.squeeze() - self.tar_state[p_temp[0]])  # 计算插入后的路径
                        c_temp += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[p_temp[0]]
                        if len(p_temp) > 1:
                            for p_loc in range(len(p_temp) - 1):
                                distance_j += np.linalg.norm(
                                    self.tar_state[p_temp[p_loc]] - self.tar_state[p_temp[p_loc + 1]])
                                c_temp += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[p_temp[p_loc + 1]]

                        c_jn = c_temp - S_p
                        c_list.append(c_jn)  # 计算每个任务的实际得分

                    max_idx = np.argmax(c_list)  # 计算插入不同位置后，奖励变化，取最大值作为任务j的实际得分
                    c_j = c_list[max_idx]
                    self.c[j] = c_j
                    best_pos[j] = max_idx

            h = (self.c > self.y)  # 任务得分大于获胜报价，为1
            if sum(h) == 0:  # No valid task 没有有效的任务
                break
            self.c[~h] = 0  # 无效任务得分赋零
            J_i = np.argmax(self.c)
            n_J = best_pos[J_i]

            # 最终输出
            self.b.append(J_i)  # 添加任务序号
            self.p.insert(n_J, J_i)  # 最佳插入位置

            self.y[J_i] = self.c[J_i]
            self.z[J_i] = self.id

    def update_task(self):  # 更新任务，输入邻居的ID，y,z,s
        """
    [input] 邻居信息
    Y: winning bid lists from neighbors (dict:{neighbor_id:(winning bid_list, winning agent list, time stamp list)})
    time: for simulation,
    """

        old_p = copy.deepcopy(self.p)

        id_list = list(self.Y.keys())  # dict:{neighbor_id:(winning bid_list, winning agent list, time stamp list 时间戳)
        id_list.insert(0, self.id)  # 加入自己ID

        # Update time list 更新时间列表
        for id in list(self.s.keys()):
            if id in id_list:  # 是邻居，有通信
                self.s[id] = self.time_step  # 同步时间戳
            else:  # 无通信
                s_list = []
                for neighbor_id in id_list[1:]:
                    s_list.append(self.Y[neighbor_id][2][id])
                if len(s_list) > 0:
                    self.s[id] = max(s_list)  # 取邻居智能体中的最大时间步，进行赋值

        # Update Process 更新y,z
        for j in range(self.task_num):
            for k in id_list[1:]:  # 依次读取邻居智能体信息
                y_k = self.Y[k][0]
                z_k = self.Y[k][1]
                s_k = self.Y[k][2]

                z_ij = self.z[j]  # 本智能体中任务j的获胜智能体
                z_kj = z_k[j]  # 邻居信息中任务j的获胜智能体
                y_kj = y_k[j]  # 邻居信息中任务j的获胜智能体出价

                i = self.id  # 本智能体ID
                y_ij = self.y[j]  # 本智能体中任务j的获胜智能体出价

                # Rule Based Update 基于规则进行更新
                # Rule 1~4
                if z_kj == k:
                    # Rule 1
                    if z_ij == self.id:
                        if y_kj > y_ij:
                            self.__update(j, y_kj, z_kj)
                        elif abs(y_kj - y_ij) < np.finfo(float).eps:  # Tie Breaker 小于极小值
                            if k < self.id:
                                self.__update(j, y_kj, z_kj)
                        else:
                            self.__leave()
                    # Rule 2
                    elif z_ij == k:
                        self.__update(j, y_kj, z_kj)
                    # Rule 3
                    elif z_ij != -1:
                        m = z_ij
                        if (s_k[m] > self.s[m]) or (y_kj > y_ij):
                            self.__update(j, y_kj, z_kj)
                        elif abs(y_kj - y_ij) < np.finfo(float).eps:  # Tie Breaker
                            if k < self.id:
                                self.__update(j, y_kj, z_kj)
                    # Rule 4
                    elif z_ij == -1:
                        self.__update(j, y_kj, z_kj)
                    else:
                        raise Exception("Error while updating")
                # Rule 5~8
                elif z_kj == i:
                    # Rule 5
                    if z_ij == i:
                        self.__leave()
                    # Rule 6
                    elif z_ij == k:  # 任务冲突
                        self.__reset(j)
                    # Rule 7
                    elif z_ij != -1:
                        m = z_ij
                        if s_k[m] > self.s[m]:
                            self.__reset(j)
                    # Rule 8
                    elif z_ij == -1:
                        self.__leave()
                    else:
                        raise Exception("Error while updating")
                # Rule 9~13
                elif z_kj != -1:
                    m = z_kj
                    # Rule 9
                    if z_ij == i:
                        if (s_k[m] >= self.s[m]) and (y_kj > y_ij):
                            self.__update(j, y_kj, z_kj)
                        elif (s_k[m] >= self.s[m]) and (abs(y_kj - y_ij) < np.finfo(float).eps):  # Tie Breaker
                            if m < self.id:
                                self.__update(j, y_kj, z_kj)
                    # Rule 10
                    elif z_ij == k:
                        if (s_k[m] > self.s[m]):
                            self.__update(j, y_kj, z_kj)
                        else:
                            self.__reset(j)
                    # Rule 11
                    elif z_ij == m:
                        if (s_k[m] > self.s[m]):
                            self.__update(j, y_kj, z_kj)
                    # Rule 12
                    elif z_ij != -1:
                        n = z_ij
                        if (s_k[m] > self.s[m]) and (s_k[n] > self.s[n]):
                            self.__update(j, y_kj, z_kj)
                        elif (s_k[m] > self.s[m]) and (y_kj > y_ij):
                            self.__update(j, y_kj, z_kj)
                        elif (s_k[m] > self.s[m]) and (abs(y_kj - y_ij) < np.finfo(float).eps):  # Tie Breaker
                            if m < n:
                                self.__update(j, y_kj, z_kj)
                        elif (s_k[n] > self.s[n]) and (self.s[m] > s_k[m]):
                            self.__update(j, y_kj, z_kj)
                    # Rule 13
                    elif z_ij == -1:
                        if (s_k[m] > self.s[m]):
                            self.__update(j, y_kj, z_kj)
                    else:
                        raise Exception("Error while updating")
                # Rule 14~17
                elif z_kj == -1:
                    # Rule 14
                    if z_ij == i:
                        self.__leave()
                    # Rule 15
                    elif z_ij == k:
                        self.__update(j, y_kj, z_kj)
                    # Rule 16
                    elif z_ij != -1:
                        m = z_ij
                        if s_k[m] > self.s[m]:
                            self.__update(j, y_kj, z_kj)
                    # Rule 17
                    elif z_ij == -1:
                        self.__leave()
                    else:
                        raise Exception("Error while updating")
                else:
                    raise Exception("Error while updating")

        n_bar = len(self.b)  # 任务束长度
        # Get n_bar
        for n in range(len(self.b)):
            b_n = self.b[n]
            if self.z[b_n] != self.id:  # 更新后的获胜智能体不是自己，需要将该任务移除，之后的也要删除
                n_bar = n  # 获取截断索引
                break

        b_idx1 = copy.deepcopy(self.b[n_bar + 1:])  # 要删除的任务序号

        if len(b_idx1) > 0:  # 清除矛盾任务后的y、z
            self.y[b_idx1] = 0
            self.z[b_idx1] = -1

        if n_bar < len(self.b):
            del self.b[n_bar:]  # 删除任务序列

        self.p = []
        for task in self.b:
            self.p.append(task)  # 只保留对应部分的路径

        self.time_step += 1

        converged = False  # 冲突
        if old_p == self.p:
            converged = True  # 不冲突

        return converged

    def __update(self, j, y_kj, z_kj):
        """
    Update values
    """
        self.y[j] = y_kj
        self.z[j] = z_kj

    def __reset(self, j):
        """
    Reset values
    """
        self.y[j] = 0
        self.z[j] = -1  # -1 means "none"

    def __leave(self):
        """
    Do nothing
    """
        pass


def target_assignment():
    np.random.seed(10)

    task_num = 10
    robot_num = 50  # 少对多分配

    task = np.random.uniform(low=0, high=1, size=(task_num, 2))  # 目标状态
    state = np.random.uniform(low=0, high=1, size=(robot_num, 2))  # 目标状态
    the = [np.ones(task_num) for _ in range(robot_num)]  # 执行任务j的静态评分

    # task = np.array([[0,1],[1,1],[1,2]])

    robot_list = [CBBA_agent(state, task, the[i], id=i, vel=1, task_num=task_num, agent_num=robot_num, L_t=1) for i in
                  range(robot_num)]
    # robot_list[0].state = np.array([[0,0]])
    # robot_list[1].state = np.array([[1,0]])

    # Network Initialize
    G = np.ones((robot_num, robot_num))  # Fully connected network 全局通信
    # G[0,1]=0
    # G[1,0]=0

    t = 0  # Iteration number

    while True:
        converged_list = []  # Converged List

        print("==Iteration {}==".format(t))
        ## Phase 1: Auction Process
        print("Auction Process")
        for robot in robot_list:
            # select task by local information
            robot.build_bundle()

        print("Bundle")
        for robot in robot_list:
            print(robot.b)
        print("Path")
        for robot in robot_list:
            print(robot.p)

        ## Communication stage
        print("Communicating...")
        # Send winning bid list to neighbors (depend on env)
        message_pool = [robot.send_message() for robot in robot_list]

        for robot_id, robot in enumerate(robot_list):
            # Recieve winning bidlist from neighbors
            g = G[robot_id]

            connected, = np.where(g == 1)
            connected = list(connected)
            connected.remove(robot_id)

            if len(connected) > 0:
                Y = {neighbor_id: message_pool[neighbor_id] for neighbor_id in connected}  # 构建通信信息
            else:
                Y = None

            robot.receive_message(Y)

        ## Phase 2: Consensus Process
        print("Consensus Process")
        for robot in robot_list:
            # Update local information and decision
            if Y is not None:
                converged = robot.update_task()
                converged_list.append(converged)

        print("Bundle")
        for robot in robot_list:
            print(robot.b)
        print("Path")
        for robot in robot_list:
            print(robot.p)

        t += 1

        if sum(converged_list) == robot_num:
            break


# if __name__ == "__main__":
#     target_assignment()
#     import matplotlib.pyplot as plt
#
#     np.random.seed(10)
#
#     task_num = 10
#     robot_num = 50  # 少对多分配
#
#     task = np.random.uniform(low=0, high=1, size=(task_num, 2))  # 目标状态
#     print(task)
#     # task = np.array([[0,1],[1,1],[1,2]])
#
#     robot_list = [CBBA_agent(id=i, vel=1, task_num=task_num, agent_num=robot_num, L_t=1) for i in
#                   range(robot_num)]
#     # robot_list[0].state = np.array([[0,0]])
#     # robot_list[1].state = np.array([[1,0]])
#
#     # Network Initialize
#     G = np.ones((robot_num, robot_num))  # Fully connected network 全局通信
#     # G[0,1]=0
#     # G[1,0]=0
#
#     t = 0  # Iteration number
#
#     while True:
#         converged_list = []  # Converged List
#
#         print("==Iteration {}==".format(t))
#         ## Phase 1: Auction Process
#         print("Auction Process")
#         for robot in robot_list:
#             # select task by local information
#             robot.build_bundle(task)
#
#         print("Bundle")
#         for robot in robot_list:
#             print(robot.b)
#         print("Path")
#         for robot in robot_list:
#             print(robot.p)
#
#         ## Communication stage
#         print("Communicating...")
#         # Send winning bid list to neighbors (depend on env)
#         message_pool = [robot.send_message() for robot in robot_list]
#
#         for robot_id, robot in enumerate(robot_list):
#             # Recieve winning bidlist from neighbors
#             g = G[robot_id]
#
#             connected, = np.where(g == 1)
#             connected = list(connected)
#             connected.remove(robot_id)
#
#             if len(connected) > 0:
#                 Y = {neighbor_id: message_pool[neighbor_id] for neighbor_id in connected}  # 构建通信信息
#             else:
#                 Y = None
#
#             robot.receive_message(Y)
#
#         ## Phase 2: Consensus Process
#         print("Consensus Process")
#         for robot in robot_list:
#             # Update local information and decision
#             if Y is not None:
#                 converged = robot.update_task()
#                 converged_list.append(converged)
#
#         print("Bundle")
#         for robot in robot_list:
#             print(robot.b)
#         print("Path")
#         for robot in robot_list:
#             print(robot.p)
#
#         t += 1
#
#         if sum(converged_list) == robot_num:
#             break
#
# print("Finished")
