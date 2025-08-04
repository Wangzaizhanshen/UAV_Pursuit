import numpy as np
import copy
from scipy.spatial import distance_matrix


class CBAA_agent():
    def __init__(self, id, task, c):
        """
    c: individual score list 得分列表
    x: local assignment list 任务列表
    y: local winning bid list 获胜者列表
    state: state of the robot
    """

        # self.J = None
        self.task_num = len(task)

        # Local Task Assignment List  初始化任务列表
        self.x = [0 for i in range(self.task_num)]
        # self.J=None
        # Local Winning Bid List 初始化目前为止每个任务最高投标
        self.y = np.array([-np.inf for _ in range(self.task_num)])

        # This part can be modified depend on the problem
        self.state = np.random.uniform(low=0, high=1, size=(1, 2))  # Agent State (Position) 智能体状态空间设计
        self.c = c  # 输入威胁矩阵作为评分 -distance_matrix(self.state, task).squeeze()  # Score (Euclidean Distance)  计算向量距离，作为得分

        # Agent ID 智能体ID
        self.id = id

    def select_task(self):  # 选择任务
        if sum(self.x) == 0:
            # Valid Task List
            h = (self.c > self.y)
            if h.any():  # 有竞标成功的任务
                # Just for euclidean distance score (negative)
                c = copy.deepcopy(self.c)
                c[h == False] = -np.inf

                self.J = np.argmax(c)  # 竞争成功的任务索引
                # print(self.J)
                self.x[self.J] = 1
                self.y[self.J] = self.c[self.J]


    def update_task(self, Y=None):
        """
    [input]
    Y: winning bid lists from neighbors (dict:{neighbor_id:bid_list}) 邻居的最高投标字典
    [output]
    converged: True or False 输出：是/否
    """

        old_x = copy.deepcopy(self.x)

        id_list = list(Y.keys())
        id_list.insert(0, self.id)

        y_list = np.array(list(Y.values()))

        # Update local winning bid list
        # When recive only one message
        if len(y_list.shape) == 1:  # 只有一个信息
            # make shape as (1,task_num)
            y_list = y_list[None, :]  # 升维度

        # Append the agent's local winning bid list and neighbors'
        y_list = np.vstack((self.y[None, :], y_list))  # 拼接两个投标列表

        self.y = y_list.max(0)  # 对位比较，取最大值，获得新的投标列表

        ## Outbid check
        # Winner w.r.t the updated local winning bid list
        # print(self.J)
        max_id = np.argmax(y_list[:, self.J])  # 获取智能体i之前竞争成功的任务的评分，与邻居的进行比较，获取最大方的索引
        z = id_list[max_id]  # 智能体ID
        # If the agent is not the winner
        if z != self.id:  # 智能体未获胜，没有竞争过邻居，则释放任务
            # Release the assignment
            self.x[self.J] = 0

        converged = False  # 存在冲突
        if old_x == self.x:  # 不存在冲突
            converged = True

        return converged

    def send_message(self):  # 给邻居发送信息
        """
    Return local winning bid list
    [output]
    y: winning bid list (list:task_num)
    """
        return self.y.tolist()

# if __name__ == "__main__":
#
#     task_num = 5
#     robot_num = 5
#
#     task = np.random.uniform(low=0, high=1, size=(task_num, 2))  # 初始化任务
#
#     robot_list = [CBAA_agent(id=i, task=task) for i in range(robot_num)]  # 初始化智能体，都是所有任务
#
#     # Network Initialize 初始化通信网络，全局通信
#     G = np.ones((robot_num, robot_num))  # Fully connected network\
#     # Configure network topology manually
#     # G[0,1]=0
#     # G[1,0]=0
#
#     t = 0  # Iteration number
#     while True:
#         converged_list = []
#
#         print("==Iteration {}==".format(t))
#         # Phase 1: Auction Process
#         print("Auction Process")
#         for robot in robot_list:
#             # select task by local information
#             robot.select_task()  # 根据当前信息选择任务
#             print(robot.state)
#             print(robot.c)
#             print(robot.x)
#
#         # Phase 2: Consensus Process
#         print("Consensus Process")
#         # Send winning bid list to neighbors (depend on env)
#         message_pool = [robot.send_message() for robot in robot_list]  # 信息都放信息池里
#
#         for robot_id, robot in enumerate(robot_list):
#             # Recieve winning bidlist from neighbors
#             g = G[robot_id]
#
#             connected, = np.where(g == 1)
#             connected = list(connected)
#             connected.remove(robot_id)  # 移除自己
#
#             if len(connected) > 0:
#                 Y = {neighbor_id: message_pool[neighbor_id] for neighbor_id in connected}  # 组成邻居投标信息
#             else:
#                 Y = None
#
#             # Update local information and decision
#             if Y is not None:
#                 converged = robot.update_task(Y)  # 更新任务列表吧
#                 converged_list.append(converged)  # 存储更新状态
#
#             print(robot.x)
#             print(converged_list)
#
#         t += 1
#
#         if sum(converged_list) == robot_num:  # 都不存在冲突就结束，不然就继续迭代
#             break
#
#     print("CONVERGED")
