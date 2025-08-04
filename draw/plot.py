import random

from matplotlib import pyplot as plt
import numpy as np
import csv
from scipy.interpolate import make_interp_spline
import pandas as pd

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 绘制损失值
x, y = [], []
# actor损失
# with open("actor_loss.csv", 'r') as csvfile:  # run-25-tag-reward_1_smooth.csv
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x.append(int(row[1]))  # 从csv读取的数据是str类型
#         y.append(float(row[2]))  # *100
# csvfile.close()
#
# plt.plot(x, y, color='orange')
# plt.xlabel('训练轮次')  # 训练轮次
# plt.ylabel('损失值')  # Success rate(%)奖励
# plt.title('actor网络损失值 ')  # Task success rate of UAV swarm
# plt.legend()
# # 损失值
# plt.grid()
# plt.savefig('./actor_loss.png', dpi=300)
# plt.show()

# critic损失
# with open("critic_loss.csv", 'r') as csvfile:  # run-25-tag-reward_1_smooth.csv
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x.append(int(row[1]))  # 从csv读取的数据是str类型
#         y.append(float(row[2]))  # *100
# csvfile.close()
#
# plt.plot(x, y, color='blue')
# plt.xlabel('训练轮次')  # 训练轮次
# plt.ylabel('损失值')  # Success rate(%)奖励
# plt.title('critic网络损失值 ')  # Task success rate of UAV swarm
# plt.legend()
# # 损失值
# plt.grid()
# plt.savefig('./critic_loss.png', dpi=300)
# plt.show()

# tar_actor损失
# with open("tar_actor_loss.csv", 'r') as csvfile:  # run-25-tag-reward_1_smooth.csv
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x.append(int(row[1]))  # 从csv读取的数据是str类型
#         y.append(float(row[2]))  # *100
# csvfile.close()
#
# plt.plot(x, y, color='orange')
# plt.xlabel('训练轮次')  # 训练轮次
# plt.ylabel('损失值')  # Success rate(%)奖励
# plt.title('actor网络损失值 ')  # Task success rate of UAV swarm
# plt.legend()
# # 损失值
# plt.grid()
# plt.savefig('./tar_actor_loss.png', dpi=300)
# plt.show()

# tar_critic损失
# with open("tar_critic_loss.csv", 'r') as csvfile:  # run-25-tag-reward_1_smooth.csv
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x.append(int(row[1]))  # 从csv读取的数据是str类型
#         y.append(float(row[2]))  # *100
# csvfile.close()
#
# plt.plot(x, y, color='blue')
# plt.xlabel('训练轮次')  # 训练轮次
# plt.ylabel('损失值')  # Success rate(%)奖励
# plt.title('critic网络损失值 ')  # Task success rate of UAV swarm
# plt.legend()
# # 损失值
# plt.grid()
# plt.savefig('./tar_critic_loss.png', dpi=300)
# plt.show()

# # 奖励值
# with open("reward.csv", 'r') as csvfile:  # run-25-tag-reward_1_smooth.csv
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x.append(int(row[1]))  # 从csv读取的数据是str类型
#         y.append(float(row[2]))  # *100
# csvfile.close()
#
# plt.plot(x, y, color='blue')
# plt.xlabel('训练轮次')  # 训练轮次
# plt.ylabel('奖励值')  # Success rate(%)奖励
# plt.title('巡飞弹集群回合平均奖励值')  # Task success rate of UAV swarm
# plt.legend()
# # 损失值
# plt.grid()
# plt.savefig('./reward.png', dpi=300)
# plt.show()
#
# tar奖励值
# with open("tar_reward.csv", 'r') as csvfile:  # run-25-tag-reward_1_smooth.csv
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x.append(int(row[1]))  # 从csv读取的数据是str类型
#         y.append(float(row[2]))  # *100
# csvfile.close()
#
# plt.plot(x, y, color='blue')
# plt.xlabel('训练轮次')  # 训练轮次
# plt.ylabel('奖励值')  # Success rate(%)奖励
# plt.title('无人机集群回合平均奖励值')  # Task success rate of UAV swarm
# plt.legend()
# # 损失值
# plt.grid()
# plt.savefig('./tar_reward.png', dpi=300)
# plt.show()

# 追击成功率
# x3 = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
# y3 = [0, 15.0, 24.520, 36.24, 45.87, 62.45, 82.21, 86.89, 93.46, 94.66, 92.58, 93.98, 94.01, 93.24, 92.58, 93.51,
#       95.85, 96.45, 96.21, 94.28, 95.36]
# plt.plot(x3, y3, "o-", color='blue')
# plt.xlabel('训练轮次')  # 训练轮次
# plt.ylabel('追击成功率(%)')  # Success rate(%)奖励
# plt.title('巡飞弹大规模集群追击任务成功率')  # Task success rate of UAV swarm
# plt.legend()
# # 奖励值
# plt.grid()
# plt.savefig('./success_rate.png', dpi=300)
# plt.show()

# 不同我方巡飞弹的突防成功率
# x1 = [40, 50, 60, 70, 80, 90]
# # y1 = [13.45, 22.83, 44.17, 69.14, 80, 86.54, 90.69, 92.14, 93.98]  # 追击成功率
# y2 = [0, 78.64, 89.66, 91.75, 92.21, 95.79]
#
# plt.grid()
# # plt.xticks(range(0, 90, 10))
# plt.plot(x1, y2, marker='o')
# # plt.plot(x1, y2, marker='o', label="GS-MFDDPG")
#
# plt.xlabel('我方巡飞弹数量')
# # plt.ylabel('我方追击率(%)')
# plt.ylabel('任务效率(%)')
# plt.title('不同我方巡飞弹数量下的任务效率')
# plt.legend()
# plt.savefig('./不同我方巡飞弹数量下的任务效率.png', dpi=300)
# plt.show()

# x0, y0, x1, y1, x2, y2 = [], [], [], [], [], []
#
# with open("./reward.csv", 'r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x0.append(int(row[1]))  # 从csv读取的数据是str类型
#         y0.append(float(row[2]))
#
# with open("./reward_1.csv", 'r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x1.append(int(row[1]))  # 从csv读取的数据是str类型
#         y1.append(float(row[2]))
#
# with open("./reward_2.csv", 'r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x2.append(int(row[1]))  # 从csv读取的数据是str类型
#         y2.append(float(row[2]))
#
#
# # 画折线图
# plt.grid()
# # plt.ylim(0.9, 1)
# plt.plot(x0, y0, label='GL-MFDDPG')
# plt.plot(x1, y1, label='GS-MFDDPG')
# plt.plot(x2, y2, label='MFDDPG')
#
# plt.xlabel('训练回合')
# plt.ylabel('奖励')
# plt.title('我方巡飞弹集群回合平均奖励值')
# plt.legend()
# plt.savefig('./奖励值对比.png', dpi=300)
# plt.show()

# 柱状图
x = np.array(['无禁飞区', '有禁飞区'])
y = np.array([56, 48])  # 追击率
y_ = np.array([92, 88])
y__ = np.array([96, 92])
# y = np.array([54.23,48.21])  # 任务成功率
# y_ = np.array([92.56, 90.18])
# y__ = np.array([95.34, 93.12])
# print(y[0], y_[0], y__[0])
plt.grid(axis='y')
# plt.ylim(0, 1)
# 设置柱状图的宽度
width = 0.25
# # 创建柱状图
plt.bar(np.arange(len(x)), y, width=width, label='MFDDPG')
plt.bar(np.arange(len(x)) + width, y_, width=width, label='GS-MFDDPG')
plt.bar(np.arange(len(x)) + 2 * width, y__, width=width, label='GL-MFDDPG')
# 添加标题和标签
plt.title('追击成功率对比图')
# plt.xlabel('X')
# plt.ylabel('Enemy penetration rate(%)')
plt.ylabel('追击成功率(%)')
# 设置X轴刻度标签
plt.xticks(np.arange(len(x)) + width, x)
# 添加图例
plt.legend()
# 显示图表
plt.savefig('./追击成功率对比图.png', dpi=300)
# plt.savefig('./我方追击率对比图.png', dpi=300)
plt.show()
