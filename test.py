# 定义二维数组
import numpy as np

array = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]

# 需要删除的列的索引
col_index = 1
if 1 in array[0]:
    print("ok")

# 删除对应的列
# array = np.delete(array,1,axis=1)
p=np.sum(array,axis=0)

print(p)
