import numpy as np
import collections

# 黑盒模型
costdb = lambda input, param: input/param + param

# 样本数据库
sampleBuffer = collections.deque()
# 训练出来一个特定的sample or 对于Buffer来说是一种延迟。对于强化学习来说，状态是Buffer，但是该状态会不断的增加。
# sample = tran(sampleBuffer)


# 下一步采样
param = sample(input, sampleBuffer)

# 采样结果
cost = costdb(input, param)
