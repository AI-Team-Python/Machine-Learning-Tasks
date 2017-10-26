# http://blog.csdn.net/u014688145/article/details/53046765?locationNum=7&fps=1

# https://www.zhihu.com/question/27935093

# http://blog.csdn.net/u010378878/article/details/51610586

import numpy as np

# BEMS
PROB_SATRT = "prob_start.py"   # 初始概率向量写在这里
INPUT_DATA = "RenMinData.txt"  # 用于训练的语料
PROB_EMIT = "prob_emit.py"     # 混淆矩阵
PROB_TRANS = "prob_trans.py"   # 状态转移矩阵

initial_vec = []
mu = 0
sigma = 0.4
np.random.seed(0)

d = np.random.random([5])
import jieba

def get_state_list(words):
    input_len = len(words)
    state_list = []
    if input_len == 1:
        state_list.append('S')
    else:
        middle_num = input_len - 2
        state_list.append('B')
        state_list.extend(['M'] * middle_num)
        state_list.append('E')
    return state_list


def get_prob_trans_array(data_set):
    for words in data_set:
        state_list = get_state_list(words)

from memory_profiler import profile

'''@profile
def run():
    s = jieba.cut('南京大学医学院關鼓楼瞒3()7_体检病历  ')
    print(list(s))

run()'''
