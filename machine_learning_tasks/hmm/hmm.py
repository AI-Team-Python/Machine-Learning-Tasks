# http://blog.csdn.net/u014688145/article/details/53046765?locationNum=7&fps=1

# https://www.zhihu.com/question/27935093

# http://blog.csdn.net/u010378878/article/details/51610586

import numpy as np

# BEMS
PROB_SATRT = "prob_start.py"   # 初始概率向量写在这里
INPUT_DATA = "RenMinData.txt"  # 用于训练的语料
PROB_EMIT = "prob_emit.py"     # 混淆矩阵
PROB_TRANS = "prob_trans.py"   # 状态转移矩阵

pi_dict = {}        # 状态初始概率向量字典表示
prob_trans_dict = {}    # 状态转移矩阵字典表示
prob_emit_dict = {}     # 混淆矩阵（发射矩阵）字典表示
state_list = ['B', 'M', 'E', 'S']

state_count_dic = {}  # 记录A[key]所在行的所有列的总和，也就是在求解A_dic每行的每一个值候(A[key1][key2])，分母就是Count_dic[key]
B_dic_element_size = {}  # 这个词典中记录的是，混淆矩阵中的对应着同一个输出字符的隐藏状态的数量



def init_arrays():
    global pi_dict
    global prob_trans_dict
    global prob_emit_dict
    global state_count_dic

    for state1 in state_list:
        prob_trans_dict[v1] = {}
        prob_emit_dict[v1] = {}
        pi_dict[v1] = {}
        state_count_dic[v1] = {}

        for state2 in state_list:
            prob_trans_dict[v1][v2] = 0.0
    
    


def get_state_list_for_words(words):
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


def get_prob_trans_dict(data_set):
    # 获取状态转移概率矩阵
    prob_trans_dict = {}
    for words in data_set:
        state_list = get_state_list(words)
        pi_dict[state_list[0]] = pi_dict.get(state_list[0], 0) + 1
        
        


def get_prob_emit_dict():
    # 发射矩阵
    pass


'''
from memory_profiler import profile

@profile
def run():
    s = jieba.cut('南京大学医学院關鼓楼瞒3()7_体检病历  ')
    print(list(s))

run()'''
