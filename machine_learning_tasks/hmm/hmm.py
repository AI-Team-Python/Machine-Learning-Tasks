# http://blog.csdn.net/u014688145/article/details/53046765?locationNum=7&fps=1

# https://www.zhihu.com/question/27935093

# http://blog.csdn.net/u010378878/article/details/51610586

# blog.csdn.net/u014688145/article/details/53012400

import numpy as np

# BEMS
PROB_SATRT = "prob_start.py"   # 初始概率向量写在这里
INPUT_DATA = "RenMinData.txt"  # 用于训练的语料
PROB_EMIT = "prob_emit.py"     # 混淆矩阵
PROB_TRANS = "prob_trans.py"   # 状态转移矩阵

pi_dict = {}        # S：状态初始概率向量字典表示
pi_dict_size = 0.0
prob_trans_dict = {}    # A：状态转移矩阵字典表示
prob_emit_dict = {}     # B：混淆矩阵（发射矩阵）字典表示
state_list = ['B', 'M', 'E', 'S']

state_count_dict = {}  # 记录每一个状态state出现的总次数
element_count_dict = {}  # 这个词典中记录的是，混淆矩阵中的对应着同一个输出字符的隐藏状态的数量(也就是一个字出现的总次数)



def init_arrays():
    global pi_dict
    global prob_trans_dict
    global prob_emit_dict
    global state_count_dict

    for state1 in state_list:
        prob_trans_dict[state1] = {}
        prob_emit_dict[state1] = {}
        pi_dict[state1] = 0.0 # 初始化状态初始概率向量中所有的元素值为0.0
        state_count_dict[state1] = {}

        for state2 in state_list:
            prob_trans_dict[state1][state2] = 0.0
        
        state_count_dict[state1] = 0
    
    


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


def get_probs(data_set):
    # 统计状态转移矩阵、发射矩阵
    global pi_dict
    global prob_trans_dict
    global prob_emit_dict
    global state_count_dict
    global element_count_dict
    global pi_dict_size
    for words in data_set:
        state_list = get_state_list_for_words(words)
        pi_dict[state_list[0]] = pi_dict.get(state_list[0], 0) + 1
        word_list = list(words)
        pi_dict_size += len(state_list)
        for i in range(len(state_list)):
            state = state_list[i]
            w = word_list[i]
            prob_emit_dict[state][w] =  prob_emit_dict[state].get(w, 0) + 1
            num = element_count_dict.get(w, 0)
            if num == 0:
                element_count_dict[w] = 1
            else:
                element_count_dict[w] = num + 1
            
            if i < len(state_list) - 1:
                prob_trans_dict[state][state_list[i+1]] += 1
            state_count_dict[state] += 1
    # 初始概率字典 pi
    for state in pi_dict:
        pi_dict[state] = pi_dict[state]/pi_dict_size

    # 状态转移矩阵
    for state1 in prob_trans_dict:
        for state2 in prob_trans_dict[state1]:
            num = prob_trans_dict[state1][state2]
            prob_trans_dict[state1][state2] = num / state_count_dict[state1]

    # 发射矩阵（混淆矩阵）
    for state in prob_emit_dict:
        for w in prob_emit_dict[state]:
            num = prob_emit_dict[state][w]
            # p(state1->w1) = num_of(state1+w1)/num_of(w1)
            # 一个状态 state1 生成一个词w1的概率 等于 w1是state1的总数量 除以 w1出现的总次数
            prob_emit_dict[state][w] = num / element_count_dict[w]


import os, jieba

dir = os.path.dirname(__file__)

def get_data_set():
    with open(dir+'/text.txt', 'r', encoding='utf-8') as f:
        t = f.readlines()
        t1 = t[0]
        words = list(jieba.cut(t1))
        return words
    
import json

def run():
    init_arrays()
    data_set = get_data_set()
    get_probs(data_set)
    print("----------------以下Pi向量-------------------")
    print(pi_dict)
    with open(dir+'/prob_emit_dict.json', 'w', encoding='utf-8') as f:
        json.dump(prob_emit_dict, f, ensure_ascii=False, indent=4)
    print('-------------以下是状态转移矩阵----------------')
    for k1 in prob_emit_dict:
        for k2 in prob_emit_dict[k1]:
            print(k1, '-->', k2, prob_emit_dict[k1][k2])

    print('---------------以下是混淆矩阵-----------------')
    for k1 in prob_emit_dict:
        for k2 in prob_emit_dict[k1]:
            print(k1, '-->', k2, prob_emit_dict[k1][k2])


'''
from memory_profiler import profile

@profile
def run():
    s = jieba.cut('南京大学医学院關鼓楼瞒3()7_体检病历  ')
    print(list(s))

run()'''

run()