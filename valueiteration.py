#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2020-12-13 20:49'

import numpy as np
from gridworld import *

env = GridworldEnv()

print(np.zeros(env.nS))

#theta 停止条件 不再更新  discount_factor = 1 不衰减
def value_iteration(env,theta = 0.0001,discount_factor = 1.0):

     #一步一步操作  state 当前状态，v一系列状态值
    def one_step_lookahead(state,V):
        #方向
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob,next_state,reward,done in env.P[state][a]:
                #贝尔曼方程 当前状态价值、下一步价值 当前的奖励
                A[a] += prob*(reward+discount_factor*V[next_state])
        return A
    #16个格子
    V=np.zeros(env.nS)

    #迭代更新
    while True:
        #判断是否更新
        delta = 0
        #取遍所有状态
        for s in range(env.nS):
            A = one_step_lookahead(s,V)
            best_action_value = np.max(A)
            delta = max(delta,np.abs(best_action_value-V[s]))
            V[s] = best_action_value

        if delta<theta:
            break

    #计算最终结果
    policy = np.zeros([env.nS,env.nA])
    for s in range(env.nS):
        A = one_step_lookahead(s,V)
        best_action = np.argmax(A)
        policy[s,best_action] =1.0
    return policy,V

policy,v = value_iteration(env)

# print("Policy Probability Distribution:")
# print(policy)
# print("")
#
# print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
# print(np.reshape(np.argmax(policy, axis=1), env.shape))
# print("")
