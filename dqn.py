#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__author__ = 'IT小叮当'
__time__ = '2020-12-14 09:29'

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import cv2
import sys
sys.path.append('game/')
from game import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
#观察1w帧图像
OBSERVER = 10000
#探索
EXPLORE = 3000000
#探索概率
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
#存储当前观察值
REPLAY_MEMORY = 50000
BATCH = 32


def weights_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial  = tf.constant(0.01,shape=shape)
    return  tf.Variable(initial)

def conv2d(x,w,stride):
    return tf.nn.conv2d(x,w,strides = [1,stride,stride,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize =[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def createNetwork():
    #3个卷积层
    #得到32个特征图
    w_conv1 = weights_variable([8,8,4,32])
    b_conv1 = bias_variable([32])
    #输入32个特征图，得到64个特征图
    w_conv2 = weights_variable([4,4,32,64])
    b_conv2 = bias_variable([64])

    w_conv3 = weights_variable([3,3,64,64])
    b_conv3 = bias_variable([64])
    #h*w*c = 1600个特征值 希望得到512维向量
    w_fc1 = weights_variable([1600,512])
    b_fc1 = weights_variable([512])

    #全连接层
    w_fc2 = weights_variable([512, ACTIONS])
    b_fc2 = weights_variable([ACTIONS])

    #按batch传 shape为None
    s = tf.placeholder('float',[None,80,80,4])

    h_conv1 = tf.nn.relu(conv2d(s,w_conv1,4)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2,2)+b_conv2)
    #h_pool2 = max_pool_2x2(h_conv1)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3, 1) + b_conv3)

    #拉平，拉成一个向量
    h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
    #全连接操作
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,w_fc1)+b_fc1)

    readout = tf.matmul(h_fc1,w_fc2)+b_fc2
    return  s,readout,h_fc1



def trainNetwork(s,readout,h_fc1,sess):

    a = tf.placeholder('float',[None,ACTIONS])
    y = tf.placeholder('float',[None])

    #当前结果，预测值
    readout_action = tf.reduce_sum(tf.multiply(readout,a),reduction_indices = 1)
    #当前结果，接近下一个值
    cost = tf.reduce_mean(tf.square(y-readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game_state = game.GameState()

    D = deque()
    #四帧图像拿出来
    do_noting = np.zeros(ACTIONS)
    do_noting[0] = 1
    #第0帧
    x_t,r_0,terminal = game_state.frame_step(do_noting)
    x_t = cv2.cvtColor(cv2.resize(x_t,(80,80)),cv2.COLOR_BGR2GRAY)
    ret,x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)

    s_t = np.stack((x_t,x_t,x_t,x_t),axis = 2)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state('saved_networks')

    #如果之前训练了，接着训练
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess,checkpoint.model_checkpoint_path)
        print('Successfully loaded:',checkpoint.model_checkpoint_path)
    else:
        print('load failed')

    #探索
    epsilon = INITIAL_EPSILON
    t = 0
    while 'flappy bird' != 'angry bird':
        #选择贪婪epsilon
        readout_t =readout.eval(feed_dict = {s:[s_t]})[0]
        a_t = np.zeros(ACTIONS)
        action_index = 0
        if t % 1 == 0:
            if random.random() <= epsilon:
                print("t_value="+str(t)+"----"+"------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 #不更新

        # 减小epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVER:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #执行选择的动作并观察下一状态和奖励
        x_t1_colored,r_t,terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1,(80,80,1))
        #取3个
        s_t1 = np.append(x_t1,s_t[:,:,:3],axis=2)

        #存储转移在D中
        D.append((s_t,a_t,r_t,s_t1,terminal))
        if len(D)>REPLAY_MEMORY:
            D.popleft()

        #只要观察结束就训练
        if t > OBSERVER:
            minibatch = random.sample(D,BATCH)

            s_j_batch = [d[0] for d in minibatch]

            a_batch = [d[1] for d in minibatch]

            r_batch = [d[2] for d in minibatch]

            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []

            #下一状态
            readout_j1_batch = readout.eval(feed_dict ={s:s_j1_batch})
            for i in range(0,len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i]+GAMMA*np.max(readout_j1_batch[i]))

            train_step.run(feed_dict={
                y:y_batch,
                a:a_batch,
                s:s_j_batch}
            )

            #状态更新
        s_t = s_t1
        t += 1


        if t % 10000 ==0:
            saver.save(sess,'saved_networks/'+ GAME + '-dqn',global_step=t)

        state=''
        if t <= OBSERVER:
            state = 'OBSERVER'
        elif t > OBSERVER and t <= OBSERVER+EXPLORE:
            state ='EXPLORE'
        else:
            state = 'train'

        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))

def playGame():
    sess = tf.InteractiveSession()
    s,readout,h_fc1 = createNetwork()
    trainNetwork(s,readout,h_fc1,sess)



def main():
    playGame()

if __name__ == '__main__':
     main()


