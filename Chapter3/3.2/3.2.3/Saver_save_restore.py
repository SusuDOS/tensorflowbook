#coding=utf-8
import numpy as np
import tensorflow as tf
import os

tf.reset_default_graph()

# 获取训练数据和测试数据
def get_data(number):
    list_x = []
    list_label = []
    for i in range(number):
        x = np.random.randn(1)
        # 这里构建数据的分布满足 y = 2 * x + 10
        label = 2 * x + np.random.randn(1) * 0.01  + 10
        list_x.append(x)
        list_label.append(label)
    return list_x, list_label

def inference(x):
    weight = tf.get_variable("weight",[1])
    bias = tf.get_variable("bias",[1])
    y = x * weight + bias
    return y

train_x = tf.placeholder(tf.float32)
train_label = tf.placeholder(tf.float32)
test_x = tf.placeholder(tf.float32)
test_label = tf.placeholder(tf.float32)

with tf.variable_scope("inference"):
    """
    # 判断重用属性是否为False？
    print(tf.get_variable_scope().reuse==False)
    
    默认作用域不可重用,修改为可重用.
    with tf.variable_scope("inference",reuse=True):
    
    下面三行相当于第一行不可重用，后面修改之后，该作用域空间可以重用.
    """    
    train_y = inference(train_x)    
    tf.get_variable_scope().reuse_variables()
    test_y = inference(test_x)  

train_loss = tf.square(train_y - train_label)
test_loss = tf.square(test_y - test_label)
opt = tf.train.GradientDescentOptimizer(0.002)
train_op = opt.minimize(train_loss)

train_data_x, train_data_label = get_data(5000) #读取训练数据的函数
test_data_x, test_data_label = get_data(1)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    if os.path.exists("./model/checkpoint"):
        saver.restore(sess,"./model/my.ckpt")
    for i in range(5000):
        sess.run(train_op, feed_dict={train_x: train_data_x[i],
                                      train_label:train_data_label[i]})
        if i % 1000 == 0:            
            test_loss_value = sess.run(test_loss, 
                            feed_dict={test_x:test_data_x[0],
                                       test_label: test_data_label[0]})            
            print("step %d eval loss is %.3f" %(i,test_loss_value))
    save_path = saver.save(sess,"./model/my.ckpt")
