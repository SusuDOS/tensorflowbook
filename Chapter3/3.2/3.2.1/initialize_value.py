#coding=utf-8
import tensorflow as tf

weight1 = tf.Variable(0.001)
# weight2的初始值是weight的2倍.
weight2 = tf.Variable(weight1.initialized_value() * 2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  print("weight1 is:")
  print(sess.run(weight1))
  print("weight2 is:")
  print(sess.run(weight2))
  
"""
# 上下就运行上是没有什么区别的，我就想看看到底有什么区别没有？
#coding=utf-8
import tensorflow as tf

weight1 = tf.Variable(0.001)
# weight2的初始值是weight的2倍.
weight2 = weight1 * 2
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  print("weight1 is:")
  print(sess.run(weight1))
  print("weight2 is:")
  print(sess.run(weight2))

"""
