# Tensorflow入门到实战

罗冬日 著 读书笔记.

## 第一、二章

主要介绍了TensorFlow的特点，以及常用的几大深度学习框架。Caffe、MXnet、Torch、Theano、CNTK等等...

主要就TensorFlow环境在不同的操作系统下的搭建问题.

- Window下搭建方案.
- 默认使用GPU版本的搭建方式.
- 需要CUDA Toolkits和cuDNN加速库的安装.
- 需要python或者是可以使用anaconda安装.
- python：`virtualenv --system-site-packages -p python3 .\tensorFlow`
- conda：`conda create --name tensorflow -p python3.6`
- python环境激活：.\TensorFlow\Script\active
- conda环境激活：conda activate tensorflow
- linux环境激活：source ~/tensorflow/bin/active
- pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.14

TensorFlow对cuda和cuDNN的依赖性很强，具有很强的版本限制，需要对应合适的版本即可。

注意：ubuntu18.04.2LST版本安装CUDA10.0会存在重启后，键盘鼠标无法使用，所以一定要记得：

```bash
sudo apt-get install openss-server
sudo service ssh start
sudo apt-get install xserver-xorg-input-all
```
## 第三章

此章节主要是基本的概念和语法的掌握的问题，一定要仔细的查看。

TensorFlow是属于结构与运算分割开来的图运算，图的运算需要创建会话来进行计算。

除相对难以记忆的，都选择性忽略.

### 变量

定义默认为 `trainable=True`，使用前必须初始化 `init = tf.Global_Variables_initializer()` .

可训练的变量在训练的过程中的值会被不断的迭代更新，但是也可以设置将其不被更新，如cc的定义.


```bash
import tensorflow as tf
import numpy as np

aa = tf.Variable(1.0, name="aa")
bb = tf.constant(2.0, name="bb")
cc = tf.Variable(1.0, trainable=False，name="cc")

init = tf.global_variable_initializer()

with tf.Session as sess：
    sess.run(init)
    sess.run(aa + bb + cc)
```
### 获取图/操作

```
graph = tf.get_default_graph()
operations = graph.get_operations()
for i in operations:
    print(i)
```

### 依赖性

```bash
x_plus_1 = tf.assign_add(x, 1, name="x_plus")
with tf.control_dependencies([x_plus_1]):
    y = tf.identity(x, name='y')
# 运行下：
    print(y.evl())
```

### 变量的共享(极其重要)

因为所有的神经网络的问题本质上就是不断迭代参数的问题，找最优参数的问题，在面向流程的算法中只需要按照流程执行即可。但是要实现图的迭代优化的问题是需要参数共享的，共享是深度学习的基础.

```python
def inference(x):
    weight = tf.get_variable("weight",[1])
    bias = tf.get_variable("bias",[1])
    y = x * weight + bias
    return y


with tf.variable_scope("inference"):
    """
    # 判断重用属性是否为False？
    print(tf.get_variable_scope().reuse==False)
    
    默认作用域不可重用,修改为可重用.
    with tf.variable_scope("inference",reuse=True):
    
    下面三行相当于第一行不可重用，第二行修改之后，该作用域下变量名可以重用.
    """    
    train_y = inference(train_x)    
    tf.get_variable_scope().reuse_variables()
    test_y = inference(test_x)
```

#### 模型的保存和载入.

使用tf.train.Saver()进行操作.
```bash
saver = tf.train.Saver()
saver_model=save.save(sess, "./model/model.ckpt",global_step=step)
saver.restore(sess, "/tmp/model.ckpt")
```
具体的如何使用还是有疑惑，可以查看如下示范.
```bash
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
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
```

#### 设备分配
```bash
with tf.device("/cpu:0"):
    v1 = tf.constant(1,name="v1")
with tf.device("/gpu:0"):
    v2 = tf.constant(2,name="v2")
```
查看变量分配设备位置.
```bash
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(v2)) 
```
tensorflow1.14版本问题，最好使用设备选择分配。

也就是说当设备不存在的时候自动分配到其他存在的设备：指的是其他的GPU，CPU还是有设备优先级的？
```bash
config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
with tf.Sess(config=config) as sess:
    sess.run(init)
```

#### 显存分配的两种方式.

```python
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5

config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    ...
```

### 数据读取.

或者叫做数据的读取方式吧.

- placeholder的方式.

```python
import tensorflow as tf
v1 = tf.placeholder(tf.float32)
v2 = tf.placeholfer(tf.float32)
v_mul = tf.multiply(v1,v2)

with tf.Session() as sess:
    while(True):
        value1 = input("value1:")
        value2 = input("value2:")
            if value1=='exit' or value2=='exit':
                break
            result = sess.run(v_mul,feed_dict{value1:value1,value2:value2})
            print("Result:%f" % result)
```
- 读取文件的方式.
	- csv文件
	- bin文件
	- TFRecord文件(重点掌握-3.5)

有点复杂，但是有需要完全掌握的必要，有可能对提高训练速度有极大的提升.

- 预先读入内存(小量数据可以一次性读入，但是未提及一次性如何将数据读入的问题.)

### 训练过程的可视化

书中重点介绍了两种可视化的使用方式.

- graph，必备.

```python
tf.summary.FileWriter('./calc_graph').add(tf.get_default_graph())
```
- 参数可视化

```python
tf.summary.scalar("weight", weight)
tf.summary.scalar("biase", biase)
tf.summary.scalar("loss", loss[0])

merged_summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./log_graph' )
# add应该也是可以的.
summary_writer.add_graph(tf.get_default_graph())


# with tf.Session() as sess:
_, summary = sess.run([train_op, merged_summary], feed_dict={x:train_x, y:train_y})
summary_writer.add_summary(summary, step)
```
- 参数分布可视化

```python
tf.summary.histogram("weight", weight)
tf.summary.histogram("biase", biase)

merged_summary = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter('./log_graph_hist' )
summary_writer.add_graph(tf.get_default_graph())

# with tf.Session as tf:
_, summary,weight_value  = sess.run([train_op, merged_summary, weight], feed_dict={x:train_x, y:train_y})
summary_writer.add_summary(summary, step)
```

### 后面章节的技术性不强，暂时忽略.
