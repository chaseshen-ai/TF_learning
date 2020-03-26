# _*_coding:utf-8_*_
import tensorflow as tf

# using saver.restore get the model weight

# 使用和保存模型代码中一样的方式来声明变量
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    # 加载已经保存的模型，并通过已经保存的模型中的变量的值来计算加法
    model_path = 'model/model.ckpt'
    saver.restore(sess, model_path)
    print(sess.run(result))

# 结果如下：[3.]
