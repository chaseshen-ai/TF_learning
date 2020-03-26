import tensorflow as tf

# 直接加载持久化的图
model_path = 'model/model.ckpt'
model_path1 = 'model/model.ckpt.meta'
saver = tf.train.import_meta_graph(model_path1)

with tf.Session() as sess:
    saver.restore(sess, model_path)
    # 通过张量的的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name('v1:0')))

    # ckeck the name in graph
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    for tensor_name in tensor_name_list:
        print(tensor_name,'\n')

# 结果如下：[3.]


