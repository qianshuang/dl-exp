# -*- coding: utf-8 -*-

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64      # 词向量维度
    seq_length = 1000        # 序列长度
    num_classes = 6        # 类别数
    num_filters = 128        # 卷积核数目
    kernel_size = 3         # 卷积核尺寸

    hidden_dim = 128        # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 50         # 总迭代轮次

    print_per_batch = 50    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            # 模型自己学出词向量矩阵
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer 1: 输入为1000*64，64为input_channels
            filter_w_1 = tf.Variable(tf.truncated_normal([self.config.kernel_size, self.config.embedding_dim, self.config.num_filters], stddev=0.1))
            conv_1 = tf.nn.conv1d(embedding_inputs, filter_w_1, 1, padding='SAME')  # [batch, 1000, 128]
            active_1 = tf.nn.relu(conv_1)  # 激活函数
            expand_1 = tf.expand_dims(active_1, 2)  # [batch, 1000, 1, 128]
            pool_1 = tf.nn.max_pool(expand_1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding="VALID") # (batch, 500, 1, 128)

            # CNN layer 2
            input_2 = tf.reshape(pool_1, (-1, 500, 128))
            filter_w_2 = tf.Variable(tf.truncated_normal([self.config.kernel_size, 128, 128], stddev=0.1))
            conv_2 = tf.nn.conv1d(input_2, filter_w_2, 1, padding='SAME')  # [batch, 500, 128]

            # 高级封装
            # tf.contrib.layers.conv2d()
            # tf.layers.conv1d()
            # tf.contrib.layers.max_pool2d()

            # global max pooling layer
            gmp = tf.reduce_max(conv_2, reduction_indices=[1], name='gmp')  # [batch, 128]
            # 或者下面的写法
            # expand_2 = tf.expand_dims(gmp, 2)  # [batch, 500, 1, 128]
            # pool_1 = tf.nn.max_pool(expand_2, ksize=[1, 500, 1, 1], strides=[1, 500, 1, 1], padding="VALID") # (batch, 1, 1, 128)
            # gmp = tf.squeeze(gmp)  # 剔除维度为1的维
            print(gmp)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # 激活函数
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(self.logits, 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
