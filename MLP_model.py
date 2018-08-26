# -*- coding: utf-8 -*-

import tensorflow as tf


class TCNNConfig(object):
    """配置参数"""
    vocab_size = 5000          # 词表大小
    num_classes = 10        # 类别数
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 50         # 总迭代轮次

    print_per_batch = 10    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

    dropout_keep_prob = 0.5  # dropout


class TextCNN(object):
    """文本分类，MLP模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.float32, [None, self.config.vocab_size], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.log()

    def log(self):
        with tf.name_scope("score"):
            # hidden layer
            # 使用truncated_normal（高斯）初始化权重，可以避免大的权重值减慢训练，切记不可用全0初始化，回忆BP原理
            W1 = tf.Variable(tf.truncated_normal([self.config.vocab_size, 1024], stddev=0.1))  # 隐藏层1024个神经元
            b1 = tf.Variable(tf.constant(0.1, shape=[1024]))
            y_conv_1 = tf.matmul(self.input_x, W1) + b1
            layer_1 = tf.nn.relu(y_conv_1)  # 激活函数

            # 以上代码还可以通过下面的方式简化实现：
            # y_conv_1 = tf.layers.dense(self.input_x, 1024)
            # y_conv_1 = tf.contrib.layers.fully_connected(self.input_x, 1024, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))

            # output layer
            W2 = tf.Variable(tf.truncated_normal([1024, self.config.num_classes], stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]))
            y_conv = tf.matmul(layer_1, W2) + b2

            self.y_pred_cls = tf.argmax(y_conv, 1)  # 预测类别

        with tf.name_scope("optimize"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=y_conv))
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
