# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


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

    # F = np.zeros((vocab_size,num_classes))


class TextCNN(object):
    """文本分类，maxent or 逻辑回归模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.float32, [None, self.config.vocab_size], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.log()

    def log(self):
        W = tf.Variable(tf.truncated_normal([self.config.vocab_size, self.config.num_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]))

        with tf.name_scope("score"):
            y_conv = tf.matmul(self.input_x, W) + b
            # y_conv = tf.matmul(self.input_x, W * self.config.F)  # 特征函数与权重点乘
            self.y_pred_cls = tf.argmax(y_conv, 1)  # 预测类别

        with tf.name_scope("optimize"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=y_conv))
            # 优化器
            # self.optim = tf.train.GradientDescentOptimizer()
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
