# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    # label_embedding_dim = 128  # label向量维度
    seq_length = 100  # 序列长度
    num_classes = 0  # 类别数
    num_tasks = 2
    num_filters = 128  # 卷积核数目

    hidden_dim = 256  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 200  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 200  # 每多少轮存入tensorboard

    # num_features = 4
    # num_label = 20


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.input_Q = tf.placeholder(tf.int32, [None, self.config.seq_length])
        self.input_label = tf.placeholder(tf.float32, [None, self.config.num_classes])
        self.input_task = tf.placeholder(tf.float32, [None, self.config.num_tasks])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        self.cnn()

    def conv_1d(self, x, gram, input_channel, output_channel):
        # 第一层卷积
        filter_w_1 = tf.Variable(tf.truncated_normal([gram, input_channel, output_channel], stddev=0.1))
        filter_b_1 = tf.Variable(tf.constant(0.1, shape=[output_channel]))
        conv_1 = tf.nn.conv1d(x, filter_w_1, padding='SAME', stride=1) + filter_b_1
        h_conv_1 = tf.nn.relu(conv_1)
        h_pool1_flat2 = tf.reduce_max(h_conv_1, reduction_indices=[1])

        # so = tf.nn.softmax(h_pool1_flat2)
        # adv_loss = tf.reduce_mean(-(1/output_channel) * tf.log(so))
        # self.adv_losses = self.adv_losses + adv_loss
        return h_pool1_flat2

    def network_bcnn(self, embedding_inputs):
        flaten_1 = self.conv_1d(embedding_inputs, 1, 64, 128)
        flaten_2 = self.conv_1d(embedding_inputs, 2, 64, 128)
        flaten_3 = self.conv_1d(embedding_inputs, 3, 64, 128)
        flaten_4 = self.conv_1d(embedding_inputs, 4, 64, 128)
        flaten_5 = self.conv_1d(embedding_inputs, 5, 64, 128)
        h_pool1 = tf.concat([flaten_1, flaten_2, flaten_3, flaten_4, flaten_5], -1)  # 列上做concat
        return h_pool1

    def fc(self, embedding_inputs, input_dim, output_dim):
        W_ = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
        b_ = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        h_fc = tf.nn.relu(tf.matmul(embedding_inputs, W_) + b_)
        return h_fc

    def adversarial_loss(self, feature, task_label):
        '''make the task classifier cannot reliably predict the task based on the shared feature'''
        # input = tf.stop_gradient(input)
        feature = flip_gradient(feature)
        feature = tf.nn.dropout(feature, self.keep_prob)

        logits = self.fc(feature, 100, 2)
        loss_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=task_label, logits=logits))
        return loss_adv

    def diff_loss(self, shared_feat, task_feat):
        '''Orthogonality Constraints from https://github.com/tensorflow/models, in directory research/domain_adaptation'''
        task_feat -= tf.reduce_mean(task_feat, 0)
        shared_feat -= tf.reduce_mean(shared_feat, 0)

        task_feat = tf.nn.l2_normalize(task_feat, 1)
        shared_feat = tf.nn.l2_normalize(shared_feat, 1)

        correlation_matrix = tf.matmul(task_feat, shared_feat, transpose_a=True)

        cost = tf.reduce_mean(tf.square(correlation_matrix)) * 0.01
        cost = tf.where(cost > 0, cost, 0, name='value')

        assert_op = tf.Assert(tf.is_finite(cost), [cost])
        with tf.control_dependencies([assert_op]):
            loss_diff = tf.identity(cost)

        return loss_diff

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        embedding_Q = tf.get_variable('embedding_Q', [self.config.vocab_size, self.config.embedding_dim])
        embedding_inputs_Q = tf.nn.embedding_lookup(embedding_Q, self.input_Q)
        # input dropout
        embedding_inputs_Q = tf.nn.dropout(embedding_inputs_Q, self.keep_prob)

        # BCNN
        rep_Q_src = self.network_bcnn(embedding_inputs_Q)  # [-1, 128 * 5]
        fc_out_src = self.fc(rep_Q_src, 128 * 5, 100)  # [-1, 100]

        rep_Q_share = self.network_bcnn(embedding_inputs_Q)  # [-1, 128 * 5]
        fc_out_share = self.fc(rep_Q_share, 128 * 5, 100)

        feature = tf.concat([fc_out_src, fc_out_share], axis=1)
        feature = tf.nn.dropout(feature, self.keep_prob)

        # 分类器
        W_fc2 = tf.Variable(tf.truncated_normal([200, self.config.num_classes], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]))
        y_conv = tf.matmul(feature, W_fc2) + b_fc2
        self.y_pred_cls = tf.argmax(y_conv, 1)  # 预测类别

        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=self.input_label)
        loss_src = tf.reduce_mean(cross_entropy)

        # adversary loss
        loss_adv = self.adversarial_loss(fc_out_share, self.input_task)
        loss_diff = self.diff_loss(fc_out_share, fc_out_src)

        self.loss = loss_src + 0.05 * loss_adv + loss_diff

        # 优化器
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # 准确率
        self.correct_pred = tf.equal(tf.argmax(self.input_label, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


class FlipGradientBuilder(object):
    '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()
