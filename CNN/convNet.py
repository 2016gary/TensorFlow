import tensorflow as tf
from tensorflow.contrib import learn


def convNet(features, mode):
    # 输入层   改变输入数据维度为 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features, [-1, 100, 100, 3])
    tf.summary.image('input', input_layer)

    # 卷积层1
    with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv1'
        )
        conv1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv1')
        tf.summary.histogram('kernel', conv1_vars[0])
        tf.summary.histogram('bias', conv1_vars[1])
        tf.summary.histogram('act', conv1)

    # 池化层1  100->50
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')

    # 卷积层2
    with tf.name_scope('conv2'):
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv2'
        )
        conv2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv2')
        tf.summary.histogram('kernel', conv2_vars[0])
        tf.summary.histogram('bias', conv2_vars[1])
        tf.summary.histogram('act', conv2)

    # 池化层2  50->25
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')

    # 卷积层3
    with tf.name_scope('conv3'):
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv3'
        )
        conv3_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv3')
        tf.summary.histogram('kernel', conv3_vars[0])
        tf.summary.histogram('bias', conv3_vars[1])
        tf.summary.histogram('act', conv3)

    # 池化层3  25->12
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name='pool3')

    # 卷积层4
    with tf.name_scope('conv4'):
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=128,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv4'
        )
        conv4_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv4')
        tf.summary.histogram('kernel', conv4_vars[0])
        tf.summary.histogram('bias', conv4_vars[1])
        tf.summary.histogram('act', conv4)

    # 池化层4  12->6
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, name='pool4')

    # 展开并列池化层输出Tensor为一个向量
    pool4_flat = tf.reshape(pool4, [-1, 6 * 6 * 128])

    # 全链接层
    with tf.name_scope('fc1'):
        fc1 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              name='fc1')
        fc1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc1')
        tf.summary.histogram('kernel', fc1_vars[0])
        tf.summary.histogram('bias', fc1_vars[1])
        tf.summary.histogram('act', fc1)

    # dropout
    fc1_dropout = tf.layers.dropout(
        inputs=fc1, rate=0.3, training=tf.equal(mode, learn.ModeKeys.TRAIN), name='fc1_dropout')

    # 全链接层
    with tf.name_scope('fc2'):
        fc2 = tf.layers.dense(inputs=fc1_dropout, units=512, activation=tf.nn.relu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              name='fc2')
        fc2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc2')
        tf.summary.histogram('kernel', fc2_vars[0])
        tf.summary.histogram('bias', fc2_vars[1])
        tf.summary.histogram('act', fc2)

    # dropout
    fc2_dropout = tf.layers.dropout(
        inputs=fc2, rate=0.3, training=tf.equal(mode, learn.ModeKeys.TRAIN), name='fc2_dropout')

    # Logits层   对输出Tensor执行分类操作
    with tf.name_scope('out'):
        logits = tf.layers.dense(inputs=fc2_dropout, units=10, activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='out')
        out_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'out')
        tf.summary.histogram('kernel', out_vars[0])
        tf.summary.histogram('bias', out_vars[1])
        tf.summary.histogram('act', logits)

    return logits
 