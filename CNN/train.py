import os
import tensorflow as tf
from tensorflow.contrib import learn
# 导入模型和数据
from convNet import convNet
from read_TFRecord import read_and_decode

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 屏蔽没有编译tf源码时的警告（据说编译源码速度能快4-6倍）

# step 1：构造Graph 提取数据的OP
TRAIN_TFRECORD = 'F:/10-image-augmentation-set/train.tfrecords'  # 训练数据集
VAL_TFRECORD = 'F:/10-image-augmentation-set/val.tfrecords'  # 验证数据集
WIDTH = 100  # 图片的宽
HEIGHT = 100  # 图片的高
CHANNEL = 3  # 图片通道色数
TRAIN_BATCH_SIZE = 80  # 一次取多少张图片进行训练，计算能力有限只能按批次进行训练
VAL_BATCH_SIZE = 10
train_img, train_label = read_and_decode(TRAIN_TFRECORD, WIDTH, HEIGHT, CHANNEL)
val_img, val_label = read_and_decode(VAL_TFRECORD, WIDTH, HEIGHT, CHANNEL)
x_train_batch, y_train_batch = tf.train.shuffle_batch([train_img, train_label],  # 使用shuffle_batch随机打乱
                                                      batch_size=TRAIN_BATCH_SIZE, capacity=160000,
                                                      min_after_dequeue=159999, num_threads=32,
                                                      name='train_shuffle_batch')
x_val_batch, y_val_batch = tf.train.shuffle_batch([val_img, val_label],
                                                  batch_size=VAL_BATCH_SIZE, capacity=20000,
                                                  min_after_dequeue=19999, num_threads=32, name='val_shuffle_batch')

# step 2：构造Graph 训练和评估模型的OP（32位数据类型改为16位能减少一半的内存）
x = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNEL], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
mode = tf.placeholder(tf.string, name='mode')
step = tf.get_variable(shape=(), dtype=tf.int32, initializer=tf.zeros_initializer(), name='step')
tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, step)
logits = convNet(x, mode)  # 调用模型得到最后一层输出[batch_size*classes]
with tf.name_scope('Loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer().minimize(loss, step)  # 梯度下降
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
with tf.name_scope('Accuracy'):
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# step 3：构造Graph Tensorboard的OP
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", acc)
merged = tf.summary.merge_all()

# step 4：运算整个Graph
with tf.Session() as sess:
    summary_dir = './logs/summary/'

    # sess.run(tf.global_variables_initializer())  # 一次把所有的Variable类型初始化，加载模型入会话这行注释
    saver = tf.train.Saver(max_to_keep=1)  # 保存模型
    model_file = tf.train.latest_checkpoint(summary_dir)
    saver.restore(sess, model_file)  # 加载模型进当前会话

    # 将Tensorboard日志文件保存至本地（启动：tensorboard --logdir=E://OnlineIR/logs/summary）
    train_writer = tf.summary.FileWriter(summary_dir + 'train',
                                         sess.graph)
    valid_writer = tf.summary.FileWriter(summary_dir + 'valid')

    coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队
    max_acc = 0
    MAX_EPOCH = 20
    for epoch in range(MAX_EPOCH):
        # training
        train_step = int(160000 / TRAIN_BATCH_SIZE)  # 总共迭代800次
        train_loss, train_acc = 0, 0
        for i in range(epoch * train_step, (epoch + 1) * train_step):
            x_train, y_train = sess.run([x_train_batch, y_train_batch])
            train_summary, _, err, ac = sess.run([merged, train_op, loss, acc],
                                                 feed_dict={x: x_train, y_: y_train,
                                                            mode: learn.ModeKeys.TRAIN})
            train_loss += err
            train_acc += ac
            if (i + 1) % 100 == 0:
                train_writer.add_summary(train_summary, i)
        print("Epoch %d,train loss= %.2f,train accuracy=%.2f%%" % (
            epoch, (train_loss / train_step), (train_acc / train_step * 100.0)))

        # validation
        val_step = int(20000 / VAL_BATCH_SIZE)
        val_loss, val_acc = 0, 0
        for i in range(epoch * val_step, (epoch + 1) * val_step):
            x_val, y_val = sess.run([x_val_batch, y_val_batch])
            val_summary, err, ac = sess.run([merged, loss, acc],
                                            feed_dict={x: x_val, y_: y_val, mode: learn.ModeKeys.EVAL})
            val_loss += err
            val_acc += ac
            if (i + 1) % 100 == 0:
                valid_writer.add_summary(val_summary, i)
        print(
            "Epoch %d,validation loss= %.2f,validation accuracy=%.2f%%" % (
                epoch, (val_loss / val_step), (val_acc / val_step * 100.0)))

        # 保存精度最大的一次模型
        if val_acc > max_acc:
            max_acc = val_acc
            saver.save(sess, summary_dir + '/10-image.ckpt', epoch)
            print("模型已保存")
        print("————————————————————————————")
    coord.request_stop()
    coord.join(threads)
