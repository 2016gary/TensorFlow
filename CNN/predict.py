from skimage import io, transform
import glob
import tensorflow as tf
import numpy as np

from convNet import convNet
from tensorflow.contrib import learn

path = 'E:/OnlineIR/predict/'
WIDTH = 100
HEIGHT = 100
CHANNEL = 3


# step 1：读取要分类的图片
def read_img(path):
    imgs = []
    for im in glob.glob(path + '*.jpg'):
        img = io.imread(im)
        img = transform.resize(img, (WIDTH, HEIGHT, CHANNEL), mode="reflect")
        imgs.append(img)
    return np.asarray(imgs, np.float32)


# step 2：构造分类的graph
x_train = read_img(path)
x = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNEL], name='x')
logits = convNet(x, learn.ModeKeys.INFER)

# step 3：运算graph
with tf.Session() as sess:
    saver = tf.train.Saver()
    summary_dir = 'E:/OnlineIR/logs/summary/'
    model_file = tf.train.latest_checkpoint(summary_dir)
    saver.restore(sess, model_file)
    predictions = sess.run([tf.argmax(logits, 1)], feed_dict={x: x_train})
    for predict in predictions:
        if predict == 0:
            result = "自行车"
            print("识别结果：自行车")
        elif predict == 1:
            result = "书"
            print("识别结果：书")
        elif predict == 2:
            result = "瓶子"
            print("识别结果：瓶子")
        elif predict == 3:
            result = "汽车"
            print("识别结果：汽车")
        elif predict == 4:
            result = "猫"
            print("识别结果：猫")
        elif predict == 5:
            result = "电脑"
            print("识别结果：电脑")
        elif predict == 6:
            result = "狗"
            print("识别结果：狗")
        elif predict == 7:
            result = "人"
            print("识别结果：人")
        elif predict == 8:
            result = "植物"
            print("识别结果：植物")
        elif predict == 9:
            result = "鞋子"
            print("识别结果：鞋子")

# step 4：保存分类结果
file_object = open('E:/OnlineIR/result.txt', 'w+')
file_object.truncate()  # 清空文件内容
file_object.write(result)
file_object.close()
