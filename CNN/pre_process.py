import tensorflow as tf
import matplotlib.image as mplimg
import glob
from skimage import io
import random

dir_name = "shoe"
files = glob.glob("F:/10-image-set2/train/" + dir_name + "/*.jpg")  # 轮循该目录下所有图片
random.shuffle(files)
i = 0
for im in files:
    with tf.Session() as sess:
        x = tf.Variable(mplimg.imread(im))
        sess.run(tf.global_variables_initializer())

        # 数据增强
        # 图像切割
        if i < 2000:
            i += 1
            io.imsave("F:/data_augmentation_set/train/" + dir_name + "/" + str(i) + ".jpg",
                      sess.run(tf.random_crop(x, [160, 160, 3])))

        # 图像翻转
        if 1999 < i < 5000:
            i += 1
            io.imsave("F:/data_augmentation_set/train/" + dir_name + "/" + str(i) + ".jpg",
                      sess.run(tf.image.random_flip_left_right(x)))

        # 图像白化
        if i > 4999:
            i += 1
            result = sess.run(tf.image.per_image_standardization(tf.cast(x, tf.float32) * (1. / 255)))
            io.imsave("F:/data_augmentation_set/train/" + dir_name + "/" + str(i) + ".jpg",
                      sess.run(tf.cast(result, tf.uint8)))

    tf.reset_default_graph()
