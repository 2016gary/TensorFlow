import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
import glob
from skimage import io, transform

files = glob.glob('F:/10-image-set2/train/book/*.jpg')  # 轮循该目录下所有图片
i = 0
for im in files:
    image_raw_data = tf.gfile.FastGFile(im, 'rb').read()

    image = mpimg.imread(im)
    height, width, depth = image.shape
    x = tf.Variable(image, name='x')
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)

        im1 = tf.image.adjust_brightness(img_data, 0.5)  # 图片亮度，减小0.5
        i += 1
        io.imsave("F:/new_set/book/" + str(i) + ".jpg", im1.eval())

        im2 = tf.image.adjust_contrast(img_data, -5)  # 图片对比度，-5
        i += 1
        io.imsave("F:/new_set/book/" + str(i) + ".jpg", im2.eval())

        im3 = tf.image.random_hue(img_data, 0.5)  # 图片色相，[-max_delta, max_delta]范围内随机调整。max_delta取值在[0, 0.5]
        i += 1
        io.imsave("F:/new_set/book/" + str(i) + ".jpg", im3.eval())

        sess.run(init)

        im4 = tf.transpose(x, perm=[1, 0, 2])  # 图片转置
        result1 = sess.run(im4)
        i += 1
        io.imsave("F:/new_set/book/" + str(i) + ".jpg", result1)

        im5 = tf.image.crop_to_bounding_box(x, 0, 0, 128, 128)  # 图片裁切，左上角
        result2 = sess.run(im5)
        i += 1
        io.imsave("F:/new_set/book/" + str(i) + ".jpg", result2)

        im6 = tf.image.crop_to_bounding_box(x, 128, 0, 128, 128)  # 图片裁切，右上角
        result3 = sess.run(im6)
        i += 1
        io.imsave("F:/new_set/book/" + str(i) + ".jpg", result3)

        im7 = tf.image.crop_to_bounding_box(x, 0, 128, 128, 128)  # 图片裁切，左下角
        result4 = sess.run(im7)
        i += 1
        io.imsave("F:/new_set/book/" + str(i) + ".jpg", result4)

        im8 = tf.image.crop_to_bounding_box(x, 128, 128, 128, 128)  # 图片裁切，右上角
        result5 = sess.run(im8)
        i += 1
        io.imsave("F:/new_set/book/" + str(i) + ".jpg", result5)
