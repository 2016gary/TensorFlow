import tensorflow as tf


# 读取tfrecords文件
def read_and_decode(filename, width, height, channel):
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [width, height, channel])  # reshape为w*h的c通道图片
    img = tf.cast(img, tf.float16) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int64)  # 在流中抛出label张量
    return img, label
