import os
import tensorflow as tf
from PIL import Image

# data_path = 'F:/10-image-set2/train/'  # 训练数据集
data_path = 'F:/10-image-set2/val/'  # 测试数据集´
classes = ['bicycle', 'book', 'bottle', 'car', 'cat', 'computer', 'dog', 'person', 'plant', 'shoe']  # 人为设定类别
# writer = tf.python_io.TFRecordWriter("F:/10-image-set2/train.tfrecords")  # 要生成的tfrecord文件
writer = tf.python_io.TFRecordWriter("F:/10-image-set2/val.tfrecords")  # 要生成的tfrecord文件
WIDTH = 100  # 图片的宽
HEIGHT = 100  # 图片的高

for index, dir_name in enumerate(classes):
    class_path = data_path + dir_name + '/'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name  # 每一个类中的每一张图片的路径
        img = Image.open(img_path)
        img = img.resize((WIDTH, HEIGHT), Image.NEAREST)
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),  # 以路径的序号作为label
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串

writer.close()
