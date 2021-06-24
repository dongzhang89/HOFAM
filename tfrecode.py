import os
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
cwd = '/data/tmpu1/WZQ/Test/dataset/dataset2/'
classes = {'train'}  # 人为设定2类
# cwd = '/data/tmpu1/WZQ/dataset2014/dataset_Preprocessing/merge/'
# classes = {'train'}  # 人为设定2类
writer = tf.python_io.TFRecordWriter("CDnet_train_merge3OP_200806_tmp.tfrecords")  # 要生成的文件


def extract_image(filename):
    image = cv.imread(filename)

    b, g, r = cv.split(image)
    rgb_image = cv.merge([r, g, b])
    return rgb_image

f = open('before/Target_num.txt','r')
readlines = f.readlines()
dictt = {}
for line in readlines:
    line = line.split('\n')[0]
    name,num = line.split(',')
    dictt[name] = int(num)
f.close()


for index, name in enumerate(classes):
    class_path = cwd + name + '/'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name  # 每一个图片的地址

        img = extract_image(img_path)

        target_num = dictt[img_name]
        print(img_name,target_num)
        img_raw = img[:, 0:256, :]
        img_flow = img[:, 256:512, :]
        img_gt = img[:,512:768,0]

        img_gt[img_gt > 200] = 255
        img_gt[img_gt <= 200] = 0

        gt_fg = np.uint8(cv.GaussianBlur(img_gt, (7, 7), 2.))
        gt_fg[gt_fg > 200] = 255
        gt_fg[gt_fg <= 200] = 0

        gt_bg = np.uint8(cv.GaussianBlur(255. - img_gt, (7, 7), 2.))
        gt_bg[gt_bg > 200] = 255
        gt_bg[gt_bg <= 200] = 0
        # cv.imshow('1.jpg', img_gt)
        # cv.imshow('2.jpg', img_raw)
        # img = img.resize((256, 512))
        img_raw = img_raw.tobytes()  # 将图片转化为二进制格式
        img_flow = img_flow.tobytes()
        # img_flowL = img_flowL.tobytes()
        img_gt = img_gt.tobytes()
        gt_fg = gt_fg.tobytes()
        gt_bg = gt_bg.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            "target_num": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(target_num)])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'img_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_flow])),
            # 'img_flowL': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_flowL])),
            'img_gt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_gt]))
            #'gt_fg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_fg])),
            #'gt_bg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_bg]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串

writer.close()









# import os
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2 as cv
# cwd = '/data/tmpu1/WZQ/Test/dataset/coco_merge/'
# classes = {'train_1k'}  # 人为设定2类
# writer = tf.python_io.TFRecordWriter("Coco_1k.tfrecords")  # 要生成的文件
#
#
# def extract_image(filename):
#     image = cv.imread(filename)
#
#     b, g, r = cv.split(image)
#     rgb_image = cv.merge([r, g, b])
#     return rgb_image
#
# for index, name in enumerate(classes):
#     class_path = cwd + name + '/'
#     for img_name in os.listdir(class_path):
#         img_path = class_path + img_name  # 每一个图片的地址
#
#         img = extract_image(img_path)
#
#         img_raw = img[:, 0:256, :]
#         img_flow = img[:, 256:512, 0]
#         img_gt = img[:, 512:768, 0]
#
#         img_gt[img_gt > 200] = 255
#         img_gt[img_gt <= 200] = 0
#
#         gt_fg = np.uint8(cv.GaussianBlur(img_gt, (7, 7), 2.))
#         gt_fg[gt_fg > 200] = 255
#         gt_fg[gt_fg <= 200] = 0
#
#         gt_bg = np.uint8(cv.GaussianBlur(255. - img_gt, (7, 7), 2.))
#         gt_bg[gt_bg > 200] = 255
#         gt_bg[gt_bg <= 200] = 0
#         # cv.imshow('1.jpg', img_gt)
#         # cv.imshow('2.jpg', img_raw)
#         # img = img.resize((256, 512))
#         img_raw = img_raw.tobytes()  # 将图片转化为二进制格式
#         img_flow = img_flow.tobytes()
#         img_gt = img_gt.tobytes()
#         gt_fg = gt_fg.tobytes()
#         gt_bg = gt_bg.tobytes()
#
#         example = tf.train.Example(features=tf.train.Features(feature={
#             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
#             'img_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_flow])),
#             'img_gt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_gt])),
#             'gt_fg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_fg])),
#             'gt_bg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_bg]))
#         }))  # example对象对label和image数据进行封装
#         writer.write(example.SerializeToString())  # 序列化为字符串
#
# writer.close()