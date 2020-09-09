# coding: utf-8
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parse_anchors, load_weights

num_class = 20#80
img_size = 416
weight_path = './data/darknet_weights/voc.weights/Epoch_32_step_91046_mAP_0.8754_loss_2.2147_lr_3e-05.data-00000-of-00001' #yolov3.weights'
save_path = './data/darknet_weights/yolov3.ckpt'
anchors = parse_anchors('./misc/experiments_on_voc/voc_anchors')#'./data/yolo_anchors.txt')

model = yolov3(0, anchors)
with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

    with tf.variable_scope('yolov3'):
        feature_map = model.forward(inputs)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

    load_ops = load_weights(tf.global_variables(scope='yolov3'), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))



