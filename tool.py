import PIL.Image as pilimg
import numpy as np
from os import listdir
import tensorflow as tf
import os
import scipy.io
import cv2
import numpy as np
import pandas as pd
from tensorflow.contrib import slim
import numpy as np
import tensorflow as tf
from PIL import Image




def total_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("number of trainable parameters: %d"%(total_parameters))

def setup_summary(list):
    variables = []
    for i in range(len(list)):
        variables.append(tf.Variable(0.))
        tf.summary.scalar(list[i], variables[i])
    summary_vars = [x for x in variables]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op



def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def list_mat_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files]

def load_image(path):
    img = Image.open(path)
    return img

def preprocess_image_RF(cv_img):
    img = np.array(cv_img, dtype=np.float32)
    img = img/255
    img=np.reshape(img,(128,3018,1))
    return img

def preprocess_image_logic(cv_img):
    cv_img = cv_img.resize((128,128))
    img = np.array(cv_img, dtype=np.float32)
    img = img
    img=np.reshape(img,(128,128,1))
    return img


def preprocess_image_16bit(cv_img):
    cv_img = cv_img.resize((128,128))
    img = np.array(cv_img, dtype=np.float32)
    img = img/65535
    img=np.reshape(img,(128,128,1))
    return img
def preprocess_image(cv_img):
    img = np.array(cv_img, dtype=np.float32)
    img = img/255
    img=np.reshape(img,(128,128,1))
    return img


def preprocess_mat(cv_img,format):
    #print(cv_img['img128'])
    #print(np.shape(cv_img['img128']))
    cv_img = cv_img[format]
    #print(np.shape(cv_img))
    img = np.array(cv_img,np.float16)/255
    #print(np.shape(img))
    img=np.reshape(img,(128,3018,1))
    return img




def write_log(callback, names, logs, batch_no):
    """
    Util to write callback for Keras training
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def AUG_RF(data1,data2,data3,data4,data5,data6,data7,label,edge):
    data1 = np.reshape(data1,[128,3018])
    data2 = np.reshape(data2, [128, 3018])
    data3 = np.reshape(data3, [128, 3018])
    data4 = np.reshape(data4, [128, 3018])
    data5 = np.reshape(data5, [128, 3018])
    data6 = np.reshape(data6, [128, 3018])
    data7 = np.reshape(data7, [128, 3018])
    label = np.reshape(label, [128, 128])
    edge = np.reshape(edge, [128, 128])
    Aug_data7 =cv2.flip(data1,0)
    Aug_data6 = cv2.flip(data2, 0)
    Aug_data5 = cv2.flip(data3, 0)
    Aug_data4 = cv2.flip(data4, 0)
    Aug_data3 = cv2.flip(data5, 0)
    Aug_data2 = cv2.flip(data6, 0)
    Aug_data1 = cv2.flip(data7, 0)
    Aug_label = cv2.flip(label, 0)
    Aug_edge = cv2.flip(edge, 0)

    Aug_data1=np.reshape(Aug_data1,[1,128,3018,1])
    Aug_data2 = np.reshape(Aug_data2, [1, 128, 3018, 1])
    Aug_data3 = np.reshape(Aug_data3, [1, 128, 3018, 1])
    Aug_data4 = np.reshape(Aug_data4, [1, 128, 3018, 1])
    Aug_data5 = np.reshape(Aug_data5, [1, 128, 3018, 1])
    Aug_data6 = np.reshape(Aug_data6, [1, 128, 3018, 1])
    Aug_data7 = np.reshape(Aug_data7, [1, 128, 3018, 1])
    Aug_label = np.reshape(Aug_label, [1, 128, 128, 1])
    Aug_edge = np.reshape(Aug_edge, [1, 128, 128, 1])

    return Aug_data1,Aug_data2,Aug_data3,Aug_data4,Aug_data5,Aug_data6,Aug_data7,Aug_label,Aug_edge


from math import log10, sqrt

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
