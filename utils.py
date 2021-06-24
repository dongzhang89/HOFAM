"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import cv2 as cv
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix


def load_data_Heatmap(image_path, flip=True, is_test=False):
    img_A, img_B, img_C , img_D = load_image_Heatmap(image_path)
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_C = img_C/127.5 - 1.
    img_D = img_D/127.5 - 1.
    img_AB = np.concatenate((img_A, img_B), axis=2)
    img_ABC = np.concatenate((img_AB, img_C), axis=2)
    img_ABCD = np.concatenate((img_ABC, img_D),axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_ABCD

def load_data(image_path, flip=True, is_test=False):
    img_A, img_B, img_C = load_image(image_path)
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_C = img_C/127.5 - 1.
    img_AB = np.concatenate((img_A, img_B), axis=2)
    img_ABC = np.concatenate((img_AB, img_C), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_ABC

def load_data2OP(image_path, flip=True, is_test=False):
    img_A, img_B, img_C = load_image2OP(image_path)
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_C = img_C/127.5 - 1.
    img_AB = np.concatenate((img_A, img_B), axis=2)
    img_ABC = np.concatenate((img_AB, img_C), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_ABC

def load_data1OP(image_path, flip=True, is_test=False):
    img_A, img_B, img_C = load_image1OP(image_path)
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_C = img_C/127.5 - 1.
    img_AB = np.concatenate((img_A, img_B), axis=2)
    img_ABC = np.concatenate((img_AB, img_C), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_ABC



def load_data_no_Heatmap(image_path, flip=True, is_test=False):
    img_A, img_B, img_C , img_D = load_image_Heatmap(image_path)
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_D = img_D/127.5 - 1.
    img_AB = np.concatenate((img_A, img_B), axis=2)
    img_ABD = np.concatenate((img_AB, img_D), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_ABD

def load_data_coco(image_path, flip=True, is_test=False):
    img_A, img_B, img_C = load_image_coco(image_path)
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_C = img_C/255.0
    img_AC = np.concatenate((img_A, img_C), axis=2)
    img_ACB = np.concatenate((img_AC, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_ACB


def load_image_Heatmap(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/4)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w2 * 2]
    img_C = input_img[:, w2 * 2:w2 * 3]
    img_D = input_img[:, w2 * 3:w]

    return img_A, img_B, img_C, img_D

def load_image_coco(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/4)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w2 * 2]
    img_C = input_img[:, w2 * 3:w]

    return img_A, img_B, img_C

def load_image2OP(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/3)
    img_A = input_img[:, 0:w2]
    img_B0 = input_img[:, w2:w2 * 2,0]
    img_B1 = input_img[:, w2:w2 * 2,1]
    img_zeros = np.copy(img_B0)
    img_zeros[img_zeros!=0] = 0
    img_B = cv.merge([img_B0,img_B1,img_zeros])
    img_C = input_img[:, w2 * 2:w]

    return img_A, img_B, img_C

def load_image1OP(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/3)
    img_A = input_img[:, 0:w2]
    img_B0 = input_img[:, w2:w2 * 2,0]
    img_zeros = np.copy(img_B0)
    img_zeros[img_zeros != 0] = 0
    img_B = cv.merge([img_B0, img_zeros, img_zeros])
    img_C = input_img[:, w2 * 2:w]

    return img_A, img_B, img_C

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/3)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w2 * 2]
    img_C = input_img[:, w2 * 2:w]

    return img_A, img_B, img_C


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.) / 2.
