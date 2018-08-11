#coding=utf-8

import sys
import os
import time
import random
import numpy as np
import multiprocessing as mp
import glob

import tensorflow as tf

GPU_NUM = 4  # GPU数量
BATCH_NUM = 80 * GPU_NUM  # 训练时每次迭代的喂入数据批次大小
DOWN_SAMPLE_RATIO = 8  # 数据采样
DATA_ITEM_LINES = 19


def showtime(t, s):
    print("{}:{:0.3f}".format(s, time.time() - t))


def get_path(s):
    return os.path.join(os.getcwd(), s)


