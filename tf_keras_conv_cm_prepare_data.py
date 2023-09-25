#
"""
keras-ex005 nn model sample to cut cm on mp4 video
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
# from keras.preprocessing.image import load_img, array_to_img, img_to_array  # deprecated in Tf2.9.1
from keras.utils import load_img, array_to_img, img_to_array, np_utils, save_img
from sklearn.model_selection import train_test_split
import re

def list_pictures(directory, ext='jpg|png'):
    # for root, _, files in os.walk(directory):
    #     for f in files:
    #         if re.match(r'([\w]+\.(?:' + ext + '))', f.lower()):
    #             print(os.path.join(root, f))
    return [os.path.join(curDir, f)
            for curDir, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
    # return [os.path.join(root, f)
    #         for root, _, files in os.walk(directory) for f in files
    #         ]

# データセット作成
# https://qiita.com/haru1977/items/17833e508fe07c004119
#
# hyperparameterをまとめてconfigに設定
config = {
    "data_dir": 'watermarks/BS12',  # face only so 
}


def main(data_dir):
# print('Creating data...skipped')
    print('Creating data...')

    if not data_dir:
        data_dir = config['data_dir']

    pic_l = []
    lst = list_pictures(data_dir)
    lst.sort()
    for pic in lst:
        pic_l.append([ 0, pic ])
    print(len(pic_l))

    import csv
    with open("watermarks.tsv", 'w', newline='\n') as fp:
        csv.writer(fp, delimiter='\t').writerows(pic_l)  
    print('Done.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', metavar='data_dir', type=str)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args.data_dir)