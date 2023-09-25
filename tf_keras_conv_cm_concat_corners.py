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

import PIL.Image

def list_pictures(directory, ext='jpg|png'):
    # for curDir, _, files in os.walk(directory):
    #     for f in files:
    #         if re.match(r'(tmplt_[\d]+\.(?:' + ext + '))', f.lower()):
    #             print(os.path.join(curDir, f))
    return [f
            for curDir, _, files in os.walk(directory) for f in files
            if re.match(r'(tmptl_[\d]+\.(?:' + ext + '))', f.lower())]

# データセット作成
# https://qiita.com/haru1977/items/17833e508fe07c004119
#
# hyperparameterをまとめてconfigに設定
config = {
    "data_dir": 'watermarks/FUJI',
}


def main(data_dir):
    print('Populating figs of corners and concat them...')

    if not data_dir:
        data_dir = config['data_dir']

    pic_l = []
    for pic in list_pictures(data_dir):
        pic_l.append(pic)
    print(len(pic_l))

    for pic in pic_l:
        # base
        dst = PIL.Image.new('RGB', (160, 90))
        # tl
        tl_pil = load_img(os.path.join(data_dir, pic))
        dst.paste(tl_pil, (0, 0))
        # bl
        bl_pil = load_img(os.path.join(data_dir, pic.replace('tmptl','tmpbl')))
        dst.paste(bl_pil, (0, 45))
        # tr
        tr_pil = load_img(os.path.join(data_dir, pic.replace('tmptl','tmptr')))
        dst.paste(tr_pil, (80, 0))
        # br
        br_pil = load_img(os.path.join(data_dir, pic.replace('tmptl','tmpbr')))
        dst.paste(br_pil, (80, 45))
        dst.save(os.path.join(data_dir, pic.replace('tmptl','tmp')))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', metavar='data_dir', type=str)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args.data_dir)