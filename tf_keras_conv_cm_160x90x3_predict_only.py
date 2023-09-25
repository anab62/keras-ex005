#
"""
keras-ex005 nn model sample to cut cm on mp4 video
"""
from logging import getLogger
import logging
logging.basicConfig(level=logging.INFO,
                    # format='%(asctime)s : %(levelname)s : %(name)s : %(message)s',
                    format='[%(levelname)s] %(asctime)s : %(message)s (%(filename)s:%(lineno)d)',
                    )
logger = getLogger(__name__)
logger.setLevel(logging.INFO)
fileHandler = logging.FileHandler(filename='log/predict.log', encoding='utf-8', mode='w')
fileHandler.setLevel(logging.INFO)
logger.addHandler(fileHandler)

import configparser
# read config
config_pc = configparser.ConfigParser()
config_pc.read('config_pc.ini')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
# print(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
# from keras.preprocessing.image import load_img, array_to_img, img_to_array  # deprecated in Tf2.9.1
from keras.utils import load_img, array_to_img, img_to_array, np_utils, save_img  # deprecated in Tf2.9.1
from sklearn.model_selection import train_test_split

from tf_keras_conv_cm_160x90x3 import model, config, class_names, get_batch

import matplotlib.pyplot as plt
def plot_image(i, predictions_array, true_label, img):
    # print(f"pred_array:{predictions_array}")
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    true_label = np.argmax(true_label)
    # print(f"predicted_label:{predicted_label}")
    if predicted_label == true_label:
        color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(config['label_size']))  # 10
    plt.yticks([])
    thisplot = plt.bar(range(config['label_size']), predictions_array, color="#777777")  # 10
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    true_label = np.argmax(true_label)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def adjust_intervals(in_list):
    '''
    本編とそれ以外の期間を調整する。
    input: 予測データ。10s毎の分類結果のリスト
    output: 期間と属性からなるリスト

    実データを見ると、0と17にきちんと分類されない。当たり前だが、対象の性質から前後の流れを把握
    できればうまく分類できるのではないか。

    アルゴリズム例
    5回のうちAABCAと連続して同じ分類（A）になれば、BCをAAとする。
    その場合SEEK位置を5番目にJUMPするかどうか。
    開始と終了をどうするか。
    AAAABとなった場合どうするか。
    AAABCとなった場合どうするか。
    分類は２種類しかない想定。
    
    もっと単純に、本編を1それ以外を0として期間5で移動平均を取り、0.8で切り分けではどうか。
    https://resanaplaza.com/2021/09/20/%E3%80%90%E8%B6%85%E7%B0%A1%E5%8D%98%E3%80%91python%E3%81%A7%E7%A7%BB%E5%8B%95%E5%B9%B3%E5%9D%87%E3%82%92%E8%A8%88%E7%AE%97%E3%81%99%E3%82%8B%E3%81%AB%E3%81%AF%EF%BC%9F%EF%BC%88pandas%E3%80%81numpy/
    import numpy as np

    data = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])

    #np.ones()の値を個数で割る方法
    print('移動平均',np.convolve(data,np.ones(5)/5, mode='valid'))

    #convolveが返す結果を個数で割る方法
    print('移動平均',np.convolve(data,np.ones(5), mode='valid') / 5)
    '''

    # https://teratail.com/questions/159424
    import collections
    counted = collections.Counter(in_list)
    print(counted.most_common()[0][0])
    bs_max = counted.most_common()[0][0]

    in_list_f = []
    for l in in_list:
        if l == bs_max:
            in_list_f.append(1.0)
        else:
            in_list_f.append(0.0)
         
    data = np.array(in_list_f)
    # print(data)
    # print('移動平均',np.convolve(data,np.ones(5), mode='valid') / 5)
    data_movmean = np.convolve(data,np.ones(5), mode='valid') / 5
    
    data_final = []
    for x in data_movmean:
      if x > 0.35:
        data_final.append(1)
      else:
        data_final.append(0)
    fst = data_final[0]
    lst = data_final[-1]
    # data_final = [fst, fst] + data_final    
    data_final = data_final + [lst, lst]
    logger.info(f"adjust_intervals() data_final:{data_final}")
    return data_final
    # print(data_final)

DELIM = config_pc['SAMPLE1']['delim_char']
MP4 = config_pc['SAMPLE1']['mp4']
FFMPEG = config_pc['SAMPLE1']['ffmpeg']

def gen_ffmpeg_cmd(start=None, stop=None, count=1):
    if start is None:
       return None
    
    # return f"file 'temp{DELIM}tmp_{count:03}.mp4'", f"{FFMPEG} -ss {start} -t {stop - start} -i {MP4} -safe 0 -c copy temp{DELIM}tmp_{count:03}.mp4"
    ret_str = f"file 'temp/tmp_{count:03}.mp4'", f"{FFMPEG} -y -ss {start} -t {stop - start} -i {MP4} -c copy temp/tmp_{count:03}.mp4"
    logger.info(f"gen_ffmpeg_cmd() ret_str:{ret_str}")
    return ret_str
    # return f"file 'tmp_{count:03}.mp4'", f"{FFMPEG} -ss {start} -t {stop - start} -i {MP4} -safe 0 -c copy tmp_{count:03}.mp4"

# TEMPLATE='''@echo off
# set FFMPEG="C:\Users\user1\ffmpeg6\bin\ffmpeg.exe"
# set PYTHON="D:\Users\user1\keras-ex005\_internal\python-3.8.10\python.exe"
# echo "According to info. partition mp4 ..."
# %FFMPEG% -f concat -i filelist.txt -c copy tmp_concat.mp4
# PAUSE
# exit /b'''

PARTITION_MP4_TEMPLATE_TXT = config_pc['SAMPLE1']['partition_mp4_template_txt']
PARTITION_MP4_CMD = config_pc['SAMPLE1']['partition_mp4_cmd']

def partition_mp4(in_list):
  '''
  ffmpegを利用したコマンド列を生成する。
  input: 期間と属性からなるリスト
  output: コマンド列
  '''
  assert in_list is not None and type(in_list) == list
  MIN = 10  # seconds
  start = 0
  stop = 0
  count = 0
  status = False
  with open(PARTITION_MP4_TEMPLATE_TXT, 'r') as fp_in:
    template_partition_mp4 = fp_in.read()
  with open(PARTITION_MP4_CMD, 'w') as fp_out:
    fp_out.write(template_partition_mp4)
  with open("filelist.txt", 'w') as fp_out2:
    fp_out2.write("")
  for i, l in enumerate(in_list):
      # 本編の開始時
      if l == 1 and not status:
          start = i * MIN
          status = True
      # CMの開始時
      if l == 0 and status:
          stop = i * MIN
          status = False
          print(f"start,stop:{start} - {stop}")
          count += 1
          f, c = gen_ffmpeg_cmd(start, stop, count)
          with open(PARTITION_MP4_CMD, 'a') as fp_a:
            fp_a.write(c + '\n')
          with open("filelist.txt", 'a') as fp_a2:
            fp_a2.write(f + '\n')

def main(data_tsv='watermarks.tsv', resultFig=False):

  print('Load weights...')
  # モデルの重みを読み込み
  model.load_weights(f"./checkpoints_conv_cm_{config['input_dim']}/checkpoint")
  print('Done')

  print('Loading data...')

  X = []
  Y = []

  for line in open(data_tsv, 'r'):
      label, file_path = line[:-1].split('\t')  
      X.append(file_path)
      Y.append(int(label))

  # fmnistでもそのままの値にしていた。one-hot化していなかった。
  Y = np_utils.to_categorical(Y, config['label_size'])
  # sys.exit(1)

  # print('Loading data...train_test_split')
  # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
  X_test = X
  Y_test = Y

  print('Done.')

  assert len(class_names) == config['label_size']

  print('Predict...')
  # ひとつめのバッチ分のみ実行する
  argmax_dict = {}
  argmax_idx = 0
  for X_batch, Y_batch in get_batch(X_test, Y_test, config['batch_size']):
    predictions = model.predict_on_batch(X_batch)
    print(f"predictions[0]:{predictions[0]}")
    print(f"np.argmax:{np.argmax(predictions[0])}")
    print(f"y_test[0]:{Y_test[0]}")
    for i, p in enumerate(predictions):
       argmax_dict[argmax_idx] = np.argmax(p)
       argmax_idx += 1
  # import pprint
  # pprint.pprint(argmax_dict)
  logger.info(f"main() argmax_dict:{argmax_dict}")

  print('Done.')

  partition_mp4(adjust_intervals(argmax_dict.values()))



  if not resultFig:
     sys.exit(0)

  print('Generating predicted results ...')
  num_rows = 5
  num_cols = 5
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], Y_batch, X_batch)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], Y_batch)
  plt.tight_layout()
  plt.savefig(f"grid/predicted_image_conv_cm_{config['input_dim']}.jpg")
  # plt.show()
  plt.close()
  print('Done.')


if __name__ == '__main__':

    # adjust_intervals([0,0,0,0,17,17,17,17,17,0,0,0,0])

    # sys.exit(1)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_tsv', type=str, default='watermarks.tsv')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(data_tsv=args.data_tsv, resultFig=False)
