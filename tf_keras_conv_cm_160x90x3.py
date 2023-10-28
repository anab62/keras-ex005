#
"""
keras-ex005 nn model sample to cut cm on mp4 video
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
# from keras.preprocessing.image import load_img, array_to_img, img_to_array  # deprecated in Tf2.9.1
from keras.utils import load_img, array_to_img, img_to_array, np_utils, save_img  # deprecated in Tf2.9.1
from sklearn.model_selection import train_test_split
import re

# hyperparameterをまとめてconfigに設定
config = {
    "image_dim": (160,90),  # 64
    "input_dim": "160x90x3",  # 28x28x1
    "data_dir": 'watermarks',  # face only so 
    "epochs": 10,
    "batch_size": 128,  # 128
    "label_size": 23,  # 23
    # "model_hdf5" : 'model_vae.hdf5',
}

class_names = ['CM', 
               'FOX', 'SUP', 'STAR', 'WPLUS', 'BS11',
               'DLIFE', 'ANIMAL', 'DISCOVER', 'HIST', 'MPLUS',
               'NATIO', 'NECO', 'NIHON', 'TOUEI', 'FUJI',
               'NPNTV', 'TBS', 'TKYMX', 'TVASAHI', 'BSNTV',
               'BS12', 'ACTION'
               ]
assert len(class_names) == config['label_size']

print('Building a model...')
# inputs = keras.Input(shape=(640, 360, 3))
inputs = keras.Input(shape=(160, 90, 3))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
# outputs = layers.Dense(2, activation="sigmoid")(x)
outputs = layers.Dense(config['label_size'], activation="softmax")(x)
model = keras.Model(inputs, outputs, name="cnn")
model.summary()

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            # loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
print('Done.')

'''
# https://aotamasaki.hatenablog.com/entry/2018/08/27/124349
# fit_generatorと考え方は同じだろう。ちょっとだけ楽になるのかな。
'''
def get_batch(X, Y, batch_size):
    # global X_train, Y_train
    SIZE = len(X)
    # n_batches
    n_batches = (SIZE + batch_size - 1) // batch_size
    # yield on for loop
    i = 0
    while(i < n_batches):
        print("doing", i, "/", n_batches)
        Y_batch = Y[(i * batch_size):(i * batch_size + batch_size)]
        X_batch_name = X[(i * batch_size):(i * batch_size + batch_size)]
        X_batch = []
        for file_path in X_batch_name:
            img = img_to_array(load_img(file_path, target_size=(config['image_dim'][0],config['image_dim'][1])))
            X_batch.append(img)
        X_batch = np.asarray(X_batch)
        X_batch = X_batch.astype('float32')
        X_batch = X_batch / 255.0

        i += 1
        yield X_batch, Y_batch


if __name__ == '__main__':
    # model.fit(X_train[:500], Y_train[:500], epochs=config['epochs'], batch_size=config['batch_size'])

    # print('Creating data...skipped')
    print('Loading data...')

    X = []
    Y = []

    for line in open("watermarks.tsv", 'r'):
        label, file_path = line[:-1].split('\t')  
        X.append(file_path)
        Y.append(int(label))

    # fmnistでもそのままの値にしていた。one-hot化していなかった。
    Y = np_utils.to_categorical(Y, config['label_size'])
    # sys.exit(1)

    print('Loading data...train_test_split')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    print('Done.')

    n_epoch = config['epochs']
    for epoch in range(n_epoch):
        print('=' * 50)
        print(epoch, '/', n_epoch)
        
        acc = []    
        for X_batch, Y_batch in get_batch(X_train, Y_train, config['batch_size']):
            model.train_on_batch(X_batch, Y_batch)
            score = model.evaluate(X_batch, Y_batch)
            print(f"batch accuracy:{score[1]}")
            acc.append(score[1])
        print(f"Train accuracy:{np.mean(acc)}")

    # score = model.evaluate(X_test, Y_test)
    # print(f"Test loss:{score[0]}")
    # print(f"Test accuracy:{score[1]}")

    # print('Save weights...skipped')
    print('Save weights...')
    # モデルを保存
    model.save_weights(f"./checkpoints_conv_cm_{config['input_dim']}/checkpoint")
    print('Done')
