# keras-ex005
<<<<<<< HEAD
* Mp4動画の変換のため、本編とその他を区別する方法についての実装例
チュートリアルなどのコードをそのまま使ってもメモリ消費が追いつかずエラーになる。
（GPUメモリだけでなくCPUメモリも）
* 画像を対象とした解析だけでなく、音声データも解析してみた。本編やCMの切り替え時に
0.5sの無音が入る規程があるそうだが、それを利用する
* ノイズキャンセルのデータも作成してみた。

#### 注意すること

* python、tensorflow, keras, NVIDIAの版数に注意
* gitのリモート先あり

## 各フォルダの意味や目的
### _internal
### checkpoint.xxx
### scripts
学習済モデルを使って元MP4ファイルからCMをカットしたMP4を作成する
### watermarks
学習用データ
#### 作成手順
* フォルダを新規作成する。フォルダ名は局名にちなんだものでよい。
* 学習に使うmp4ファイルを、上記フォルダに配置する
* 直下にある`get_masked_from_mp4.bat`をフォルダにCOPYする。
* `get_masked_from_mp4.bat`の内容を変更する。
* Win10コマンドとして`get_masked_from_mp4.bat`を実行する。
* `tf_keras_conv_cm_prepare_data.py`を実行する
* `watermarks.tsv`を`watermarks_NN_XXX.tsv`変更して保存する
* `watermarks_NN_XXX.tsv`すべてをcatして`watermarks.tsv`を再作成する
#### 学習手順
* `tf_keras_conv_cm_160x90x3.py`を実行する
### misc
#### log
#### grid
#### output

## root下のファイル

* tf_keras_conv_cm_160x90x3_predict.py
学習済モデルを使って本編の切り替わりを推定する
* tf_keras_conv_cm_160x90x3.py
モデルを学習する
* check_platform.py
環境チェック
* keras-ex005.code-workspace
python interpreterの設定やデバッグ時オプション指定、など

* keras_movmean_wav_51ch.py
WAVファイルを入力して5.1chのトラックに分ける。plotする
* keras_movmean_wav.py
WAVファイルを入力して移動平均した波形を出力する。plotする
* keras_fft_wav.py
WAVファイルを入力して高速フーリエ変換する
* adjust_adts_acc.py
ネットから拾ったコード

#### requirements.txt

```reuirements.txt
python: 3.8.10
tensoflow:
keras:
```

以上
=======
>>>>>>> refs/remotes/origin/master
