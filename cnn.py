import keras
import tensorflow
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras.callbacks import TensorBoard,ModelCheckpoint

# ハイパーパラメータの設定
hp1 = {}
hp1['class_num'] = 2 # 分類するクラスの数
hp1['batch_size'] = 5 # 一度に処理する画像の数
hp1['epoch'] = 10 # 訓練の繰り返し回数

# データセットの読み込み
X_train, X_test, y_train, y_test = np.load("./dataset.npy", allow_pickle=True)

# 入力データの形状を設定
input_shape = X_train.shape[1:]

# CNNモデルの定義
def CNN(input_shape):
    model = Sequential()
    # 畳み込み層、活性化関数、プーリング層、ドロップアウト層を追加してネットワークを構築
    # 最後に全結合層を追加し、softmax関数でクラス分類を行う
    return model

# モデルの構築
model = CNN(input_shape)

# モデルのコンパイル
# 損失関数、最適化アルゴリズム、評価指標を設定
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# 訓練時のコールバック設定
log_dir = os.path.join(os.path.dirname(__file__), "logdir")
model_file_name = "model_file.hdf5"

# モデルの訓練
# 訓練データと検証データに分け、エポック数だけ訓練を繰り返す
# TensorBoardとModelCheckpointをコールバックとして使用
history = model.fit(
    X_train, y_train,
    epochs=hp1['epoch'],
    validation_split=0.2,
    callbacks=[
        TensorBoard(log_dir=log_dir),
        ModelCheckpoint(os.path.join(log_dir, model_file_name), save_best_only=True)
    ]
)

# モデルの評価
# テストデータセットを使用して損失と精度を計算
loss, accuracy = model.evaluate(X_test, y_test, batch_size=hp1['batch_size'])