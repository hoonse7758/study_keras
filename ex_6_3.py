# -*- coding: utf-8 -*-
# github : https://github.com/fchollet/deep-learning-with-python-notebooks

from typing import Any

import os
import math
import requests               # /path/to/python -m pip install requests
import zipfile
# import clint                # /path/to/python -m pip install clint

import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from matplotlib import pyplot as plt

# 예제 - 기후 예측 문제 ( Time Series )

# 얻어갈 기술들
#   1) Recurrent Dropout
#       - Fight Overfitting
#   2) Stacking Recurrent Layers
#       - Increase Representational Power
#   3) Bidirectional Recurrent Layers
#       - Increase Accuracy
#       - Mitigate Forgetting Issue


# 버전 확인 : 2.0.8
print(f"keras.__version__ = {keras.__version__}")



# =========================================== #
# 6.3.1     A Temperature-Forecasting Problem #
# =========================================== #



# 1. 데이터셋 취득 -------------------------------------------------------
#   0) 경로 설정
dataset_url = "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip"
download_path = "./downloaded_dataset.zip"  # 각자 원하는 경로로 다운로드 경로 지정
unzip_path = "./dataset"  # 각자 원하는 경로로 압축 해제 경로 지정

#   1) Download
if not os.path.exists(download_path):  # 해당 경로에 동일 파일이 없어야 다운로드
    response = requests.get(dataset_url, stream=True)
    with open(download_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

#   2) Unzip
with zipfile.ZipFile(download_path) as zip_file:
    zip_file.extractall(unzip_path)
# -----------------------------------------------------------------------


# 2. Inspecting the Data ------------------------------------------------
#   0) 경로 설정
csv_path = os.path.join(unzip_path, 'jena_climate_2009_2016.csv')

#   1) 데이터 읽어오기
with open(csv_path, 'r') as csv_file:
    csv_data = csv_file.read()

#   2) 헤더, 행데이터 단위로 분할하기
csv_data_lines = csv_data.split('\n')
csv_data_header = csv_data_lines[0].split(',')
csv_data_rows = csv_data_lines[1:]

#   3) Numpy Array 로 변환하기 - 첫 번째 [Date Time] 칼럼을 뺀 나머지 부분만
csv_data_array = np.zeros((len(csv_data_rows), len(csv_data_header) - 1))
for i, row in enumerate(csv_data_rows):
    row_values = list(map(float, row.split(',')[1:]))
    csv_data_array[i, :] = row_values
# -----------------------------------------------------------------------


# 3. Plotting the Data --------------------------------------------------
#   1) [T (degC)]
temperature_array = csv_data_array[:, csv_data_header.index('"T (degC)"') - 1]
plt.plot(range(len(temperature_array)), temperature_array)  # 전체 그림
plt.plot(range(1440), temperature_array[:1440])             # 첫 14,400분 (첫 10일)까지의 데이터
# -----------------------------------------------------------------------



# =============================== #
#       6.3.2. Preparing the Data #
# =============================== #

# 0. Need to do ----------------------------------------------------------
#   1) 각 칼럼 별 표준화 ( min-max- or z- scaling )
#   2) generator 작성
# ------------------------------------------------------------------------

# 1. Example of z-scaling - 각 칼럼 별 Z-표준화 예시 ------------------------
sample_array = csv_data_array
sample_array_mean = sample_array.mean(axis=0)
sample_array_std = sample_array.std(axis=0)
sample_array_z_scaled = (sample_array - sample_array_mean) / sample_array_std
# 칼럼별 표준화 결과 보기 - plt.hist(sample_array_z_scaled[:, col_idx])
# ------------------------------------------------------------------------


# 2. Generator 생성기 ---------------------------------------------------
def data_generator(
        in_data: np.ndarray,
        lookback: int, delay: int,
        min_index: int, max_index: Any,
        batch_size: int = 128, step: int = 6,
        shuffle: bool = False):
    """

    :param in_data:
    :param lookback:
    :param delay:
    :param min_index:
    :param max_index:
    :param batch_size:
    :param step:
    :param shuffle:
    :return:
    """
    if max_index is None:
        max_index = len(in_data) - delay - 1
    idx = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if idx + batch_size >= max_index:
                idx = min_index + lookback
            rows = np.arange(idx, min(idx + batch_size, max_index))
            idx += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            in_data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = in_data[indices]
            targets[j] = in_data[rows[j] + delay][1]
        yield samples, targets

# ------------------------------------------------------------------------


# 3. train/test/validation 데이터셋 각각에 대한 Generator 객체 생성 -----------
lookback = 1440
step = 6
delay = 144
batch_size = 128
generator_train = data_generator(
    in_data=sample_array_z_scaled,
    lookback=lookback,
    delay=delay,
    min_index=0,        # Hardcoded Row Index
    max_index=200000,   # Hardcoded Row Index
    batch_size=batch_size,
    step=step,
    shuffle=True
)
generator_valid = data_generator(
    in_data=sample_array_z_scaled,
    lookback=lookback,
    delay=delay,
    min_index=200001,       # Hardcoded Row Index
    max_index=300000,       # Hardcoded Row Index
    batch_size=batch_size,
    step=step,
    shuffle=True
)
valid_steps = 1000  # Hardcoded     원래 책에서는 (300000 - 200001 - lookback) 사용함, 98559 번 돌도록 ...
generator_test = data_generator(
    in_data=sample_array_z_scaled,
    lookback=lookback,
    delay=delay,
    min_index=300001,       # Hardcoded Row Index
    max_index=None,         # Hardcoded Row Index
    batch_size=batch_size,
    step=step,
    shuffle=True
)
test_steps = 1000  # Hardcoded     원래 책에서는 (len(csv_data_array) - 300001 - lookback) 사용함, 98559 번 돌도록 ...
# ------------------------------------------------------------------------


# ============================================= #
#       6.3.3. Non-ML Baseline - a common-sense #
# ============================================= #

# 0. 의미와 목적 ----------------------------------------------------------
#   0) 의미 : 내일의 날씨는 오늘 날씨와 (크게) 다르지 않을 것이다
#   1) 분류 기준선으로서 참고하기 위해
# -----------------------------------------------------------------------

# 1. Evaluation by Naive Method -----------------------------------------
def evaluate_naive_method():    # 시간 많이 오래걸림... 의미만 짧게 말하고 넘어가는 게..?
    batch_maes = []
    for step in range(valid_steps):
        samples, targets = next(generator_valid)
        preds = samples[:, -1, 1]               # 내일 날씨는 오늘 23시 50분의 날씨와 다르지 않을 것이다
        mae = np.mean(np.abs(preds - targets))  # Mean Absolute Error
        batch_maes.append(mae)
        idx_str = '%0{}d'.format(math.floor(math.log(valid_steps, 10)) + 1) % step
        print(f"batch {idx_str}/{valid_steps} - {mae}")
    print(f"batch_maes = ({np.mean(batch_maes)}) / {batch_maes}")


evaluate_naive_method()
# -----------------------------------------------------------------------


# ======================================= #
#       6.3.4. Basic ML Baseline Approach #
# ======================================= #

# 0. 의미와 목적 ----------------------------------------------------------
#   0) 작고 간단한 도구부터 활용해 보자는 것
#   1) 보다 복잡한 정의를 갖는 모델을 시험해 보기 전에 기본적이고 간단한 것부터 시도해 보자
# -----------------------------------------------------------------------


# 1. Dense Layer 2개 테스트 ----------------------------------------------
simple_model = Sequential()
simple_model.add(layers.Flatten(input_shape=(lookback // step, sample_array_z_scaled.shape[-1])))
simple_model.add(layers.Dense(32, activation='relu'))
simple_model.add(layers.Dense(1))
simple_model.compile(optimizer=RMSprop(), loss='mae')
simple_set_train = next(generator_train)
simple_model_history = simple_model.fit_generator(
    generator_train,
    steps_per_epoch=100,      # 원래 500번, 100 번만 하도록 수정
    epochs=10,                # 원래 40번, 10 번만 하도록 수정
    validation_data=generator_valid,
    validation_steps=valid_steps
)
simple_model_loss = simple_model_history.history['loss']
simple_model_val_loss = simple_model_history.history['val_loss']
simple_model_epochs = range(1, len(simple_model_loss) + 1)
plt.figure()
plt.plot(simple_model_epochs, simple_model_loss, 'bo', label='Training Loss')
plt.plot(simple_model_epochs, simple_model_val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# -----------------------------------------------------------------------


# ======================================= #
#       6.3.5. A first recurrent baseline #
# ======================================= #

# 0. 의미와 목적 ----------------------------------------------------------
#   0) 6.3.4 에서 사용한 기본 구조는 그리 좋은 결과가 나오지 않았다
#   1) 현재 데이터는 시간 순서가 있는 Sequence 이다. = 인과관계(causality), 순서(order) 가 중요하다
#   2) 그렇다면 이런 특징에 맞는 구조를 갖는 층을 써보면 된다 = Recurrent Layer
#   3) LSTM 이 아닌 GRU 를 쓸 것
#   4) 6.3.4 보다야 훨씬 나은 결과가 나오기는 하나, Overfitting 됨 -> 6.3.6 에서 해결
# -----------------------------------------------------------------------
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, sample_array_z_scaled.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(generator_train,
                              steps_per_epoch=100,      # 원래 500번, 100 번만 하도록 수정
                              epochs=5,    # 원래 40번, 5번만 하도록 수정
                              validation_data=generator_valid,
                              validation_steps=valid_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, 1 + len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



# ========================================================= #
#       6.3.6. Using recurrent dropout to fight overfitting #
# ========================================================= #

# 0. 의미와 목적 ----------------------------------------------------------
#   0) 의미 : Recurrent Layer (GRU) 에 dropout 값을 설정하여 Overfitting 문제를 개선해 보자
# -----------------------------------------------------------------------
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, sample_array_z_scaled.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(generator_train,
                              steps_per_epoch=100,      # 원래 500번, 100 번만 하도록 수정
                              epochs=5,    # 원래 40번, 5 번만 하도록 수정
                              validation_data=generator_valid,
                              validation_steps=valid_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, 1 + len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



# ====================================== #
#       6.3.7. Stacking recurrent layers #
# ====================================== #

# 0. 의미와 목적 ----------------------------------------------------------
#   0) GRU 층을 여러 개 사용하면 Loss 측도가 더 개선될 수 있는지 알아보자
#   1) 하나만 썼을 때랑 비교해서 Dramatic 하게 개선되진 않은 것으로 보여짐
# -----------------------------------------------------------------------
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, sample_array_z_scaled.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(generator_train,
                              steps_per_epoch=100,      # 원래 500번, 100 번만 하도록 수정
                              epochs=5,    # 원래 40번, 5 번만 하도록 수정
                              validation_data=generator_valid,
                              validation_steps=valid_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# ===================================== #
#       6.3.8. Using bidirectional RNNs #
# ===================================== #

# 0. 의미와 목적 ----------------------------------------------------------
#   *) the Swiss army knife of deep learning for NLP
#   0) 본래 정의된 정방향 순서와 reverse 된 데이터셋을 둘 다 사용해 보자
#   1) 적어도 NLP 문제에서만큼은 reverse 된 데이터셋을 쓰더라도 문제가 없다
#      ( 오히려 새로운 관점의 해석을 보여주기 때문에 모델 성능 향상에 도움이 됨 )
#   2) 이런 점을 이용해서 양방향으로 학습해 보자
# -----------------------------------------------------------------------

# 2. 데이터 역순 Generator 생성기 ---------------------------------------------------
def reverse_order_generator(data, lookback, delay, min_index, max_index,
                            shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples[:, ::-1, :], targets

# 3. train/validation 데이터셋 각각에 대한 역순 데이터 생성용 Generator 객체 할당 -----------
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen_reverse = reverse_order_generator(
    sample_array_z_scaled,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000,
    shuffle=True,
    step=step,
    batch_size=batch_size)
train_steps = 1000  # Hardcoded     원래 책에서는 98559 번 돌도록 ...
val_gen_reverse = reverse_order_generator(
    sample_array_z_scaled,
    lookback=lookback,
    delay=delay,
    min_index=200001,
    max_index=300000,
    step=step,
    batch_size=batch_size)
valid_steps = 1000  # Hardcoded     원래 책에서는 98559 번 돌도록 ...


# 4. jena 의 기후 데이터셋 시간 순서를 역순으로 뒤집고 가장 단순한 GRU 1층짜리 모델로 학습 시도 --------------------
# 결과 : 본래 순서 데이터를 썼을 때보다 못함
# 의미 : 시간 순서가 굉장히 중요했던 것이었다
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, sample_array_z_scaled.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen_reverse,
                              steps_per_epoch=100,      # 원래 500번, 100 번만 하도록 수정
                              epochs=5,    # 원래 40번, 5 번만 하도록 수정
                              validation_data=val_gen_reverse,
                              validation_steps=valid_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# 5. 오랜만에 imdb --------------------------------------------------------------------------------------
# 여기부터 jena 데이터 안씀
# - allow_pickle 이 False 면 저장할 수 없다는 에러가 날 경우
#   : \site-packages\keras\datasets\imdb.py 의 load_data() 의 52 번째 라인을
#   : with np.load(path, allow_pickle=True) as f: 로 변경

# Number of words to consider as features
max_features = 10000
# Cut texts after this number of words (among top max_features most common words)
maxlen = 500
# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# Reverse sequences - 순서가 뒤집힌 데이터셋을 사용
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]
# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


# 6. 각 문서 내 단어들의 순서가 뒤집힌 imdb 데이터셋에 Recurrent 모델을 사용해 보자 -------------------------------
# 결과 : word order does matter in understanding language, which order you use isn't crucial.
# 역순 데이터를 사용하면 기존의 정방향 데이터를 사용했을 때와는 또다른 형식(관점?)의 표현(해석?)을 볼 수 있다
# [저자의 말]
# Importantly, a RNN trained on reversed sequences will learn different representations
# than one trained on the original sequences
# ... they offer a new angle from which to look at your data,
# capturing aspects of the data that were missed by other approaches,
# and thus they can allow to boost performance on a task.
# This is the intuition behind "ensembling"
# ---------------------------code---------------------------
model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# 7. Bidirectional 모델을 실제로 사용해 보자 ------------------------------------------------------------

# 백엔드 세션 정리
from keras import backend as K
K.clear_session()

# Bidirectional 모델 정의 및 학습
# - val 정확도가 개선되었으나, 빠르게 Overfitting 되고있음
# - 약간의 정규화(regularization) 를 통해 이를 해결한다면
# - Bidirectional 은 NLP 문제에서 상당히 쓸만한 모델이 될 것
model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# ----------------------------------------------------------------------------------------------------

print("debug point")
