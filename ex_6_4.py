# -*- coding: utf-8 -*-
# github : https://github.com/fchollet/deep-learning-with-python-notebooks
import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


# =========================================== #
# 6.4.1     A Temperature-Forecasting Problem #
# =========================================== #


# 1. Implementing a 1D convnet

#   1) 데이터셋 준비 -----------------------------------------------------------------------------
max_features = 10000  # number of words to consider as features
max_len = 500  # cut texts after this number of words (among top max_features most common words)

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
# ----------------------------------------------------------------------------------------------

# 2) 1D convnet for the IMDB dataset
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# =========================================================== #
# 6.4.2     Combining CNNs and RNNs to process long sequences #
# =========================================================== #


# 2. Inspecting the Data ------------------------------------------------
#   0) 경로 설정
unzip_path = "./dataset"  # 각자 원하는 경로로 압축 해제 경로 지정
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

# 0. 사용하는 데이터 --------------------------------------------------------
#   1) 6.3 에서 썼던 jena 의 기후 데이터
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


# starting with two Conv1D layers and following-up with a GRU layer
model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
                        input_shape=(None, sample_array_z_scaled.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(generator_train,
                              steps_per_epoch=500,
                              epochs=20,
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


# step 값 6으로 변경
step = 6

# 변경된 설정값에 맞춰 generator 객체를 다시 생성
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

# 모델 구조 변경
# - starting with two Conv1D layers and following-up with a GRU layer
# - regularized GRU 층 1 개 짜리 모델에서 Loss 측도 개선은 없지만
# - 처리 속도가 상당히 빨라졌음
mdel = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
                        input_shape=(None, sample_array_z_scaled.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(generator_train,
                              steps_per_epoch=500,
                              epochs=20,
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


print('debug point')
