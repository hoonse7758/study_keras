# -*- coding: utf-8 -*-

from typing import Any

import os
import math
import requests
import zipfile
# import clint                # /path/to/python -m pip install clint

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
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


# 버전 확인 : 2.2.4-tf
print(f"tf.keras.__version__ = {tf.keras.__version__}")

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
# plt.plot(range(len(temperature_array)), temperature_array)  # 전체 그림
# plt.plot(range(1440), temperature_array[:1440])             # 첫 14,400분 (첫 10일)까지의 데이터
# -----------------------------------------------------------------------


# =============================== #
#       6.3.2. Preparing the Data #
# =============================== #

# 0. Need to do ----------------------------------------------------------
#   1) 각 칼럼 별 표준화 ( min-max- or z- scaling )
#   2) generator 작성
# ------------------------------------------------------------------------

# 1. Example of z-scaling - 각 칼럼 별 Z-표준화 예시 ------------------------
sample_array = csv_data_array[:200000]
sample_array_mean = sample_array.mean(axis=0)
sample_array_std = sample_array.std(axis=0)
sample_array_z_scaled = (sample_array - sample_array_mean) / sample_array_std
# 칼럼별 표준화 결과 보기 - plt.hist(sample_array_z_scaled[:, col_idx])
# ------------------------------------------------------------------------


# 2. Generator 함수 선언 ---------------------------------------------------
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
    in_data=csv_data_array,
    lookback=lookback,
    delay=delay,
    min_index=0,        # Hardcoded Row Index
    max_index=200000,   # Hardcoded Row Index
    batch_size=batch_size,
    step=step,
    shuffle=True
)
generator_valid = data_generator(
    in_data=csv_data_array,
    lookback=lookback,
    delay=delay,
    min_index=200001,       # Hardcoded Row Index
    max_index=300000,       # Hardcoded Row Index
    batch_size=batch_size,
    step=step,
    shuffle=True
)
valid_steps = (300000 - 200001 - lookback)  # Hardcoded
generator_test = data_generator(
    in_data=csv_data_array,
    lookback=lookback,
    delay=delay,
    min_index=300001,       # Hardcoded Row Index
    max_index=None,         # Hardcoded Row Index
    batch_size=batch_size,
    step=step,
    shuffle=True
)
test_steps = (len(csv_data_array) - 300001 - lookback)  # Hardcoded
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


# evaluate_naive_method()
# -----------------------------------------------------------------------


# ======================================= #
#       6.3.4. Basic ML Baseline Approach #
# ======================================= #

# 0. 의미와 목적 ----------------------------------------------------------
#   0) 의미 : 작고 간단한 도구부터 활용해 보자는 것
#   1) 보다 복잡한 정의를 갖는 모델을 시험해 보기 전에 간단한 것부터 시도해 보자
# -----------------------------------------------------------------------


# 1. Dense Layer 2개 테스트 ----------------------------------------------
simple_model = Sequential()
simple_model.add(layers.Flatten(input_shape=()))
simple_model.add(layers.Dense(32, activation='relu'))
simple_model.add(layers.Dense(1))
simple_model.compile(optimizer=RMSprop(), loss='mae')
simple_set_train = next(generator_train)
simple_model_history = simple_model.fit(
    generator_train,
    steps_per_epoch=500, epochs=20,
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


print("debug point")
