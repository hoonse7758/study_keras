# -*- coding: utf-8 -*-

from tensorflow.keras.applications import VGG16


# Let's instantiate the VGG16 model:
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

conv_base.summary()
