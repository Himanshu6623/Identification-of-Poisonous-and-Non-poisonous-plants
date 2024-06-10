import tensorflow as tf
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras import models,layers
from keras.models import Sequential
import matplotlib.pyplot as plt
from IPython.display import HTML
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=50
dataset = tf.keras.preprocessing.image_dataset_from_directory("/content/drive/MyDrive/Plants")
class_names = dataset.class_names
train_size=0.7
train_ds=dataset.take(int(len(dataset)*train_size))
test_ds=dataset.skip(int(len(dataset)*train_size))
val_size=0.2
val_ds=test_ds.take(int(len(dataset)*val_size))
test_ds=test_ds.skip(int(len(dataset)*val_size))
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1/255),
])
data_augmentation = tf.keras.Sequential([
 layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
 layers.experimental.preprocessing.RandomRotation(0.2),
 layers.experimental.preprocessing.RandomZoom(0.3)
])
model=Sequential()
model.add(resize_and_rescale)
model.add(data_augmentation)
model.add(Conv2D(32,kernel_size=(3,3),input_shape=
                 (256,256,3),padding='same',activation='relu'))
model.add(MaxPooling2D())

#model.add(Dropout(0.2))
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
#model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))
model.add(Dense(12,activation='softmax'))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_ds,
    validation_data = val_ds,
    verbose=1,
    epochs=20,
)
score=model.evaluate(test_ds)
