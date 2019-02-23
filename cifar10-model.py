import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Activation
from tensorflow.keras.datasets.cifar10 import load_data

train_tuple, test_tuple = load_data()
train = [train_tuple[0], train_tuple[1]]
test = [test_tuple[0], test_tuple[1]]
train[1] = tf.keras.utils.to_categorical(train[1], 10)
test[1] = tf.keras.utils.to_categorical(test[1], 10)
"""
print(train[0].shape)
print(train[1].shape)
print(test[0].shape)
print(test[1].shape)
"""

from tensorflow.keras.models import Sequential

model = Sequential()

model.add(Conv2D(64, (3,3), padding='same', input_shape = train[0].shape[1:]))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(16, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x = train[0],
    y = train[1],
    epochs = 100,
    batch_size = 128,
    validation_data = (test[0], test[1])
)
