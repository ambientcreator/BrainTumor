import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input
import numpy as np
import time
import os

start_time = time.time()

data_dir = r"path_to_dataset_directory"

image_size = (224, 224)
batch_size = 64

if not os.path.exists(data_dir):
    raise ValueError(f"директория не найдена: {data_dir}")

subdirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
if not subdirs:
    raise ValueError(f"не найдено дочерних директорий {data_dir}")

print(f"найдено {len(subdirs)} дочерних директории {data_dir}")
for subdir in subdirs:
    print(f"- {os.path.basename(subdir)}")

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=batch_size,
    color_mode='rgb',
    validation_split=0.2,
    subset='training',
    seed=123,
    shuffle=True,
    interpolation='bilinear'
)

def process_image(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, len(subdirs))
    return image, label

train_dataset = train_dataset.map(process_image).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

inputs = Input(shape=(224, 224, 1))
x = Flatten()(inputs)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(len(subdirs), activation='softmax')(x)
outputs = x
simple_nn = keras.Model(inputs=inputs, outputs=outputs)

class Model(tf.keras.Model):
    def __init__(self, nn):
        super(Model, self).__init__()
        self.nn = nn

    def get_loss(self, y, preds):
        loss = tf.keras.losses.CategoricalCrossentropy()(y, preds)
        return loss

    def training(self, x, y):
        with tf.GradientTape() as tape:
            preds = self.nn(x, training=True)
            loss = self.get_loss(y, preds)

        gradients = tape.gradient(loss, self.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_variables))
        return tf.reduce_mean(loss)

model = Model(simple_nn)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

for x, y in train_dataset.take(1):
    print(model.training(x, y))

end_time = time.time()
print(f"тренировочное время: {end_time - start_time} секунд")