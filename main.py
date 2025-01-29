import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input
import numpy as np
import time
import os
import flet as ft
from PIL import Image

def predict(model, image_patch):
    img = Image.open(image_patch).convert("L")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])

    prediction = model(img_array, training=False),
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    confidence = tf.reduce_max(prediction).numpy()

    return predicted_class, confidence

def train_model(data_dir):
    start_time = time.time()

    image_size = (224, 224)
    batch_size = 64

    if not os.path.exists(data_dir):
        return f"директория не найдена: {data_dir}"

    subdirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    if not subdirs:
        return f"не найдено дочерних директорий в {data_dir}"

    print(f"найдено {len(subdirs)} дочерних директорий {data_dir}")
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
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=10)

    end_time = time.time()
    training_time = end_time - start_time

    model.save('brain_tumor_model.keras')

    return f"тренировочное время: {training_time} секунд"


def flet_app(page:ft.Page):
    page.title = "BrainTrumor"

    data_dir = ft.Ref[ft.TextField]()
    result_text = ft.Ref[ft.Text]()
    image_patch = ft.Ref[ft.TextField]()
    prediction_result = ft.Ref[ft.Text]()

    def start_training(e):
        if not data_dir.current.value:
            result_text.current.value = "укажите путь к датасету"
        else:
            result_text.current.value = "тренировка началась"
            page.update()

            result = train_model(data_dir.current.value)

            result_text.current.value = result
        page.update()

    def make_prediction(e):
        global trained_model
        trained_model = keras.models.load_model(
            'brain_tumor_model.keras'
        )
        if trained_model is None:
            prediction_result.current.value = "модель не обучена"
        elif not image_patch.current.value:
            prediction_result.current.value = "укажите путь к изображению"
        else:
            try:
                class_id, confidence = predict(trained_model, image_patch.current.value)
                prediction_result.current.value = f"предсказание: {class_id}, уверенность: {confidence:.2f}"
                print(class_id, confidence)
            except Exception as ex:
                prediction_result.current.value = f"ошибка предсказания: {str(ex)}"


    page.add(
        ft.TextField(ref=data_dir, label="путь к датасету",),
        ft.FilledButton(text="начать тренировку", on_click=start_training),
        ft.Text(ref=result_text),
        ft.TextField(ref=image_patch, label="путь к изображению"),
        ft.FilledButton(text="предсказать", on_click=make_prediction),
        ft.Text(ref=prediction_result),
    )
    page.update()

if __name__ == "__main__":
    ft.app(flet_app)