import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input, MaxPooling2D, Conv2D, Dropout
import numpy as np
import time
import os
import flet as ft
from PIL import Image


TUMOR_CLASSES = {
    0: "нет опухоли",
    1: "глиома",
    2: "менингиома",
    3: "питуитарная"
}


def create_model(num_classes):
    model = tf.keras.Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def predict(model, image_patch):
    img = Image.open(image_patch)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = tf.argmax(prediction, axis=1)[0].numpy()
    confidence = np.max(prediction)

    result = TUMOR_CLASSES.get(int(predicted_class), "неизвестная опухоль")
    return result, confidence


def train_model(data_dir):
    start_time = time.time()

    image_size = (224, 224)
    batch_size = 32

    if not os.path.exists(data_dir):
        return f"директория не найдена: {data_dir}"

    subdirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    if not subdirs:
        return f"не найдено дочерних директорий в {data_dir}"

    print(f"найдено {len(subdirs)} дочерних директорий {data_dir}")
    for subdir in subdirs:
        print(f"- {os.path.basename(subdir)}")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
    )

    num_classes = len(subdirs)
    model = create_model(num_classes)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=3, factor=0.5,
                                                     min_lr=1e-6, verbose=2)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy',
                                                          save_best_only=True, mode='max')

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        epochs=100,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
    )

    accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    end_time = time.time()
    training_time = end_time - start_time

    best_model = tf.keras.models.load_model('best_model.keras')
    best_model.save('brain_tumor_model.keras')
    return f"тренировочное время: {training_time:.2f} секунд, точность: {accuracy:.2f}, валидационная точность: {val_accuracy:.2f}, потеря: {loss:.4f}, валидационная потеря: {val_loss:.4f}"


def process_image(image, label, num_classes):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def flet_app(page:ft.Page):
    page.title = "BrainTumor"

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
        if 'trained_model' not in globals():
            try:
                trained_model = keras.models.load_model('brain_tumor_model.keras')
            except:
                prediction_result.current.value = "модель не найдена"

        if not image_patch.current.value:
            prediction_result.current.value = "укажите путь к изображению"
        else:
            try:
                result, confidence = predict(trained_model, image_patch.current.value)
                prediction_result.current.value = f"результат: {result}\nуверенность: {confidence:.2f}"
            except Exception as ex:
                prediction_result.current.value = f"ошибка предсказания: {str(ex)}"
        page.update()


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