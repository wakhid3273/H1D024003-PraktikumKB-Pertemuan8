# ============================================================
# Klasifikasi Rock Paper Scissors dengan CNN
# ============================================================

# Langkah 3: Import Library
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# Langkah 4: Persiapan Data dengan ImageDataGenerator
# ============================================================
dataset_path = "./rps-cv-images"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)

# ============================================================
# Langkah 5: Membangun Model CNN
# ============================================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

model.summary()

# ============================================================
# Langkah 6: Kompilasi Model
# ============================================================
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ============================================================
# Langkah 7: Pelatihan Model
# ============================================================
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# ============================================================
# Langkah 8: Evaluasi Model
# ============================================================
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')

# ============================================================
# Langkah 9: Prediksi pada Data Validasi
# ============================================================
predictions = model.predict(validation_generator)
print(predictions)  # Output berupa probabilitas prediksi tiap kelas