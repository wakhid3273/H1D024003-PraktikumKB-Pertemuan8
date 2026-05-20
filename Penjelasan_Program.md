# Penjelasan Program Klasifikasi Rock Paper Scissors dengan CNN

## 📋 Ringkasan Program
Program ini adalah implementasi **Convolutional Neural Network (CNN)** untuk mengklasifikasi tiga jenis hand gesture: **Rock (Batu), Paper (Kertas), dan Scissors (Gunting)** menggunakan dataset gambar berkualitas tinggi.

---

## 🏗️ Langkah-Langkah Program

### **Langkah 1: Import Library**
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
**Penjelasan:**
- **numpy & pandas**: Untuk manipulasi data numerik dan tabel
- **tensorflow & keras**: Framework deep learning untuk membangun model neural network
- **Conv2D, MaxPooling2D**: Layer khusus untuk CNN yang mendeteksi fitur visual
- **ImageDataGenerator**: Untuk preprocessing dan augmentasi data gambar

---

### **Langkah 2: Persiapan Data**
```python
dataset_path = "./rps-cv-images"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
```
**Penjelasan:**
- **rescale=1./255**: Normalisasi nilai pixel dari range [0, 255] menjadi [0, 1]
- **validation_split=0.2**: Membagi data menjadi 80% training dan 20% validation
- Dataset terdiri dari 2,188 gambar:
  - Rock: 726 gambar
  - Paper: 710 gambar
  - Scissors: 752 gambar

#### **Training Generator**
```python
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)
```
**Penjelasan:**
- Membaca gambar dari folder yang terstruktur (rock, paper, scissors)
- **target_size=(150, 150)**: Mengubah ukuran semua gambar menjadi 150x150 pixel
- **batch_size=32**: Memproses 32 gambar per iterasi
- **class_mode='categorical'**: One-hot encoding untuk 3 kelas output

#### **Validation Generator**
```python
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)
```
**Penjelasan:**
- Data validasi digunakan untuk mengecek performa model selama training tanpa memodifikasi weight

---

### **Langkah 3: Membangun Arsitektur Model CNN**
```python
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
```
**Penjelasan Setiap Layer:**

| Layer | Fungsi | Detail |
|-------|--------|--------|
| **Conv2D (32 filter)** | Ekstraksi fitur dasar | Menggunakan 32 kernel 3x3 untuk mendeteksi edge, texture |
| **MaxPooling2D** | Reduksi dimensi | Mengambil nilai maksimal dari 2x2 region |
| **Conv2D (64 filter)** | Ekstraksi fitur kompleks | Mendeteksi pola yang lebih kompleks |
| **MaxPooling2D** | Reduksi dimensi | Mengurangi ukuran feature map |
| **Conv2D (128 filter)** | Ekstraksi fitur tingkat tinggi | Mendeteksi pola detail |
| **MaxPooling2D** | Reduksi dimensi | Reduksi akhir sebelum flattening |
| **Flatten** | Konversi 2D ke 1D | Mengubah feature map ke vektor 1D |
| **Dense (512 units)** | Fully connected layer | Pembelajaran non-linear dengan 512 neuron |
| **Dense (3 units)** | Output layer | 3 neuron untuk 3 kelas (rock, paper, scissors) |

**Visualisasi model:**
```
Input: (150, 150, 3) → Conv2D(32) → MaxPooling2D → Conv2D(64) → MaxPooling2D 
→ Conv2D(128) → MaxPooling2D → Flatten → Dense(512) → Dense(3) → Output
```

---

### **Langkah 4: Kompilasi Model**
```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```
**Penjelasan:**
- **loss='categorical_crossentropy'**: Loss function untuk multi-class classification
- **optimizer='adam'**: Algoritma optimisasi dengan learning rate adaptif
- **metrics=['accuracy']**: Metrik evaluasi menggunakan akurasi

---

### **Langkah 5: Pelatihan Model**
```python
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)
```
**Penjelasan:**
- **epochs=10**: Model dilatih sebanyak 10 kali melalui seluruh dataset training
- Setiap epoch:
  - Model melihat semua training data
  - Melakukan forward pass dan backward pass
  - Update weight berdasarkan loss
  - Evaluasi terhadap validation data

**Output Pelatihan:** (Typical output)
```
Epoch 1/10
42/42 [==============================] - 15s 300ms/step - loss: 1.0856 - accuracy: 0.3842 - val_loss: 0.9234 - val_accuracy: 0.5621
Epoch 2/10
42/42 [==============================] - 12s 290ms/step - loss: 0.7234 - accuracy: 0.6234 - val_loss: 0.5123 - val_accuracy: 0.7421
...
Epoch 10/10
42/42 [==============================] - 12s 285ms/step - loss: 0.1245 - accuracy: 0.9523 - val_loss: 0.2456 - val_accuracy: 0.9234
```

---

### **Langkah 6: Evaluasi Model**
```python
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')
```
**Penjelasan:**
- Mengevaluasi performa model pada data validation yang belum pernah dilihat
- **val_loss**: Seberapa jauh prediksi dari target (semakin rendah semakin baik)
- **val_accuracy**: Persentase prediksi yang benar pada data validation

**Output Contoh:**
```
Validation loss: 0.2456, Validation accuracy: 0.9234
```
(92.34% akurasi pada data validation)

---

### **Langkah 7: Prediksi pada Data Validasi**
```python
predictions = model.predict(validation_generator)
print(predictions)
```
**Penjelasan:**
- Model melakukan prediksi pada semua gambar validation
- Output berupa **probabilitas** untuk setiap kelas

**Output Contoh:**
```
[
  [0.01 0.97 0.02],   # Gambar 1: 97% Paper
  [0.89 0.05 0.06],   # Gambar 2: 89% Rock
  [0.03 0.05 0.92],   # Gambar 3: 92% Scissors
  [0.08 0.02 0.90],   # Gambar 4: 90% Scissors
  ...
]
```
Setiap baris = 1 gambar, dengan 3 kolom untuk probabilitas [Rock, Paper, Scissors]

---

## 📊 Output Yang Diharapkan

### 1. **Model Summary**
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 148, 148, 32)     896       
max_pooling2d (MaxPooling2D) (None, 74, 74, 32)       0         
conv2d_1 (Conv2D)            (None, 72, 72, 64)       18496     
max_pooling2d_1 (MaxPooling) (None, 36, 36, 64)       0         
conv2d_2 (Conv2D)            (None, 34, 34, 128)      73856     
max_pooling2d_2 (MaxPooling) (None, 17, 17, 128)      0         
flatten (Flatten)            (None, 36992)            0         
dense (Dense)                (None, 512)              18940416  
dense_1 (Dense)              (None, 3)                1539      
=================================================================
Total params: 19,035,199
Trainable params: 19,035,199
Non-trainable params: 0
```

### 2. **Training History**
```
- Loss menurun dari ~1.0 menjadi ~0.1-0.3
- Accuracy naik dari ~33% (random) menjadi ~92-95%
- Validation loss dan accuracy menunjukkan trend serupa
```

### 3. **Validation Metrics**
```
Validation loss: 0.24-0.35
Validation accuracy: 0.90-0.95 (90-95%)
```

### 4. **Prediction Output**
```
Array dengan shape (N, 3) dimana N = jumlah gambar validation
Setiap baris: [prob_rock, prob_paper, prob_scissors]
Nilai: 0-1 (probabilitas)
Jumlah per baris: 1.0
```

---

## 🎯 Interpretasi Hasil

✅ **Model Berhasil Jika:**
- Validation accuracy > 85%
- Tidak ada overfitting signifikan (training loss ≈ validation loss)
- Prediksi konsisten dan confident (satu probabilitas mendominasi)

❌ **Masalah Umum:**
- Validation accuracy < 70%: Model kurang terlatih, tambah epochs atau layer
- Overfitting: Training acc 95%, validation acc 60%: Gunakan regularization
- Underfitting: Semua accuracy rendah: Model terlalu sederhana

---

## 📁 Dataset Information

**Sumber:** Rock-Paper-Scissors Computer Vision Dataset
- **Format:** RGB PNG images, 300x200 pixel original
- **Preprocessing:** Diubah ke 150x150 pixel saat training
- **Total:** 2,188 gambar terbagi rata di 3 kelas
- **Background:** Green screen dengan lighting konsisten
- **Author:** Julien de la Bruère-Terreault (CC-BY-SA 4.0)

---

## 🔧 Tips Optimisasi

1. **Untuk Akurasi Lebih Tinggi:**
   - Tambah epochs dari 10 menjadi 20-30
   - Gunakan data augmentation (rotasi, flip, zoom)
   - Tambah filter CNN atau layer tambahan

2. **Untuk Mengurangi Waktu Training:**
   - Kurangi batch_size dari 32 menjadi 16
   - Kurangi epochs
   - Gunakan GPU (jika tersedia)

3. **Untuk Menghindari Overfitting:**
   - Tambahkan Dropout layer
   - Gunakan L1/L2 regularization
   - Data augmentation yang lebih aggressive

---

**Program ini berhasil mengimplementasikan deep learning untuk image classification dengan arsitektur CNN yang efektif.**
