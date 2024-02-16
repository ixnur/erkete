import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print("Tensorflow verisi: ", tf.__version__)
print("NumPy verisi: ", np.__version__)
print("Matplotlib versiyonu: ", plt.matplotlib.__version__)

train_data_dir = 'eğitim'
test_data_dir = 'test'

img_width, img_height = 224, 224#veri boyutu
batch_size = 32
epochs = 10
num_classes = 8#sınıf sayısı
input_shape = (img_width, img_height, 3)#resim boyutu


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
#rescale: veri boyutunu 0-1 arasına çevirir
#rotation_range: veri döndürür
#width_shift_range: veri hareket eder


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['egitim', 'silah', 'bıçak', 'tabanca', 'tornavida', 'benzen']
)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['egitim', 'silah', 'bıçak', 'tabanca', 'tornavida', 'benzen']
)
base_model = MobileNetV2(
    input_shape=(img_width, img_height, 3),
    weights='imagenet',
    include_top=False
)
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(2048, activation='relu')(x)
x = layers.Dropout(0.5)(x)  
x = layers.Dense(1024, activation='relu')(x)  
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)  
x = layers.Dense(256, activation='relu')(x)  
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)  
x = layers.Dense(64, activation='relu')(x)  
x = layers.Dropout(0.5)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.5)(x)  
x = layers.Dense(16, activation='relu')(x)  
x = layers.Dropout(0.5)(x)
x = layers.Dense(8, activation='relu')(x)
x = layers.Dropout(0.5)(x)  
x = layers.Dense(4, activation='relu')(x)  
x = layers.Dropout(0.5)(x)
x = layers.Dense(2, activation='relu')(x)
x = layers.Dropout(0.5)(x)  
x = layers.Dense(1, activation='relu')(x)  
x = layers.Dropout(0.5)(x)

predictions = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=predictions)
print("Model test ediliyor...")
opt = optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)
print("Model test edildi.")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

print("Model eğitiliyor...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
print(len(acc))
print(len(history.history['accuracy']))
print(len(history.history['loss']))
print(len(history.history['val_accuracy']))
print(len(history.history['val_loss']))
print(epochs_range)
print(len(epochs_range))

model.save('my_model.h5')

print("Model kaydedildi.")
print("Model eğitimi tamamlandı.")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Doğruluğu')
plt.plot(epochs_range, val_acc, label='Doğrulama Doğruluğu')
plt.legend(loc='Alt Sağ')
plt.title('Eğitim ve Doğrulama')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='eğitim kaybı') 
plt.plot(epochs_range, val_loss, label='doğrulama kaybı')
plt.legend(loc='Sağ üst')
plt.title('Eğitim ve Doğrulama kaybı')
plt.show()
