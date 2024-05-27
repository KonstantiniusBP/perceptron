import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Создаем модель
model = Sequential()

# Добавляем сверточный слой с 32 фильтрами, размером ядра 3x3 и активацией ReLU
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Добавляем слой пулинга (уменьшает размерность)
model.add(MaxPooling2D(pool_size=(2, 2)))

# Добавляем второй сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Преобразуем 2D-представление в вектор
model.add(Flatten())

# Полносвязные слои
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))  # Выводной слой

# Компилируем модель
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Создаем генератор изображений для обучения
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# Обучаем модель
model.fit_generator(training_set, steps_per_epoch=8000, epochs=25, validation_steps=2000)

# Сохраняем модель
model.save('cat_dog_classifier.h5')
