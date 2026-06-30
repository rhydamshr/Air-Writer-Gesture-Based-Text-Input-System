import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data loading
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    'data',
    target_size=(512, 128),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'data',
    target_size=(512, 128),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# Model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(512, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=10)

model.save("airchar_HI_model.h5")
