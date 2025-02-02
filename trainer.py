# Violence Detection AI

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset structure
# /dataset/
# 	/violence/
# 		violence1.jpg
# 		violence2.mp4
# 		...
# 	/non-violence/
# 		non-violence1.mp4
# 		non-violence2.jpg
# 		...

# Define paths
train_dir = 'dataset'
val_dir = 'dataset'

# Create ImageDataGenerator objects
train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create data generators
train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(224, 224),
	batch_size=32,
	class_mode='binary',
	subset='training'
)

val_generator = val_datagen.flow_from_directory(
	val_dir,
	target_size=(224, 224),
	batch_size=32,
	class_mode='binary',
	subset='validation'
)

# Define the model
model = models.Sequential([
	layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
	layers.MaxPooling2D((2, 2)),
	layers.Conv2D(64, (3, 3), activation='relu'),
	layers.MaxPooling2D((2, 2)),
	layers.Conv2D(128, (3, 3), activation='relu'),
	layers.MaxPooling2D((2, 2)),
	layers.Conv2D(128, (3, 3), activation='relu'),
	layers.MaxPooling2D((2, 2)),
	layers.Flatten(),
	layers.Dense(512, activation='relu'),
	layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the model
model.save('trained_model/violence_detection.h5')